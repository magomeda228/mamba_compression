"""Small Mamba-2 blocks with an autograd-compatible PyTorch SSD scan.

Uses scalar A per head, input-dependent B/C and dt, depthwise causal convolution,
gated RMS normalization and persistent convolution/SSM state. This is a native
PyTorch implementation, not the official fused CUDA implementation.

Architecture references:
https://arxiv.org/abs/2405.21060
https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2_simple.py
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, values):
        scale = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        return values * scale * self.weight


def ssd_scan(x, dt, a, b, c, state):
    """Exact causal SSD recurrence over a short window, including initial state.

    x: [batch, time, heads, head_dim]; dt: [batch, time, heads].
    a: [heads]; b/c: [batch, time, state_dim] (shared across heads).
    state: [batch, heads, head_dim, state_dim].
    """
    steps = x.shape[1]
    if steps == 1:
        # Same SSD recurrence, without materializing a window attention matrix.
        decay = (dt[:, 0] * a).exp()[:, :, None, None]
        injected = x[:, 0, :, :, None] * dt[:, 0, :, None, None] * b[:, 0, None, None, :]
        next_state = state * decay + injected
        output = (next_state * c[:, 0, None, None, :]).sum(-1)
        return output[:, None], next_state
    transition = (dt * a).transpose(1, 2)  # batch, heads, time
    positions = torch.arange(steps, device=x.device)
    lower = positions[:, None] > positions[None, :]
    # For output t and source s, accumulate A[s+1] ... A[t].
    segments = torch.where(lower, transition.unsqueeze(-1), 0.0).cumsum(dim=-2)
    decay = segments.masked_fill(positions[:, None] < positions[None, :], -torch.inf).exp()

    injected = x * dt.unsqueeze(-1)
    content = torch.einsum("btn,bsn->bts", c, b)
    output = torch.einsum("bhts,bshp->bthp", decay * content[:, None], injected)

    prefix = transition.cumsum(dim=-1).transpose(1, 2)
    initial_output = torch.einsum("btn,bhpn->bthp", c, state)
    output = output + initial_output * prefix.exp().unsqueeze(-1)

    tail_decay = (prefix[:, -1:, :] - prefix).exp()
    final_state = state * prefix[:, -1, :].exp()[:, :, None, None]
    final_state = final_state + torch.einsum("bth,btn,bthp->bhpn", tail_decay, b, injected)
    return output, final_state


class Mamba2Block(nn.Module):
    def __init__(self, d_model=128, d_state=16, expand=2, head_dim=32, d_conv=4):
        super().__init__()
        self.inner = d_model * expand
        if self.inner % head_dim:
            raise ValueError("d_model * expand must be divisible by head_dim")
        self.heads = self.inner // head_dim
        self.head_dim = head_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_channels = self.inner + 2 * d_state

        self.pre_norm = RMSNorm(d_model)
        self.project = nn.Linear(d_model, self.inner + self.conv_channels + self.heads, bias=False)
        self.conv = nn.Conv1d(
            self.conv_channels,
            self.conv_channels,
            d_conv,
            groups=self.conv_channels,
            bias=True,
        )
        log_dt = torch.empty(self.heads).uniform_(math.log(0.001), math.log(0.1))
        dt = log_dt.exp()
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.a_log = nn.Parameter(torch.empty(self.heads).uniform_(1, 16).log())
        self.skip = nn.Parameter(torch.ones(self.heads))
        self.gate_norm = RMSNorm(self.inner)
        self.out = nn.Linear(self.inner, d_model, bias=False)

    def init_state(self, batch_size):
        parameter = self.project.weight
        return (
            parameter.new_zeros(batch_size, self.conv_channels, self.d_conv - 1),
            parameter.new_zeros(batch_size, self.heads, self.head_dim, self.d_state),
        )

    def forward(self, values, state):
        history, memory = state
        gate, conv_input, dt_input = self.project(self.pre_norm(values)).split(
            [self.inner, self.conv_channels, self.heads], dim=-1
        )
        conv_sequence = torch.cat((history, conv_input.transpose(1, 2)), dim=-1)
        if values.shape[1] == 1:
            # Depthwise convolution over exactly one causal receptive field.
            features = (conv_sequence * self.conv.weight[:, 0]).sum(-1) + self.conv.bias
            features = F.silu(features[:, None])
        else:
            features = F.silu(self.conv(conv_sequence).transpose(1, 2))
        new_history = conv_sequence[:, :, -(self.d_conv - 1) :] if self.d_conv > 1 else history
        x, b, c = features.split([self.inner, self.d_state, self.d_state], dim=-1)
        x = x.reshape(*x.shape[:2], self.heads, self.head_dim)
        dt = F.softplus(dt_input + self.dt_bias)
        output, new_memory = ssd_scan(x, dt, -self.a_log.exp(), b, c, memory)
        output = output + x * self.skip[None, None, :, None]
        output = output.flatten(-2) * F.silu(gate)
        output = self.out(self.gate_norm(output))
        return values + output, (new_history, new_memory)


class NeuralByteModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=1, **block_params):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [Mamba2Block(d_model=d_model, **block_params) for _ in range(num_layers)]
        )
        self.final_norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.log_tau = nn.Parameter(torch.zeros(()))

    def init_hidden(self, batch_size):
        return [block.init_state(batch_size) for block in self.blocks]

    def forward(self, symbols, hidden):
        values = self.embedding(symbols)
        new_hidden = []
        for block, state in zip(self.blocks, hidden, strict=True):
            values, next_state = block(values, state)
            new_hidden.append(next_state)
        logits = self.output(self.final_norm(values[:, -1]))
        tau = self.log_tau.clamp(math.log(0.9), math.log(1.1)).exp()
        return logits / tau, new_hidden


def detach_hidden(hidden):
    # Переносим числовую память между шагами, но не весь граф предыдущего обучения.
    return [(history.detach(), memory.detach()) for history, memory in hidden]
