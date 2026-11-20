"""Optional CUDA acceleration; imported only when the environment supports it."""

import functools

import torch

import codec_core as codec
from mamba2 import NeuralByteModel

ORIGINAL_ADAM = torch.optim.Adam


class CompiledModel(NeuralByteModel):
    """Compile the NN pass while retaining recurrent state across CUDA Graph replays."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compiled_forward = torch.compile(
            super().forward, fullgraph=True, dynamic=False, mode="reduce-overhead"
        )

    def forward(self, symbols, hidden):
        torch.compiler.cudagraph_mark_step_begin()
        logits, new = self.compiled_forward(symbols, hidden)
        # Граф повторно использует свои буферы: состояние следующего шага хранится отдельно.
        return logits.clone(), [(conv.clone(), state.clone()) for conv, state in new]


def configure(backend):
    """Enable acceleration only for our codec, after the original LSTM has finished."""
    import fast_memory

    codec.ContinuationMemory = fast_memory.ContinuationMemory
    codec.make_freqs_from_probs_1d = fast_memory.frequencies
    # Keep the model's ordinary PyTorch RMSNorm. torch.compile handles fusion;
    # this backend contains no hand-written Triton or CuTe kernels.
    codec.NeuralByteModel = CompiledModel
    codec.torch.optim.Adam = (
        functools.partial(ORIGINAL_ADAM, fused=True)
        if backend == "compiled_fused"
        else ORIGINAL_ADAM
    )
    # New signatures: old archives require the removed custom RMSNorm kernel.
    codec.ARCHIVE_MAGIC = b"FSB2" if backend == "compiled_fused" else b"FCP2"
