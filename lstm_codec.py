import os
os.environ["PYTHONHASHSEED"] = "1234"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import time
import math
import random
import contextlib
import torch
import torch.nn as nn
import numpy as np
from contextlib import closing
import traceback
from verify import verify_files
from ArithmeticCoder import (
    ArithmeticEncoder,
    ArithmeticDecoder,
    BitOutputStream,
    BitInputStream
)
DEFAULT_PARAMS = {
    "seq_length": 13,
    "num_layers": 1,
    "rnn_units": 768,
    "embedding_size": 150,
    "start_learning_rate": 0.005,
    "end_learning_rate": 0.0001,
    "batch_size": 128
}
params = DEFAULT_PARAMS.copy()
if os.path.exists("params.json"):
    with open("params.json", "r", encoding="utf-8") as f:
        params.update(json.load(f))

seq_length = int(params["seq_length"])
num_layers = int(params["num_layers"])
rnn_units = int(params["rnn_units"])
batch_size = int(params["batch_size"])
embedding_size = int(params["embedding_size"])
start_learning_rate = float(params["start_learning_rate"])
end_learning_rate = float(params["end_learning_rate"])

mode = "both"
FREQ_SCALE = 16000000
SEED = 1234

path_to_file = "data/enwik5"
path_to_compressed = path_to_file + "_compressed.dat"
path_to_decompressed = path_to_file + "_decompressed.dat"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reset_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def reset_gpu_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def get_peak_gpu_memory_gb():
    if not torch.cuda.is_available():
        return None
    try:
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        return max(allocated, reserved) / (1024.0 ** 3)
    except Exception:
        return None

class NeuralByteModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, rnn_units, num_layers):
        super().__init__()

        self.vocab_size = int(vocab_size)
        self.embedding_size = int(embedding_size)
        self.rnn_units = int(rnn_units)
        self.num_layers = int(num_layers)

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        self.rnn = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.rnn_units,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.0
        )

        self.output = nn.Linear(self.rnn_units, self.vocab_size)
        self.log_tau = nn.Parameter(torch.zeros(()))

    def forward(self, x, hidden):
        emb = self.embedding(x)
        out, new_hidden = self.rnn(emb, hidden)
        last = out[:, -1, :]

        logits = self.output(last)

        log_tau = torch.clamp(
            self.log_tau,
            min=math.log(0.9),
            max=math.log(1.1)
        )

        tau = torch.exp(log_tau)
        logits = logits / tau

        return logits, new_hidden

    def init_hidden(self, batch_size_value):
        h = torch.zeros(
            self.num_layers,
            batch_size_value,
            self.rnn_units,
            device=device
        )

        c = torch.zeros(
            self.num_layers,
            batch_size_value,
            self.rnn_units,
            device=device
        )

        return h, c


def detach_hidden(hidden):
    h, c = hidden
    return h.detach(), c.detach()


def set_learning_rate(optimizer, step, total_steps):
    total_steps = max(1, int(total_steps))
    step = min(max(0, int(step)), total_steps)

    progress = float(step) / float(total_steps)
    lr = end_learning_rate + (start_learning_rate - end_learning_rate) * (1.0 - progress)

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr

def make_freqs_from_probs_1d(probs_1d):
    counts = (probs_1d * float(FREQ_SCALE) + 1.0).astype(np.int64)
    counts[counts < 1] = 1

    freqs = np.cumsum(counts, dtype=np.int64)

    if freqs[-1] <= 0:
        freqs = np.arange(1, len(probs_1d) + 1, dtype=np.int64)

    return freqs

def make_uniform_freqs(vocab_size):
    probs = np.full((int(vocab_size),), 1.0 / float(vocab_size), dtype=np.float64)
    return make_freqs_from_probs_1d(probs)

def get_symbol(index, length, freq, coder, compress, data):
    symbol = 0

    if index < length:
        if compress:
            symbol = int(data[index])
            coder.write(freq, symbol)
        else:
            symbol = int(coder.read(freq))
            data[index] = symbol

    return symbol

def train_step(pos, seq_input, length, vocab_size, coder, model, optimizer, compress, data, hidden):
    split = math.ceil(length / batch_size)

    model.train()

    optimizer.zero_grad(set_to_none=True)
    set_learning_rate(optimizer, pos, split)

    logits, new_hidden = model(seq_input, hidden)
    probs = torch.softmax(logits, dim=1)
    probs_np = probs.detach().cpu().numpy()

    symbols = []
    mask = []

    for i in range(batch_size):
        index = pos + 1 + i * split
        freq = make_freqs_from_probs_1d(probs_np[i])
        symbol = get_symbol(index, length, freq, coder, compress, data)

        symbols.append(symbol)

        if index < length:
            mask.append(1.0)
        else:
            mask.append(0.0)

    symbols_tensor = torch.tensor(symbols, dtype=torch.long, device=device)
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)

    selected_probs = probs.gather(1, symbols_tensor.view(-1, 1)).view(-1)
    selected_probs = torch.clamp(selected_probs, min=1e-7, max=1.0)

    loss = torch.sum(-torch.log(selected_probs) * mask_tensor)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
    optimizer.step()

    next_col = symbols_tensor.view(batch_size, 1)
    seq_input = torch.cat([seq_input[:, 1:], next_col], dim=1).detach()
    hidden = detach_hidden(new_hidden)

    denom = int(mask_tensor.sum().item())

    if denom > 0:
        cross_entropy = float(torch.sum(torch.log2(selected_probs.detach()) * mask_tensor).item())
    else:
        cross_entropy = 0.0

    return seq_input, hidden, cross_entropy, denom

def process(compress, length, vocab_size, coder, data):
    reset_seed()

    model = NeuralByteModel(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        rnn_units=rnn_units,
        num_layers=num_layers
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=start_learning_rate,
        betas=(0.0, 0.9999),
        eps=1e-5
    )

    split = math.ceil(length / batch_size)

    uniform_freq = make_uniform_freqs(vocab_size)

    symbols = []

    for i in range(batch_size):
        index = i * split
        symbol = get_symbol(index, length, uniform_freq, coder, compress, data)
        symbols.append(symbol)

    seq_input = torch.tensor(symbols, dtype=torch.long, device=device).view(batch_size, 1)
    seq_input = seq_input.repeat(1, seq_length)

    hidden = model.init_hidden(batch_size)

    pos = 0
    cross_entropy = 0.0
    denom = 0

    while pos < split:
        seq_input, hidden, ce, d = train_step(
            pos=pos,
            seq_input=seq_input,
            length=length,
            vocab_size=vocab_size,
            coder=coder,
            model=model,
            optimizer=optimizer,
            compress=compress,
            data=data,
            hidden=hidden
        )

        cross_entropy += ce
        denom += d
        pos += 1

    if compress:
        coder.finish()

def compression():
    raw = open(path_to_file, "rb").read()
    vocab = sorted(set(raw))

    vocab_size = max(1, len(vocab))

    char2idx = {b: i for i, b in enumerate(vocab)}
    int_list = [char2idx[b] for b in raw]

    length = len(int_list)

    with open(path_to_compressed, "wb") as out:
        out.write(length.to_bytes(5, byteorder="big", signed=False))

        with closing(BitOutputStream(out)) as bitout:
            vocab_set = set(vocab)

            for i in range(256):
                bitout.write(1 if i in vocab_set else 0)

            encoder = ArithmeticEncoder(32, bitout)

            process(
                compress=True,
                length=length,
                vocab_size=vocab_size,
                coder=encoder,
                data=int_list
            )

def decompression():
    with open(path_to_compressed, "rb") as inp:
        length = int.from_bytes(inp.read(5), byteorder="big", signed=False)

        with open(path_to_decompressed, "wb") as out:
            with closing(BitInputStream(inp)) as bitin:
                vocab = []

                for i in range(256):
                    bit = bitin.read()

                    if bit == 1:
                        vocab.append(i)

                vocab_size = max(1, len(vocab))
                output = [0] * length

                decoder = ArithmeticDecoder(32, bitin)

                process(
                    compress=False,
                    length=length,
                    vocab_size=vocab_size,
                    coder=decoder,
                    data=output
                )

                idx2char = np.array(vocab, dtype=np.uint8)

                for i in range(length):
                    out.write(bytes((int(idx2char[output[i]]),)))


def write_result(success, compression_ratio=None, time_spent=None, error=None, peak_gpu_memory_gb=None):
    result = {
        "success": bool(success),
        "compression_ratio": compression_ratio,
        "time_spent": round(float(time_spent), 2) if time_spent is not None else None,
        "peak_gpu_memory_gb": round(float(peak_gpu_memory_gb), 4) if peak_gpu_memory_gb is not None else None
    }

    if error is not None:
        result["error"] = str(error)

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)


def main():
    start = time.time()

    try:
        reset_seed()
        reset_gpu_memory_stats()

        if mode == "compress" or mode == "both":
            print('Encoding started')
            compression()

        if mode == "decompress" or mode == "both":
            print('Decoding started')
            decompression()
            verified_size = verify_files(path_to_file, path_to_decompressed)
            print(f'Verification passed: all {verified_size} bytes match exactly.')

        original_size = os.path.getsize(path_to_file)
        compressed_size = os.path.getsize(path_to_compressed)

        if compressed_size <= 0:
            raise RuntimeError("Compressed file is empty")

        compression_ratio = original_size / compressed_size
        elapsed = time.time() - start
        peak_gpu_memory_gb = get_peak_gpu_memory_gb()

        if peak_gpu_memory_gb is None:
            gpu_mem_text = "N/A (CPU)"
        else:
            gpu_mem_text = f"{peak_gpu_memory_gb:.2f} GB"

        print(f'Compression ratio: {compression_ratio:.2f}, Time: {elapsed:.2f}, GPU memory: {gpu_mem_text}')
      
        write_result(
            success=True,
            compression_ratio=round(float(compression_ratio), 4),
            time_spent=elapsed,
            peak_gpu_memory_gb=peak_gpu_memory_gb
        )

    except Exception as e:
        elapsed = time.time() - start

        write_result(
            success=False,
            compression_ratio=None,
            time_spent=elapsed,
            error=str(e),
            peak_gpu_memory_gb=get_peak_gpu_memory_gb()
        )

        raise

if __name__ == "__main__":
    #import faulthandler
    #faulthandler.enable(open("faulthandler.log", "w"))

    #try:
    main()
    #except Exception:
    #    traceback.print_exc()
    #    raise
