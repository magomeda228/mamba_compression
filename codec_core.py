import os

os.environ["PYTHONHASHSEED"] = "1234"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import hashlib
import json
import math
import random
import time
from contextlib import closing

import numpy as np
import torch

from ArithmeticCoder import ArithmeticDecoder, ArithmeticEncoder, BitInputStream, BitOutputStream
from context_memory import ContinuationMemory
from mamba2 import NeuralByteModel, detach_hidden
from verify import verify_files

DEFAULT_PARAMS = {
    "seq_length": 1,
    "num_layers": 1,
    "d_model": 64,
    "d_state": 8,
    "expand": 2,
    "head_dim": 16,
    "d_conv": 4,
    "start_learning_rate": 0.005,
    "end_learning_rate": 0.0001,
    "batch_size": 128,
    "train_updates": 1,
    "beta1": 0.0,
    "beta2": 0.9999,
    "adam_eps": 1e-5,
    "expert_mode": "mix",
    "memory_order": 6,
    "initial_neural_weight": 0.5,
    "mixture_rate": 0.01,
    "escape_scale": 4.0,
}
ARCHIVE_MAGIC = b"SSM1"


def apply_params(values):
    global params, seq_length, batch_size, start_learning_rate, end_learning_rate
    unknown = set(values) - set(DEFAULT_PARAMS)
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}")
    params = {**DEFAULT_PARAMS, **values}
    if params["expert_mode"] not in ("mamba", "memory", "mix"):
        raise ValueError("expert_mode must be mamba, memory or mix")
    floats = {
        "start_learning_rate",
        "end_learning_rate",
        "beta1",
        "beta2",
        "adam_eps",
        "initial_neural_weight",
        "mixture_rate",
        "escape_scale",
    }
    for key in DEFAULT_PARAMS:
        if key == "expert_mode":
            continue
        if key in floats:
            params[key] = float(params[key])
        else:
            if isinstance(params[key], bool) or int(params[key]) != params[key]:
                raise ValueError(f"{key} must be an integer")
            params[key] = int(params[key])
        if not math.isfinite(params[key]):
            raise ValueError(f"{key} must be finite")
        if key in ("beta1", "beta2"):
            if not 0 <= params[key] < 1:
                raise ValueError("Adam betas must be in [0, 1)")
        elif key == "memory_order":
            if not 0 <= params[key] <= 220:
                raise ValueError("memory_order must be between 0 and 220")
        elif key == "escape_scale":
            if params[key] < 0:
                raise ValueError("escape_scale must be nonnegative")
        elif params[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 < params["initial_neural_weight"] < 1:
        raise ValueError("initial_neural_weight must be between 0 and 1")
    if params["seq_length"] != 1:
        raise ValueError("StateSMix processes one new byte per step; seq_length must be 1")
    if params["d_model"] * params["expand"] % params["head_dim"]:
        raise ValueError("d_model * expand must be divisible by head_dim")
    seq_length = params["seq_length"]
    batch_size = params["batch_size"]
    start_learning_rate = params["start_learning_rate"]
    end_learning_rate = params["end_learning_rate"]


apply_params({})
if os.path.exists("params.json"):
    with open("params.json", encoding="utf-8") as f:
        apply_params(json.load(f))

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

    # Supported by the required PyTorch versions; report setup errors instead
    # of silently disabling the encoder/decoder's determinism configuration.
    torch.use_deterministic_algorithms(True, warn_only=True)


def reset_gpu_memory_stats():
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def get_peak_gpu_memory_gb():
    if device.type != "cuda":
        return None
    try:
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        return max(allocated, reserved) / (1024.0**3)
    except Exception:
        return None


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


def train_step(
    pos, seq_input, length, vocab_size, coder, model, optimizer, compress, data, hidden, memory
):
    """Predict from known history, transfer bytes, then train on the revealed targets."""
    split = math.ceil(length / batch_size)
    probs = None
    if model is not None:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        set_learning_rate(optimizer, pos, split)
        logits, next_hidden = model(seq_input, hidden)
        probs = torch.softmax(logits, dim=1)
        mamba_probabilities = probs.detach().cpu().numpy()
    symbols, mask = [], []
    encoded_log_likelihood = 0.0
    for stream in range(batch_size):
        # Обход цепочек одинаков при кодировании и декодировании; это не порядок
        # непрерывного префикса исходного файла.
        index = pos + 1 + stream * split
        if index >= length:
            symbols.append(0)
            mask.append(0.0)
            continue
        local_mamba = mamba_probabilities[stream] if model is not None else None
        local_memory = memory.predict(stream) if memory is not None else None
        if params["expert_mode"] == "mamba":
            probabilities = local_mamba
        elif params["expert_mode"] == "memory":
            probabilities = local_memory
        else:
            probabilities = memory.combine(local_mamba, local_memory)
        freqs = make_freqs_from_probs_1d(probabilities)
        symbol = get_symbol(index, length, freqs, coder, compress, data)
        symbols.append(symbol)
        mask.append(1.0)
        encoded_log_likelihood += math.log2(max(float(probabilities[symbol]), 1e-12))
        if memory is not None:
            if params["expert_mode"] == "mix":
                memory.observe(stream, symbol, local_mamba[symbol], local_memory[symbol])
            else:
                memory.observe(stream, symbol)

    if model is None:
        return None, None, encoded_log_likelihood, int(sum(mask))
    targets = torch.tensor(symbols, dtype=torch.long, device=device)
    # Пустые позиции последнего пакета не должны участвовать в функции потерь.
    valid = torch.tensor(mask, dtype=torch.float32, device=device)
    denominator = valid.sum().clamp_min(1.0)
    for update in range(params["train_updates"]):
        if update:
            optimizer.zero_grad(set_to_none=True)
            logits, next_hidden = model(seq_input, hidden)
            probs = torch.softmax(logits, dim=1)
        selected = probs.gather(1, targets[:, None]).flatten().clamp_min(1e-9)
        loss = (-selected.log() * valid).sum() / denominator
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
    seq_input = torch.cat((seq_input[:, 1:], targets[:, None]), dim=1).detach()
    return seq_input, detach_hidden(next_hidden), encoded_log_likelihood, int(sum(mask))


def process(compress, length, vocab_size, coder, data):
    reset_seed()

    model, optimizer = None, None
    if params["expert_mode"] != "memory":
        model = NeuralByteModel(
            vocab_size=vocab_size,
            d_model=params["d_model"],
            num_layers=params["num_layers"],
            d_state=params["d_state"],
            expand=params["expand"],
            head_dim=params["head_dim"],
            d_conv=params["d_conv"],
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=start_learning_rate,
            betas=(params["beta1"], params["beta2"]),
            eps=params["adam_eps"],
        )
    memory = None
    if params["expert_mode"] != "mamba":
        memory = ContinuationMemory(
            vocab_size,
            batch_size,
            params["memory_order"],
            params["initial_neural_weight"],
            params["mixture_rate"],
            params["escape_scale"],
        )

    split = math.ceil(length / batch_size)

    uniform_freq = make_uniform_freqs(vocab_size)

    symbols = []

    for i in range(batch_size):
        index = i * split
        symbol = get_symbol(index, length, uniform_freq, coder, compress, data)
        symbols.append(symbol)
        if memory is not None and index < length:
            memory.observe(i, symbol)

    seq_input = None
    if model is not None:
        seq_input = torch.tensor(symbols, dtype=torch.long, device=device).view(batch_size, 1)
        seq_input = seq_input.repeat(1, seq_length)
    hidden = model.init_hidden(batch_size) if model is not None else None

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
            hidden=hidden,
            memory=memory,
        )

        cross_entropy += ce
        denom += d
        pos += 1
        if pos % 100 == 0:
            phase = "Encoding" if compress else "Decoding"
            print(f"{phase}: {pos}/{split}", flush=True)

    process_stats["encoding" if compress else "decoding"] = {
        "prediction_steps": split,
        "optimizer_steps": split * params["train_updates"] if model is not None else 0,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters())
        if model is not None
        else 0,
        "stateful_single_byte": model is not None,
        "memory": memory.stats() if memory is not None else None,
    }
    if compress:
        coder.finish()


def compression():
    """Write replay settings, original checksum/alphabet, and the arithmetic payload."""
    with open(path_to_file, "rb") as original:
        raw = original.read()
    vocab = sorted(set(raw))

    vocab_size = max(1, len(vocab))

    char2idx = {b: i for i, b in enumerate(vocab)}
    int_list = [char2idx[b] for b in raw]

    length = len(int_list)

    with open(path_to_compressed, "wb") as out:
        config = json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
        out.write(ARCHIVE_MAGIC)
        out.write(len(config).to_bytes(2, "big"))
        out.write(config)
        out.write(length.to_bytes(5, byteorder="big", signed=False))
        out.write(hashlib.sha256(raw).digest())

        with closing(BitOutputStream(out)) as bitout:
            vocab_set = set(vocab)

            for i in range(256):
                bitout.write(1 if i in vocab_set else 0)

            encoder = ArithmeticEncoder(32, bitout)

            process(
                compress=True, length=length, vocab_size=vocab_size, coder=encoder, data=int_list
            )


def decompression():
    """Recreate the model from the archive and verify the restored byte checksum."""
    with open(path_to_compressed, "rb") as inp:
        if inp.read(4) != ARCHIVE_MAGIC:
            raise ValueError(f"Archive backend mismatch: expected signature {ARCHIVE_MAGIC!r}")
        size_bytes = inp.read(2)
        if len(size_bytes) != 2:
            raise ValueError("Truncated archive header")
        config_size = int.from_bytes(size_bytes, "big")
        config = inp.read(config_size)
        if len(config) != config_size:
            raise ValueError("Truncated Mamba configuration")
        # Настройки из архива имеют приоритет над выбранным профилем запуска.
        apply_params(json.loads(config))
        length_bytes = inp.read(5)
        if len(length_bytes) != 5:
            raise ValueError("Truncated original length")
        length = int.from_bytes(length_bytes, "big")
        expected_digest = inp.read(32)
        if len(expected_digest) != 32:
            raise ValueError("Truncated SHA-256 checksum")

        with open(path_to_decompressed, "wb") as out, closing(BitInputStream(inp)) as bitin:
            vocab = []

            for i in range(256):
                bit = bitin.read()
                if bit == -1:
                    raise ValueError("Truncated alphabet")

                if bit == 1:
                    vocab.append(i)

            vocab_size = max(1, len(vocab))
            output = [0] * length

            decoder = ArithmeticDecoder(32, bitin)

            process(
                compress=False, length=length, vocab_size=vocab_size, coder=decoder, data=output
            )

            idx2char = np.array(vocab, dtype=np.uint8)

            for i in range(length):
                out.write(bytes((int(idx2char[output[i]]),)))

    with open(path_to_decompressed, "rb") as restored:
        actual_digest = hashlib.sha256(restored.read()).digest()
    if actual_digest != expected_digest:
        raise RuntimeError("Decompressed SHA-256 does not match the archive checksum")


def write_result(
    success, compression_ratio=None, time_spent=None, error=None, peak_gpu_memory_gb=None
):
    result = {
        "success": bool(success),
        "model": "StateSMix-random-init-Mamba2-ngram",
        "device": str(device),
        "params": params.copy(),
        "process_stats": process_stats.copy(),
        "original_size": os.path.getsize(path_to_file)
        if os.path.exists(path_to_file)
        else (
            os.path.getsize(path_to_decompressed) if os.path.exists(path_to_decompressed) else None
        ),
        "compressed_size": os.path.getsize(path_to_compressed)
        if os.path.exists(path_to_compressed)
        else None,
        "compression_ratio": compression_ratio,
        "time_spent": round(float(time_spent), 2) if time_spent is not None else None,
        "peak_gpu_memory_gb": round(float(peak_gpu_memory_gb), 4)
        if peak_gpu_memory_gb is not None
        else None,
    }

    if error is not None:
        result["error"] = str(error)

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)


process_stats = {}


def main():
    start = time.perf_counter()
    process_stats.clear()

    try:
        reset_seed()
        reset_gpu_memory_stats()
        print(
            f"Device: {device}; mode={params['expert_mode']}, width={params['d_model']}, updates={params['train_updates']}",
            flush=True,
        )

        if mode == "compress" or mode == "both":
            print("Encoding started")
            compression()

        if mode == "decompress" or mode == "both":
            print("Decoding started")
            decompression()
            if os.path.exists(path_to_file):
                verified_size = verify_files(path_to_file, path_to_decompressed)
                print(f"Verification passed: all {verified_size} bytes match exactly.")
            else:
                print("Verification passed: restored SHA-256 matches the archive checksum.")

        original_size = os.path.getsize(
            path_to_decompressed if mode == "decompress" else path_to_file
        )
        compressed_size = os.path.getsize(path_to_compressed)

        if compressed_size <= 0:
            raise RuntimeError("Compressed file is empty")

        compression_ratio = original_size / compressed_size
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_gpu_memory_gb = get_peak_gpu_memory_gb()

        gpu_mem_text = "N/A (CPU)" if peak_gpu_memory_gb is None else f"{peak_gpu_memory_gb:.2f} GB"

        print(
            f"Compression ratio: {compression_ratio:.2f}, Time: {elapsed:.2f}, GPU memory: {gpu_mem_text}"
        )

        write_result(
            success=True,
            compression_ratio=round(float(compression_ratio), 4),
            time_spent=elapsed,
            peak_gpu_memory_gb=peak_gpu_memory_gb,
        )

    except Exception as e:
        elapsed = time.perf_counter() - start

        write_result(
            success=False,
            compression_ratio=None,
            time_spent=elapsed,
            error=str(e),
            peak_gpu_memory_gb=get_peak_gpu_memory_gb(),
        )

        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Online StateSMix: randomly initialized Mamba-2 + n-gram probabilities"
    )
    parser.add_argument("--mode", choices=("both", "compress", "decompress"), default="both")
    parser.add_argument("--input", default="data/enwik5")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--expert-mode", choices=("mamba", "memory", "mix"))
    parser.add_argument("--memory-order", type=int)
    parser.add_argument("--escape-scale", type=float)
    parser.add_argument("--train-updates", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--d-state", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--seq-length", type=int)
    args = parser.parse_args()
    updates = {
        name: getattr(args, name)
        for name in (
            "d_model",
            "d_state",
            "head_dim",
            "batch_size",
            "num_layers",
            "seq_length",
            "expert_mode",
            "memory_order",
            "escape_scale",
            "train_updates",
        )
        if getattr(args, name) is not None
    }
    if updates:
        apply_params({**params, **updates})
    mode = args.mode
    path_to_file = args.input
    path_to_compressed = path_to_file + "_compressed.dat"
    path_to_decompressed = path_to_file + "_decompressed.dat"
    if args.device != "auto":
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    main()
