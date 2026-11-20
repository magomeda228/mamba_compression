"""Compress, restore and verify enwik5; auto CUDA/CPU with optional acceleration."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

import codec_core as codec

ROOT = Path(__file__).resolve().parent
MAGICS = {b"PCR1": "cpu", b"PGR1": "cuda", b"FCP2": "compiled", b"FSB2": "compiled_fused"}


def native_extension():
    """Build the optional CPU extension, or retain the Python implementation."""
    source = ROOT / "fast_memory.cpp"
    extension = importlib.util.find_spec("fast_memory")
    # Пересобирать изменённый C++: иначе Python незаметно загрузит старый .so.
    if extension is not None and extension.origin is not None:
        binary = Path(extension.origin)
        if binary.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            return True
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None or importlib.util.find_spec("pybind11") is None:
        return False
    import pybind11

    target = ROOT / ("fast_memory" + sysconfig.get_config_var("EXT_SUFFIX"))
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    # Preserve the Python reference's rounding when converting probabilities.
    command = [
        compiler,
        "-O3",
        "-ffp-contract=off",
        "-shared",
        "-std=c++17",
        "-fPIC",
        "-Wall",
        "-Wextra",
        "-I" + sysconfig.get_paths()["include"],
        "-I" + pybind11.get_include(),
        str(source),
        "-o",
        str(temporary),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        temporary.unlink(missing_ok=True)
        print("Native extension unavailable; using Python context tables.")
        (ROOT / "native_build.log").write_text(result.stdout + result.stderr)
        return False
    # Publish only a completely built library, including during parallel startup.
    temporary.replace(target)
    importlib.invalidate_caches()
    return True


def main():
    """Run the unchanged LSTM first, then our codec, and compare verified results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "data/enwik5"))
    parser.add_argument("--profile", choices=["speed", "quality"], default="speed")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--backend", choices=["auto", "portable", "compiled", "compiled_fused"], default="auto"
    )
    parser.add_argument("--mode", choices=["both", "compress", "decompress"], default="both")
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument(
        "--compare",
        action="store_true",
        help="Run LSTM first, then ours (default for --mode both).",
    )
    comparison.add_argument(
        "--only-ours", action="store_true", help="Skip the original LSTM baseline."
    )
    args = parser.parse_args()
    if args.device == "cuda" and not codec.torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch installation; use --device cpu.")
    filename = Path(args.input).resolve()
    archive = Path(str(filename) + "_compressed.dat")
    restored = Path(str(filename) + "_decompressed.dat")
    profile = json.loads((ROOT / "profiles.json").read_text())[args.profile]
    settings = profile["params"]
    device = (
        "cuda"
        if args.device == "cuda" or args.device == "auto" and codec.torch.cuda.is_available()
        else "cpu"
    )
    backend = args.backend
    baseline = None
    if args.compare and args.mode != "both":
        parser.error("--compare requires --mode both")
    if args.mode == "both" and not args.only_ours:
        print("\n=== 1. Original LSTMcompressor: full baseline ===", flush=True)
        # The baseline is a byte-identical copy of the original main.py. Import
        # it in an empty cwd so its optional params.json loader cannot pick up
        # settings from this project. Run its original main(), including verify.
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="original_lstm_") as directory:
            try:
                os.chdir(directory)
                import lstm_codec as lstm

                lstm.device = codec.torch.device(device)
                lstm.path_to_file = str(filename)
                lstm.path_to_compressed = str(filename) + "_lstm_compressed.dat"
                lstm.path_to_decompressed = str(filename) + "_lstm_decompressed.dat"
                if device == "cuda":
                    codec.torch.cuda.synchronize()
                t = time.perf_counter()
                lstm.main()
                if device == "cuda":
                    codec.torch.cuda.synchronize()
                elapsed = time.perf_counter() - t
                original_result = json.loads(Path("results.json").read_text())
            finally:
                os.chdir(previous_cwd)
        if not original_result["success"]:
            raise RuntimeError("Original LSTM baseline failed")
        size = Path(lstm.path_to_compressed).stat().st_size
        baseline = {
            "archive_bytes": size,
            "compression_ratio": filename.stat().st_size / size,
            "full_seconds": elapsed,
            "verified": True,
            "params": lstm.params.copy(),
            "implementation": "Unmodified original lstmcompressor/main.py",
        }
        print("\n=== 2. Our Mamba-2 + n-gram implementation ===", flush=True)
    if args.mode == "decompress":
        # The archive selects the replay backend; a new CLI profile cannot change it.
        with archive.open("rb") as f:
            signature = f.read(4)
        if signature in (b"FCP1", b"FSB1"):
            parser.error(
                "This archive uses the previous custom RMSNorm backend. "
                "Compress the original again with the current version."
            )
        saved = MAGICS.get(signature)
        if saved is None:
            parser.error("Unknown archive signature.")
        if saved in ["cpu", "cuda"]:
            device = saved
            backend = "portable"
        else:
            device = "cuda"
            backend = saved
        if device == "cuda" and not codec.torch.cuda.is_available():
            parser.error(
                "This archive requires CUDA. To check on CPU, compress the original locally with --device cpu."
            )
    setup_start = time.perf_counter()
    native = native_extension()
    if native:
        import fast_memory

        codec.ContinuationMemory = fast_memory.ContinuationMemory
        codec.make_freqs_from_probs_1d = fast_memory.frequencies
    # Triton here is an internal torch.compile dependency on CUDA. We do not
    # import or install any custom Triton/CuTe operations into the model.
    can_compile = (
        device == "cuda"
        and native
        and sys.platform == "linux"
        and importlib.util.find_spec("triton") is not None
        and hasattr(codec.torch, "compile")
        and hasattr(getattr(codec.torch, "compiler", None), "cudagraph_mark_step_begin")
    )
    if backend == "auto":
        backend = profile["backend"] if can_compile else "portable"
    if backend != "portable":
        if not can_compile:
            parser.error(
                "Compiled CUDA backend needs Linux, CUDA PyTorch, Triton and a C++ compiler. Use --backend portable or --device cpu."
            )
        from accelerators import configure

        configure(backend)
    else:
        codec.ARCHIVE_MAGIC = b"PCR1" if device == "cpu" else b"PGR1"
    codec.device = codec.torch.device(device)
    codec.apply_params(settings)
    codec.path_to_file = str(filename)
    codec.path_to_compressed = str(archive)
    codec.path_to_decompressed = str(restored)
    print(f"Device: {device}; backend: {backend}; native context tables: {native}", flush=True)
    print(f"Setup: {time.perf_counter() - setup_start:.3f}s; profile: {args.profile}", flush=True)
    start = time.perf_counter()
    encoded = None
    decoded = None
    if args.mode in ["both", "compress"]:
        codec.compression()
        if device == "cuda":
            codec.torch.cuda.synchronize()
        encoded = time.perf_counter() - start
    decoding_start = time.perf_counter()
    if args.mode in ["both", "decompress"]:
        codec.decompression()
        if device == "cuda":
            codec.torch.cuda.synchronize()
        decoded = time.perf_counter() - decoding_start
    if args.mode == "both":
        codec.verify_files(filename, restored)
    full = time.perf_counter() - start
    print(f"Archive: {archive.stat().st_size} bytes", flush=True)
    if args.mode != "decompress":
        print(
            f"Compression ratio: {filename.stat().st_size / archive.stat().st_size:.6f}", flush=True
        )
    if encoded is not None:
        print(f"Encoding: {encoded:.3f}s", flush=True)
    if decoded is not None:
        print(f"Decoding: {decoded:.3f}s", flush=True)
    print(f"Full codec wall time: {full:.3f}s (includes first CUDA/compiler setup)", flush=True)
    if args.mode == "both":
        print("Verification: exact byte-for-byte match", flush=True)
    result = {
        "device": device,
        "backend": backend,
        "native_context": native,
        "archive_bytes": archive.stat().st_size,
        "encode_seconds": encoded,
        "decode_seconds": decoded,
        "full_seconds": full,
        "verified": args.mode == "both",
    }
    if baseline:
        # Compare complete archives (including headers), using this machine's times.
        result["baseline"] = baseline
        result["ratio_gain_percent"] = (
            baseline["archive_bytes"] / result["archive_bytes"] - 1
        ) * 100
        result["time_change_percent"] = (full / baseline["full_seconds"] - 1) * 100
        print("\n=== Direct comparison on the same input and device ===", flush=True)
        print(
            f"{'Implementation':<23} {'Input, bytes':>13} {'Archive, bytes':>15} {'Ratio':>11} {'Full time, s':>14} {'Verified':>10}",
            flush=True,
        )
        for name, row in [("Original LSTM", baseline), ("Mamba-2 + n-gram", result)]:
            ratio = filename.stat().st_size / row["archive_bytes"]
            print(
                f"{name:<23} {filename.stat().st_size:>13} {row['archive_bytes']:>15} {ratio:>11.6f} {row['full_seconds']:>14.3f} {'exact':>10}",
                flush=True,
            )
        print(
            f"Ratio gain vs original LSTM: {result['ratio_gain_percent']:+.2f}%; full time change: {result['time_change_percent']:+.2f}%",
            flush=True,
        )
        print(
            "This local run includes runtime initialization/compilation, unlike warmed report medians.",
            flush=True,
        )
    (ROOT / "results.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
