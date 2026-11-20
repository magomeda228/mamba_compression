"""Check that decompression reproduces every byte of the original file."""

import argparse
import os


def verify_files(original_path, restored_path):
    """Return the verified byte count, or raise RuntimeError on a mismatch."""
    offset = 0
    chunk_size = 1024 * 1024

    # Одинаковый размер файлов не гарантирует совпадения: сравниваем все байты,
    # читая блоками, чтобы проверка не зависела от объёма доступной RAM.
    with open(original_path, "rb") as original, open(restored_path, "rb") as restored:
        original_size = os.fstat(original.fileno()).st_size
        restored_size = os.fstat(restored.fileno()).st_size

        while True:
            expected = original.read(chunk_size)
            actual = restored.read(chunk_size)

            if expected != actual:
                common_length = min(len(expected), len(actual))
                difference = next(
                    (i for i in range(common_length) if expected[i] != actual[i]),
                    common_length,
                )
                expected_byte = (
                    f"0x{expected[difference]:02x}" if difference < len(expected) else "EOF"
                )
                actual_byte = f"0x{actual[difference]:02x}" if difference < len(actual) else "EOF"
                raise RuntimeError(
                    f"Decompression mismatch at byte {offset + difference} (0-based): "
                    f"expected {expected_byte}, got {actual_byte}; "
                    f"original size={original_size}, restored size={restored_size}"
                )

            if not expected:
                return offset

            offset += len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", nargs="?", default="data/enwik5")
    parser.add_argument("restored", nargs="?", default="data/enwik5_decompressed.dat")
    args = parser.parse_args()

    try:
        size = verify_files(args.original, args.restored)
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"Verification failed: {error}\n")

    print(f"Verification passed: all {size} bytes match exactly.")


if __name__ == "__main__":
    main()
