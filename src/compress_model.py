"""Compress an existing joblib model without retraining."""
import argparse
import os
import time

import joblib


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress a saved joblib model.")
    parser.add_argument(
        "--input",
        default="models/flight_delay_model.pkl",
        help="Path to the existing model file.",
    )
    parser.add_argument(
        "--output",
        default="models/flight_delay_model_compressed.pkl",
        help="Path for the compressed output (does not overwrite input by default).",
    )
    parser.add_argument(
        "--compress",
        type=int,
        default=3,
        choices=range(10),
        help="joblib compression level (3 recommended).",
    )
    args = parser.parse_args()

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        raise SystemExit("Refusing to overwrite input file; choose a different --output.")

    in_mb = os.path.getsize(args.input) / (1024 * 1024)
    print(f"Loading {args.input} ({in_mb:.2f} MB)...")
    t0 = time.perf_counter()
    model = joblib.load(args.input)
    load_in = time.perf_counter() - t0

    print(f"Saving with compress={args.compress} -> {args.output}")
    t0 = time.perf_counter()
    joblib.dump(model, args.output, compress=args.compress)
    dump_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = joblib.load(args.output)
    load_out = time.perf_counter() - t0

    out_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Output size: {out_mb:.2f} MB ({100 * out_mb / in_mb:.1f}% of original)")
    print(f"Load original: {load_in:.2f}s | dump compressed: {dump_sec:.1f}s | load compressed: {load_out:.2f}s")


if __name__ == "__main__":
    main()
