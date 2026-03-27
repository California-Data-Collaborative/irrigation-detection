"""
CLI entry point for irrigation detection.

Usage:
    irrigation-detect --input readings.csv --model model.pth --output results.csv
"""

import argparse
import logging
import os
import sys

import pandas as pd

from .models import load_model
from .detection import detect_irrigation

logger = logging.getLogger("irrigation_detection")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="irrigation-detect",
        description="Detect irrigation usage from hourly smart meter data.",
    )
    parser.add_argument(
        "--input",
        default=os.environ.get("INPUT_PATH"),
        required=not os.environ.get("INPUT_PATH"),
        help="Path to input CSV or Parquet file (columns: meter_id, timestamp, reading)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH"),
        required=not os.environ.get("MODEL_PATH"),
        help="Path to trained model weights (.pth file)",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("OUTPUT_PATH", "results.csv"),
        help="Path for output CSV (default: results.csv)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "cpu"),
        help="Torch device: cpu or cuda (default: cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("GPU_BATCH", "2000")),
        help="Windows per inference batch (default: 2000)",
    )
    parser.add_argument(
        "--source",
        default=os.environ.get("SOURCE_LABEL"),
        help="Optional source label column in output",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args(argv)


def _read_input(path: str) -> pd.DataFrame:
    """Read CSV or Parquet input file."""
    if path.endswith(".parquet") or path.endswith(".pq"):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Validate paths
    if not os.path.isfile(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)

    # Load data
    logger.info(f"Reading input from {args.input}")
    df = _read_input(args.input)
    logger.info(f"Loaded {len(df):,} rows")

    # Load model
    model = load_model(args.model, device=args.device)

    # Run detection
    results = detect_irrigation(
        df=df,
        model=model,
        device=args.device,
        source_label=args.source,
        gpu_batch=args.batch_size,
    )

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results.to_csv(args.output, index=False)
    logger.info(f"Results written to {args.output} ({len(results):,} rows)")


if __name__ == "__main__":
    main()
