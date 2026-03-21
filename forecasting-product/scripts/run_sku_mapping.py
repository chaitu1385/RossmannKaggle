#!/usr/bin/env python3
"""
CLI entry point for the SKU Mapping Discovery Pipeline (Phase 1).

Examples
--------
# Run on the built-in mock data (no real data required):
    python scripts/run_sku_mapping.py --mock --output output/mappings.csv

# Run on a real product master CSV:
    python scripts/run_sku_mapping.py \
        --product-master data/product_master.csv \
        --output output/mappings.csv

# Tune key parameters:
    python scripts/run_sku_mapping.py \
        --mock \
        --launch-window 120 \
        --min-similarity 0.75 \
        --min-confidence Medium \
        --output output/mappings.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sku_mapping import build_phase1_pipeline
from src.sku_mapping.data.mock_generator import generate_product_master

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sku_mapping")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SKU Mapping Discovery Pipeline — Phase 1 MVP"
    )

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--product-master",
        metavar="PATH",
        help="Path to product master CSV file",
    )
    source.add_argument(
        "--mock",
        action="store_true",
        help="Use the built-in synthetic product master (no real data needed)",
    )

    p.add_argument(
        "--output",
        metavar="PATH",
        default="output/sku_mappings.csv",
        help="Path for the output CSV  [default: output/sku_mappings.csv]",
    )
    p.add_argument(
        "--launch-window",
        type=int,
        default=180,
        metavar="DAYS",
        help="Max days between old-SKU and new-SKU launch dates  [default: 180]",
    )
    p.add_argument(
        "--min-similarity",
        type=float,
        default=0.70,
        metavar="THRESHOLD",
        help="Minimum base-name similarity for naming method (0–1)  [default: 0.70]",
    )
    p.add_argument(
        "--min-confidence",
        choices=["High", "Medium", "Low", "Very Low"],
        default="Low",
        help="Minimum confidence level to include in output  [default: Low]",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Build pipeline ──────────────────────────────────────────────────────
    pipeline = build_phase1_pipeline(
        launch_window_days=args.launch_window,
        min_base_similarity=args.min_similarity,
        min_confidence=args.min_confidence,
    )

    # ── Load data ──────────────────────────────────────────────────────────
    if args.mock:
        logger.info("Using built-in mock product master")
        product_master = generate_product_master()
        result_df = pipeline.run(product_master, output_path=args.output)
    else:
        result_df = pipeline.run_from_csv(
            args.product_master, output_path=args.output
        )

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  SKU Mapping Discovery — Phase 1 Results")
    print("─" * 60)
    print(f"  Total mappings found : {len(result_df)}")

    if not result_df.is_empty():
        for level in ("High", "Medium", "Low", "Very Low"):
            count = result_df.filter(
                result_df["confidence_level"] == level
            ).height
            if count:
                print(f"  {level:<12}: {count}")

        type_counts = (
            result_df.group_by("mapping_type")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        print()
        for row in type_counts.iter_rows(named=True):
            print(f"  {row['mapping_type']:<15}: {row['count']}")

    print("─" * 60)
    print(f"  Output written to    : {args.output}")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
