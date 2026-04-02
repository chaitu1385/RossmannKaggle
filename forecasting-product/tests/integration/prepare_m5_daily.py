"""Prepare the Walmart M5 dataset for daily-frequency E2E testing.

Reads the raw M5 files (sales_train_validation.csv, calendar.csv) from
a configurable source directory, melts from wide to long format, samples
50 series from 3 stores, keeps the last 730 days of history, and writes
the result as a CSV fixture for the integration test suite.

Usage
-----
    python -m tests.integration.prepare_m5_daily                      # uses default paths
    python -m tests.integration.prepare_m5_daily --m5-dir /path/to/m5 # override M5 data dir
    python -m tests.integration.prepare_m5_daily --output /tmp/out.csv # override output path
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_M5_DIR = Path(r"C:\Users\ksanka\Downloads\Dev\m5-forecasting-accuracy")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "fixtures" / "m5_daily_sample.csv"

# Sampling parameters
STORES = ["CA_1", "TX_1", "WI_1"]
ITEMS_PER_DEPT = 5
N_HISTORY_DAYS = 730
SEED = 42


def prepare(m5_dir: Path, output: Path) -> pl.DataFrame:
    """Transform M5 wide-format sales into a long daily sample.

    Returns the resulting DataFrame (also written to *output*).
    """
    print(f"[1/6] Reading calendar.csv from {m5_dir} …")
    calendar = pl.read_csv(
        m5_dir / "calendar.csv",
        try_parse_dates=True,
    ).select(["date", "d"])
    # Ensure date is a Date column
    calendar = calendar.with_columns(pl.col("date").cast(pl.Date))

    # Build mapping: d column string (e.g. "d_1") -> date
    d_to_date: dict[str, str] = dict(
        zip(calendar["d"].to_list(), calendar["date"].cast(pl.Utf8).to_list())
    )

    print(f"[2/6] Reading sales_train_validation.csv (wide) …")
    sales_wide = pl.read_csv(m5_dir / "sales_train_validation.csv")

    # Identify d_* columns (the day columns)
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    d_cols = [c for c in sales_wide.columns if c.startswith("d_")]

    print(f"       {len(d_cols)} day columns found ({d_cols[0]} … {d_cols[-1]})")

    print(f"[3/6] Filtering to stores {STORES} and sampling {ITEMS_PER_DEPT} items/dept …")
    sales_filtered = sales_wide.filter(pl.col("store_id").is_in(STORES))

    # Sample items per (store, dept) for diversity
    sampled_ids: list[str] = []
    for group_key, group_df in sales_filtered.group_by(["store_id", "dept_id"]):
        store, dept = group_key
        ids = group_df["id"].to_list()
        rng_sample = ids[:ITEMS_PER_DEPT] if len(ids) <= ITEMS_PER_DEPT else _deterministic_sample(ids, ITEMS_PER_DEPT)
        sampled_ids.extend(rng_sample)

    sales_sampled = sales_filtered.filter(pl.col("id").is_in(sampled_ids))
    n_series = sales_sampled.height
    print(f"       Sampled {n_series} series")

    print("[4/6] Melting wide → long …")
    sales_long = sales_sampled.unpivot(
        index=id_cols,
        on=d_cols,
        variable_name="d",
        value_name="quantity",
    )

    # Map d column to actual date
    sales_long = sales_long.with_columns(
        pl.col("d").replace_strict(d_to_date, default=None).cast(pl.Date).alias("date")
    ).drop("d")

    # Create series_id = store_id + "_" + item_id
    sales_long = sales_long.with_columns(
        (pl.col("store_id") + "_" + pl.col("item_id")).alias("series_id")
    )

    print(f"[5/6] Filtering to last {N_HISTORY_DAYS} days …")
    max_date = sales_long["date"].max()
    cutoff = max_date - pl.duration(days=N_HISTORY_DAYS)
    # Polars date comparison
    sales_long = sales_long.filter(pl.col("date") >= cutoff)

    # Select and order final columns
    sales_long = sales_long.select([
        "series_id", "date", "quantity",
        "state_id", "store_id", "cat_id", "dept_id",
    ]).sort(["series_id", "date"])

    n_series_final = sales_long["series_id"].n_unique()
    n_rows = sales_long.height
    date_min = sales_long["date"].min()
    date_max = sales_long["date"].max()

    print(f"[6/6] Writing {n_rows} rows ({n_series_final} series, {date_min} → {date_max}) to {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    sales_long.write_csv(output)

    print("Done ✓")
    return sales_long


def _deterministic_sample(items: list[str], k: int) -> list[str]:
    """Deterministic sample of *k* items (sorted, then sliced evenly)."""
    import random
    rng = random.Random(SEED)
    shuffled = sorted(items)
    rng.shuffle(shuffled)
    return shuffled[:k]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare M5 daily sample for E2E tests")
    parser.add_argument(
        "--m5-dir",
        type=Path,
        default=DEFAULT_M5_DIR,
        help="Path to m5-forecasting-accuracy directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path",
    )
    args = parser.parse_args()

    if not args.m5_dir.exists():
        print(f"ERROR: M5 directory not found: {args.m5_dir}", file=sys.stderr)
        sys.exit(1)

    df = prepare(args.m5_dir, args.output)

    # Summary
    print(f"\nSummary:")
    print(f"  Series:     {df['series_id'].n_unique()}")
    print(f"  Rows:       {df.height}")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"  Stores:     {df['store_id'].n_unique()}")
    print(f"  Categories: {df['cat_id'].n_unique()}")
    print(f"  Depts:      {df['dept_id'].n_unique()}")


if __name__ == "__main__":
    main()
