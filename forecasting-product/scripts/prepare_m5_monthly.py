"""Aggregate the M5 daily sample fixture into monthly frequency.

Reads the daily CSV (50 series × 731 days) and outputs a monthly CSV
with a configurable subset of series for faster demo/testing.

Usage
-----
    python scripts/prepare_m5_monthly.py                                     # defaults
    python scripts/prepare_m5_monthly.py --input path/to/daily.csv           # custom input
    python scripts/prepare_m5_monthly.py --output data/demo/m5_monthly.csv   # custom output
    python scripts/prepare_m5_monthly.py --n-series 10                       # series count
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

DEFAULT_INPUT = _HERE / "tests" / "integration" / "fixtures" / "m5_daily_sample.csv"
DEFAULT_OUTPUT = _HERE / "data" / "demo" / "m5_monthly.csv"
N_SERIES = 10
SEED = 42


def prepare_monthly(
    input_path: Path,
    output_path: Path,
    n_series: int = N_SERIES,
) -> pl.DataFrame:
    """Aggregate daily M5 data to monthly frequency.

    Parameters
    ----------
    input_path : Path to the daily CSV fixture.
    output_path : Where to write the monthly CSV.
    n_series : Number of series to keep (deterministic subset).

    Returns
    -------
    The monthly DataFrame (also written to *output_path*).
    """
    print(f"[1/4] Reading daily data from {input_path} …")
    df = pl.read_csv(input_path, try_parse_dates=True)
    df = df.with_columns(pl.col("date").cast(pl.Date))

    all_series = sorted(df["series_id"].unique().to_list())
    n_available = len(all_series)
    print(f"       {n_available} series, {df.height} rows")

    # Deterministic subset (same seed as weekly for consistency)
    import random
    rng = random.Random(SEED)
    selected = sorted(rng.sample(all_series, min(n_series, n_available)))
    print(f"[2/4] Selecting {len(selected)} series: {selected[:5]}{'…' if len(selected) > 5 else ''}")
    df = df.filter(pl.col("series_id").is_in(selected))

    print("[3/4] Aggregating daily → monthly (first of month) …")
    df = df.with_columns(
        pl.col("date").dt.truncate("1mo").alias("month")
    )

    monthly = (
        df.group_by(["series_id", "month"])
        .agg([
            pl.col("quantity").sum(),
            pl.col("state_id").first(),
            pl.col("store_id").first(),
            pl.col("cat_id").first(),
            pl.col("dept_id").first(),
        ])
        .rename({"month": "date"})
        .sort(["series_id", "date"])
    )

    n_months = monthly.group_by("series_id").len()["len"].mean()
    print(f"       {monthly.height} rows, ~{n_months:.0f} months/series")

    print(f"[4/4] Writing to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    monthly.write_csv(output_path)
    print("Done ✓")
    return monthly


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate M5 daily → monthly")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-series", type=int, default=N_SERIES)
    args = parser.parse_args()
    prepare_monthly(args.input, args.output, args.n_series)


if __name__ == "__main__":
    main()
