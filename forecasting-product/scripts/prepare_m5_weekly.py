"""Aggregate the M5 daily sample fixture into weekly frequency.

Reads the daily CSV (50 series × 731 days) and outputs a weekly CSV
with a configurable subset of series for faster demo/testing.

Usage
-----
    python scripts/prepare_m5_weekly.py                                    # defaults
    python scripts/prepare_m5_weekly.py --input path/to/daily.csv          # custom input
    python scripts/prepare_m5_weekly.py --output data/demo/m5_weekly.csv   # custom output
    python scripts/prepare_m5_weekly.py --n-series 10                      # series count
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
DEFAULT_OUTPUT = _HERE / "data" / "demo" / "m5_weekly.csv"
N_SERIES = 10
SEED = 42


def prepare_weekly(
    input_path: Path,
    output_path: Path,
    n_series: int = N_SERIES,
) -> pl.DataFrame:
    """Aggregate daily M5 data to weekly frequency.

    Parameters
    ----------
    input_path : Path to the daily CSV fixture.
    output_path : Where to write the weekly CSV.
    n_series : Number of series to keep (deterministic subset).

    Returns
    -------
    The weekly DataFrame (also written to *output_path*).
    """
    print(f"[1/4] Reading daily data from {input_path} …")
    df = pl.read_csv(input_path, try_parse_dates=True)
    df = df.with_columns(pl.col("date").cast(pl.Date))

    all_series = sorted(df["series_id"].unique().to_list())
    n_available = len(all_series)
    print(f"       {n_available} series, {df.height} rows")

    # Deterministic subset
    import random
    rng = random.Random(SEED)
    selected = sorted(rng.sample(all_series, min(n_series, n_available)))
    print(f"[2/4] Selecting {len(selected)} series: {selected[:5]}{'…' if len(selected) > 5 else ''}")
    df = df.filter(pl.col("series_id").is_in(selected))

    print("[3/4] Aggregating daily → weekly (Monday start) …")
    # Truncate date to week start (Monday) and aggregate
    df = df.with_columns(
        pl.col("date").dt.truncate("1w").alias("week")
    )

    # Aggregate: sum quantity, keep first of each dimension column
    weekly = (
        df.group_by(["series_id", "week"])
        .agg([
            pl.col("quantity").sum(),
            pl.col("state_id").first(),
            pl.col("store_id").first(),
            pl.col("cat_id").first(),
            pl.col("dept_id").first(),
        ])
        .rename({"week": "date"})
        .sort(["series_id", "date"])
    )

    n_weeks = weekly.group_by("series_id").len()["len"].mean()
    print(f"       {weekly.height} rows, ~{n_weeks:.0f} weeks/series")

    print(f"[4/4] Writing to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weekly.write_csv(output_path)
    print("Done ✓")
    return weekly


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate M5 daily → weekly")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-series", type=int, default=N_SERIES)
    args = parser.parse_args()
    prepare_weekly(args.input, args.output, args.n_series)


if __name__ == "__main__":
    main()
