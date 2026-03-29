"""Data loading utilities for the forecasting platform."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import polars as pl


def read_csv_with_dates(
    path: str,
    date_columns: Optional[Dict[str, str]] = None,
    schema_overrides: Optional[Dict[str, pl.DataType]] = None,
    **kwargs,
) -> pl.DataFrame:
    """
    Read a CSV with explicit date format parsing.

    Avoids relying on ``try_parse_dates=True`` which can silently fail
    when combined with ``schema_overrides`` or on ambiguous formats.

    Parameters
    ----------
    path:
        Path to the CSV file.
    date_columns:
        Mapping of column name to strftime format string.
        E.g. ``{"Date": "%Y-%m-%d", "week": "%m/%d/%Y"}``.
    schema_overrides:
        Polars dtype overrides for columns with mixed types
        (e.g. ``{"StateHoliday": pl.Utf8}``).
    **kwargs:
        Passed through to ``pl.read_csv``.

    Returns
    -------
    Polars DataFrame with date columns correctly typed.
    """
    df = pl.read_csv(
        path,
        try_parse_dates=False,
        schema_overrides=schema_overrides,
        **kwargs,
    )
    if date_columns:
        for col, fmt in date_columns.items():
            if col in df.columns:
                df = df.with_columns(pl.col(col).str.to_date(fmt))
    return df


class DataLoader:
    """Loads and validates raw data for forecasting."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_train(self) -> pl.DataFrame:
        """Load training data."""
        path = self.data_dir / "train.csv"
        if not path.exists():
            raise FileNotFoundError(f"No such file or directory: '{path}'")
        df = pl.read_csv(str(path), try_parse_dates=True)
        return df

    def load_test(self) -> pl.DataFrame:
        """Load test data."""
        path = self.data_dir / "test.csv"
        if not path.exists():
            raise FileNotFoundError(f"No such file or directory: '{path}'")
        df = pl.read_csv(str(path), try_parse_dates=True)
        return df

    def load_store(self) -> pl.DataFrame:
        """Load store metadata."""
        path = self.data_dir / "store.csv"
        if not path.exists():
            raise FileNotFoundError(f"No such file or directory: '{path}'")
        df = pl.read_csv(str(path))
        return df

    def load_all(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Load all datasets and return (train, test, store)."""
        train = self.load_train()
        test = self.load_test()
        store = self.load_store()
        return train, test, store

    def merge_with_store(
        self, df: pl.DataFrame, store: pl.DataFrame
    ) -> pl.DataFrame:
        """Merge sales data with store metadata."""
        return df.join(store, on="Store", how="left")
