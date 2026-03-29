"""Data preprocessing for the forecasting platform."""

from typing import List, Optional

import polars as pl


class DataPreprocessor:
    """Handles data cleaning and preprocessing."""

    def __init__(self, remove_closed: bool = True):
        self.remove_closed = remove_closed

    def clean(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all cleaning steps."""
        if self.remove_closed and "Open" in df.columns:
            df = df.filter(pl.col("Open") == 1)

        if "Sales" in df.columns:
            df = df.filter(pl.col("Sales") > 0)

        df = self._fill_missing(df)
        return df

    def _fill_missing(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fill missing values with sensible defaults."""
        if "CompetitionDistance" in df.columns:
            median_val = df.get_column("CompetitionDistance").median()
            df = df.with_columns(
                pl.col("CompetitionDistance").fill_null(median_val)
            )

        for col in ["Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"]:
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null(0))

        for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]:
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null(0))

        return df

    def encode_categoricals(
        self, df: pl.DataFrame, columns: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """Label encode categorical columns."""
        if columns is None:
            columns = [
                col for col in df.columns
                if df.schema[col] == pl.Utf8 or df.schema[col] == pl.Categorical
            ]

        for col in columns:
            if col in df.columns:
                # Build a mapping from unique values to integer codes
                unique_vals = df.get_column(col).unique().sort().to_list()
                mapping = {v: i for i, v in enumerate(unique_vals)}
                df = df.with_columns(
                    pl.col(col).replace(mapping).cast(pl.Int64).alias(col)
                )

        return df
