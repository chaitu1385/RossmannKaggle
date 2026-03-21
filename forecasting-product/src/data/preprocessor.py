"""Data preprocessing for the forecasting platform."""

from typing import List, Optional

import pandas as pd


class DataPreprocessor:
    """Handles data cleaning and preprocessing."""

    def __init__(self, remove_closed: bool = True):
        self.remove_closed = remove_closed

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning steps."""
        df = df.copy()

        if self.remove_closed and "Open" in df.columns:
            df = df[df["Open"] == 1]

        if "Sales" in df.columns:
            df = df[df["Sales"] > 0]

        df = self._fill_missing(df)
        return df.reset_index(drop=True)

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with sensible defaults."""
        if "CompetitionDistance" in df.columns:
            df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
                df["CompetitionDistance"].median()
            )

        for col in ["Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    def encode_categoricals(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Label encode categorical columns."""
        if columns is None:
            columns = df.select_dtypes(include=["object"]).columns.tolist()

        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = pd.Categorical(df[col]).codes

        return df
