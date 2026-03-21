"""Feature engineering for time series forecasting."""

from typing import List, Optional

import pandas as pd


class FeatureEngineer:
    """Creates features from raw time series data."""

    def __init__(
        self,
        lag_periods: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
    ):
        self.lag_periods = lag_periods or [1, 7, 14, 30]
        self.rolling_windows = rolling_windows or [7, 14, 30]

    def create_temporal_features(self, df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
        """Extract temporal features from a date column."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        df["Year"] = df[date_col].dt.year
        df["Month"] = df[date_col].dt.month
        df["Day"] = df[date_col].dt.day
        df["DayOfWeek"] = df[date_col].dt.dayofweek
        df["WeekOfYear"] = df[date_col].dt.isocalendar().week.astype(int)
        df["Quarter"] = df[date_col].dt.quarter
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
        df["IsMonthStart"] = df[date_col].dt.is_month_start.astype(int)
        df["IsMonthEnd"] = df[date_col].dt.is_month_end.astype(int)

        return df

    def create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str = "Sales",
        group_col: str = "Store",
    ) -> pd.DataFrame:
        """Create lag features grouped by store."""
        df = df.copy().sort_values([group_col, "Date"])

        for lag in self.lag_periods:
            col_name = f"{target_col}_lag_{lag}"
            df[col_name] = df.groupby(group_col)[target_col].shift(lag)

        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str = "Sales",
        group_col: str = "Store",
    ) -> pd.DataFrame:
        """Create rolling statistics grouped by store."""
        df = df.copy().sort_values([group_col, "Date"])

        for window in self.rolling_windows:
            group = df.groupby(group_col)[target_col]
            df[f"{target_col}_roll_mean_{window}"] = group.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f"{target_col}_roll_std_{window}"] = group.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )

        return df

    def create_competition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create competition-related features."""
        df = df.copy()

        if "CompetitionOpenSinceMonth" in df.columns and "Date" in df.columns:
            df["CompetitionOpen"] = (
                12 * (df["Year"] - df["CompetitionOpenSinceYear"])
                + (df["Month"] - df["CompetitionOpenSinceMonth"])
            ).clip(lower=0)

        return df

    def create_promo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create promotion-related features."""
        df = df.copy()

        if "Promo2SinceWeek" in df.columns and "WeekOfYear" in df.columns:
            df["Promo2Open"] = (
                12 * (df["Year"] - df["Promo2SinceYear"])
                + (df["WeekOfYear"] - df["Promo2SinceWeek"]) / 4.0
            ).clip(lower=0)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        df = self.create_temporal_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_competition_features(df)
        df = self.create_promo_features(df)
        return df
