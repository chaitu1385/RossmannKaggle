"""Feature engineering for time series forecasting."""

from typing import List, Optional

import polars as pl


class FeatureEngineer:
    """Creates features from raw time series data."""

    def __init__(
        self,
        lag_periods: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
    ):
        self.lag_periods = lag_periods or [1, 7, 14, 30]
        self.rolling_windows = rolling_windows or [7, 14, 30]

    def create_temporal_features(self, df: pl.DataFrame, date_col: str = "Date") -> pl.DataFrame:
        """Extract temporal features from a date column."""
        df = df.with_columns(pl.col(date_col).cast(pl.Date))

        df = df.with_columns([
            pl.col(date_col).dt.year().alias("Year"),
            pl.col(date_col).dt.month().alias("Month"),
            pl.col(date_col).dt.day().alias("Day"),
            pl.col(date_col).dt.weekday().alias("DayOfWeek"),
            pl.col(date_col).dt.week().alias("WeekOfYear"),
            pl.col(date_col).dt.quarter().alias("Quarter"),
            (pl.col(date_col).dt.weekday() >= 5).cast(pl.Int64).alias("IsWeekend"),
            (pl.col(date_col).dt.day() == 1).cast(pl.Int64).alias("IsMonthStart"),
            pl.col(date_col).dt.month_end().alias("_month_end"),
        ])

        df = df.with_columns(
            (pl.col(date_col) == pl.col("_month_end")).cast(pl.Int64).alias("IsMonthEnd")
        ).drop("_month_end")

        return df

    def create_lag_features(
        self,
        df: pl.DataFrame,
        target_col: str = "Sales",
        group_col: str = "Store",
    ) -> pl.DataFrame:
        """Create lag features grouped by store."""
        df = df.sort([group_col, "Date"])

        lag_exprs = [
            pl.col(target_col).shift(lag).over(group_col).alias(f"{target_col}_lag_{lag}")
            for lag in self.lag_periods
        ]
        df = df.with_columns(lag_exprs)

        return df

    def create_rolling_features(
        self,
        df: pl.DataFrame,
        target_col: str = "Sales",
        group_col: str = "Store",
    ) -> pl.DataFrame:
        """Create rolling statistics grouped by store."""
        df = df.sort([group_col, "Date"])

        exprs = []
        for window in self.rolling_windows:
            exprs.append(
                pl.col(target_col)
                .shift(1)
                .rolling_mean(window_size=window, min_periods=1)
                .over(group_col)
                .alias(f"{target_col}_roll_mean_{window}")
            )
            exprs.append(
                pl.col(target_col)
                .shift(1)
                .rolling_std(window_size=window, min_periods=1)
                .over(group_col)
                .alias(f"{target_col}_roll_std_{window}")
            )
        df = df.with_columns(exprs)

        return df

    def create_competition_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create competition-related features."""
        if "CompetitionOpenSinceMonth" in df.columns and "Date" in df.columns:
            df = df.with_columns(
                (
                    12 * (pl.col("Year") - pl.col("CompetitionOpenSinceYear"))
                    + (pl.col("Month") - pl.col("CompetitionOpenSinceMonth"))
                ).clip(lower_bound=0).alias("CompetitionOpen")
            )

        return df

    def create_promo_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create promotion-related features."""
        if "Promo2SinceWeek" in df.columns and "WeekOfYear" in df.columns:
            df = df.with_columns(
                (
                    12 * (pl.col("Year") - pl.col("Promo2SinceYear"))
                    + (pl.col("WeekOfYear") - pl.col("Promo2SinceWeek")) / 4.0
                ).clip(lower_bound=0).alias("Promo2Open")
            )

        return df

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all feature engineering steps."""
        df = self.create_temporal_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_competition_features(df)
        df = self.create_promo_features(df)
        return df
