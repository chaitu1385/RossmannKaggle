"""
SparkFeatureEngineer — distributed feature engineering using PySpark.

Mirrors the logic in ``src.data.feature_engineering.FeatureEngineer`` but
operates on Spark DataFrames using native PySpark functions (no UDFs for the
core transformations — only SQL functions and Window specs for maximum
performance on large clusters).

Usage
-----
>>> from src.spark.feature_engineering import SparkFeatureEngineer
>>> eng = SparkFeatureEngineer(lag_periods=[1, 7, 14], rolling_windows=[7, 14])
>>> df_features = eng.fit_transform(df, date_col="Date", target_col="Sales", group_col="Store")
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class SparkFeatureEngineer:
    """
    Creates features from raw time-series data using PySpark.

    Parameters
    ----------
    lag_periods:
        List of lag periods (in rows) to compute for the target column.
    rolling_windows:
        List of window sizes for rolling mean / std.
    """

    def __init__(
        self,
        lag_periods: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
    ):
        self.lag_periods = lag_periods or [1, 7, 14, 30]
        self.rolling_windows = rolling_windows or [7, 14, 30]

    # ── temporal features ─────────────────────────────────────────────────────

    def create_temporal_features(self, df, date_col: str = "Date"):
        """
        Extract calendar features from a date column.

        Adds: Year, Month, Day, DayOfWeek, WeekOfYear, Quarter,
              IsWeekend, IsMonthStart, IsMonthEnd.
        """
        from pyspark.sql import functions as F

        df = df.withColumn(date_col, F.to_date(F.col(date_col)))
        df = (
            df
            .withColumn("Year",         F.year(date_col))
            .withColumn("Month",        F.month(date_col))
            .withColumn("Day",          F.dayofmonth(date_col))
            .withColumn("DayOfWeek",    F.dayofweek(date_col))
            .withColumn("WeekOfYear",   F.weekofyear(date_col))
            .withColumn("Quarter",      F.quarter(date_col))
            .withColumn("IsWeekend",    (F.dayofweek(date_col).isin(1, 7)).cast("int"))
            .withColumn("IsMonthStart", (F.dayofmonth(date_col) == 1).cast("int"))
            # last day of month: add 1 day, check if month changes
            .withColumn(
                "IsMonthEnd",
                (F.month(F.date_add(date_col, 1)) != F.month(date_col)).cast("int"),
            )
        )
        logger.debug("Created temporal features from column '%s'", date_col)
        return df

    # ── lag features ──────────────────────────────────────────────────────────

    def create_lag_features(
        self,
        df,
        target_col: str = "Sales",
        group_col: str = "Store",
        date_col: str = "Date",
    ):
        """
        Create lag features partitioned by ``group_col``, ordered by ``date_col``.
        """
        from pyspark.sql import Window
        from pyspark.sql import functions as F

        window = Window.partitionBy(group_col).orderBy(date_col)

        for lag in self.lag_periods:
            col_name = f"{target_col}_lag_{lag}"
            df = df.withColumn(col_name, F.lag(F.col(target_col), lag).over(window))

        logger.debug(
            "Created %d lag features for '%s'", len(self.lag_periods), target_col
        )
        return df

    # ── rolling features ──────────────────────────────────────────────────────

    def create_rolling_features(
        self,
        df,
        target_col: str = "Sales",
        group_col: str = "Store",
        date_col: str = "Date",
    ):
        """
        Create rolling mean / std features (lagged by 1 to avoid leakage).
        """
        from pyspark.sql import Window
        from pyspark.sql import functions as F

        for window_size in self.rolling_windows:
            # Lag by 1 then compute rolling stats to avoid target leakage.
            lagged_col = f"__lag1_{target_col}"
            w_lag = Window.partitionBy(group_col).orderBy(date_col)
            w_roll = (
                Window.partitionBy(group_col)
                .orderBy(date_col)
                .rowsBetween(-window_size + 1, 0)
            )

            df = df.withColumn(lagged_col, F.lag(F.col(target_col), 1).over(w_lag))
            df = df.withColumn(
                f"{target_col}_roll_mean_{window_size}",
                F.mean(F.col(lagged_col)).over(w_roll),
            )
            df = df.withColumn(
                f"{target_col}_roll_std_{window_size}",
                F.stddev(F.col(lagged_col)).over(w_roll),
            )
            df = df.drop(lagged_col)

        logger.debug(
            "Created %d rolling windows for '%s'", len(self.rolling_windows), target_col
        )
        return df

    # ── competition features ──────────────────────────────────────────────────

    def create_competition_features(self, df):
        """
        Compute months since competition opened (clipped to 0).

        Requires columns: Year, Month, CompetitionOpenSinceYear,
        CompetitionOpenSinceMonth.
        """
        from pyspark.sql import functions as F

        cols = {f.name for f in df.schema.fields}
        if "CompetitionOpenSinceMonth" not in cols or "Year" not in cols:
            return df

        df = df.withColumn(
            "CompetitionOpen",
            F.greatest(
                F.lit(0),
                (
                    12 * (F.col("Year") - F.col("CompetitionOpenSinceYear"))
                    + (F.col("Month") - F.col("CompetitionOpenSinceMonth"))
                ),
            ),
        )
        logger.debug("Created CompetitionOpen feature")
        return df

    # ── promo features ────────────────────────────────────────────────────────

    def create_promo_features(self, df):
        """
        Compute months since Promo2 started (clipped to 0).

        Requires columns: Year, WeekOfYear, Promo2SinceYear, Promo2SinceWeek.
        """
        from pyspark.sql import functions as F

        cols = {f.name for f in df.schema.fields}
        if "Promo2SinceWeek" not in cols or "Year" not in cols:
            return df

        df = df.withColumn(
            "Promo2Open",
            F.greatest(
                F.lit(0.0),
                (
                    12 * (F.col("Year") - F.col("Promo2SinceYear")).cast("double")
                    + (F.col("WeekOfYear") - F.col("Promo2SinceWeek")).cast("double") / 4.0
                ),
            ),
        )
        logger.debug("Created Promo2Open feature")
        return df

    # ── combined ──────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        df,
        date_col: str = "Date",
        target_col: str = "Sales",
        group_col: str = "Store",
    ):
        """
        Apply all feature engineering steps in order.

        Returns a Spark DataFrame with all feature columns appended.
        """
        df = self.create_temporal_features(df, date_col=date_col)
        df = self.create_lag_features(df, target_col=target_col, group_col=group_col, date_col=date_col)
        df = self.create_rolling_features(df, target_col=target_col, group_col=group_col, date_col=date_col)
        df = self.create_competition_features(df)
        df = self.create_promo_features(df)
        logger.info("Feature engineering complete. Schema has %d columns.", len(df.columns))
        return df
