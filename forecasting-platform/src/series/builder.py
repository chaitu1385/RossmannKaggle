"""
Series builder — constructs model-ready time series from raw actuals.

Responsibilities:
  1. Validate and align raw data to the expected schema.
  2. Apply product transition stitching (via TransitionEngine).
  3. Build composite series IDs from hierarchy columns.
  4. Fill gaps (missing weeks) with zeros or interpolation.
  5. Return a clean DataFrame ready for forecasting.
"""

from datetime import date
from typing import Optional

import polars as pl

from ..config.schema import PlatformConfig
from .transition import TransitionEngine


class SeriesBuilder:
    """
    Constructs model-ready time series.

    Usage
    -----
    >>> builder = SeriesBuilder(config)
    >>> series = builder.build(actuals, product_master, mapping_table)
    """

    def __init__(self, config: PlatformConfig):
        self.config = config
        self._transition_engine = TransitionEngine(config.transition)

    def build(
        self,
        actuals: pl.DataFrame,
        external_features: Optional[pl.DataFrame] = None,
        product_master: Optional[pl.DataFrame] = None,
        mapping_table: Optional[pl.DataFrame] = None,
        forecast_origin: Optional[date] = None,
        overrides: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """
        Build model-ready time series from raw actuals.

        Parameters
        ----------
        actuals:
            Raw weekly actuals.  Must contain the time column, value column,
            and hierarchy identifier columns.
        product_master:
            Product metadata with launch_date.  Required for transitions.
        mapping_table:
            SKU mapping output.  Required for transitions.
        forecast_origin:
            The forecast start date.  Required for transitions.
        overrides:
            Planner overrides for transitions.

        Returns
        -------
        Model-ready DataFrame with columns: series_id, week, quantity
        (plus any extra feature columns).
        """
        fc = self.config.forecast
        time_col = fc.time_column
        value_col = fc.target_column
        sid_col = fc.series_id_column

        # Step 1: Build composite series ID if not present
        df = self._ensure_series_id(actuals, sid_col)

        # Step 2: Apply transitions
        if (
            mapping_table is not None
            and product_master is not None
            and forecast_origin is not None
        ):
            plans = self._transition_engine.compute_plans(
                mapping_table=mapping_table,
                product_master=product_master,
                forecast_origin=forecast_origin,
                horizon_weeks=fc.horizon_weeks,
                overrides=overrides,
            )
            df = self._transition_engine.stitch_series(
                actuals=df,
                plans=plans,
                time_column=time_col,
                series_id_column=sid_col,
                value_column=value_col,
            )

        # Step 3: Fill missing weeks (controlled by data_quality config)
        dq = self.config.data_quality
        if dq.fill_gaps:
            df = self._fill_gaps(df, time_col, sid_col, value_col, dq.fill_value)

        # Step 3a: Drop short series
        if dq.min_series_length_weeks > 0:
            series_lengths = (
                df.group_by(sid_col)
                .agg(pl.col(time_col).count().alias("_len"))
            )
            valid_ids = series_lengths.filter(
                pl.col("_len") >= dq.min_series_length_weeks
            )[sid_col].to_list()
            df = df.filter(pl.col(sid_col).is_in(valid_ids))

        # Step 3b: Drop all-zero series
        if dq.drop_zero_series:
            nonzero = (
                df.group_by(sid_col)
                .agg(pl.col(value_col).abs().sum().alias("_total"))
                .filter(pl.col("_total") > 0)
            )[sid_col].to_list()
            df = df.filter(pl.col(sid_col).is_in(nonzero))

        # Step 3c: Join external features (if provided and enabled)
        if external_features is not None and self.config.forecast.external_regressors.enabled:
            feature_cols = self.config.forecast.external_regressors.feature_columns
            available_cols = [c for c in feature_cols if c in external_features.columns]
            if available_cols:
                join_cols = [time_col]
                select_cols = [time_col] + available_cols

                # If features have series_id, join on both; otherwise broadcast
                if sid_col in external_features.columns:
                    join_cols = [sid_col, time_col]
                    select_cols = [sid_col, time_col] + available_cols

                df = df.join(
                    external_features.select(select_cols),
                    on=join_cols,
                    how="left",
                )

                # Fill nulls in feature columns with 0
                for col in available_cols:
                    df = df.with_columns(
                        pl.col(col).fill_null(0).alias(col)
                    )

        # Step 4: Sort
        df = df.sort([sid_col, time_col])

        return df

    def _ensure_series_id(
        self, df: pl.DataFrame, sid_col: str
    ) -> pl.DataFrame:
        """
        If series_id column doesn't exist, build it from the output grain
        columns defined in config.
        """
        if sid_col in df.columns:
            return df

        grain = self.config.output.grain
        grain_cols = list(grain.values())
        missing = [c for c in grain_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Cannot build series_id: columns {missing} not in data. "
                f"Available: {df.columns}"
            )

        # Concatenate grain columns with "|" separator
        concat_expr = pl.concat_str(
            [pl.col(c).cast(pl.Utf8) for c in grain_cols],
            separator="|",
        ).alias(sid_col)

        return df.with_columns(concat_expr)

    def _fill_gaps(
        self,
        df: pl.DataFrame,
        time_col: str,
        sid_col: str,
        value_col: str,
        fill_value: float = 0.0,
    ) -> pl.DataFrame:
        """Fill missing weeks for each series."""
        if df.is_empty():
            return df

        min_date = df[time_col].min()
        max_date = df[time_col].max()

        if min_date is None or max_date is None:
            return df

        all_weeks = pl.date_range(
            min_date, max_date, interval="1w", eager=True
        ).alias(time_col)
        all_weeks_df = pl.DataFrame({time_col: all_weeks})

        series_ids = df.select(sid_col).unique()
        complete_grid = series_ids.join(all_weeks_df, how="cross")

        filled = complete_grid.join(
            df, on=[sid_col, time_col], how="left"
        )

        if value_col in filled.columns:
            filled = filled.with_columns(
                pl.col(value_col).fill_null(fill_value)
            )

        return filled
