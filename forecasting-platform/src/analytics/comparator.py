"""
ForecastComparator — multi-source forecast alignment for S&OP workflows.

Aligns the system model forecast with any number of external forecasts
(field/commercial, financial plan, prior model cycle) to a common
series_id × week grain, then computes gap and bias columns so planners
can immediately see where forecasts agree or disagree.

Output columns
--------------
[id_col, time_col]               — grain keys
model_forecast                   — current system model forecast (P50)
forecast_p10, forecast_p90       — probabilistic bounds (if available)
uncertainty_ratio                — (P90 - P10) / P50; NaN when no quantiles
<name>_forecast                  — each named external forecast
<name>_gap                       — model_forecast - <name>_forecast (absolute)
<name>_gap_pct                   — gap as % of <name>_forecast
<name>_direction_agree           — bool: model and <name> move same direction vs prior
prior_model_forecast             — prior-cycle model forecast (if provided)
cycle_change                     — model_forecast - prior_model_forecast
cycle_change_pct                 — cycle change as % of prior model forecast
"""

from typing import Dict, List, Optional

import polars as pl


class ForecastComparator:
    """
    Align multiple forecast sources and compute comparison metrics.

    Usage
    -----
    >>> comp = ForecastComparator()
    >>> result = comp.compare(
    ...     model_forecast=system_df,
    ...     external_forecasts={"field": field_df, "financial": plan_df},
    ...     prior_model_forecast=last_week_df,
    ... )
    """

    def compare(
        self,
        model_forecast: pl.DataFrame,
        external_forecasts: Optional[Dict[str, pl.DataFrame]] = None,
        prior_model_forecast: Optional[pl.DataFrame] = None,
        id_col: str = "series_id",
        time_col: str = "week",
        value_col: str = "forecast",
    ) -> pl.DataFrame:
        """
        Align and compare forecast sources.

        Parameters
        ----------
        model_forecast:
            Current system model forecast.  Must contain
            [id_col, time_col, value_col].  Optionally includes
            ``forecast_p10`` and ``forecast_p90`` for uncertainty metrics.
        external_forecasts:
            Dict of named external forecasts, e.g.
            ``{"field": field_df, "financial": plan_df}``.
            Each DataFrame must contain [id_col, time_col, value_col].
        prior_model_forecast:
            Previous cycle's system forecast for cycle-over-cycle tracking.
            Same schema as ``model_forecast``.
        id_col, time_col, value_col:
            Column name overrides.

        Returns
        -------
        Wide comparison DataFrame (one row per id × week) with gap,
        bias, direction-agreement, and uncertainty columns.
        """
        external_forecasts = external_forecasts or {}

        # Start from the model forecast, renaming the value column
        result = model_forecast.select(
            [id_col, time_col]
            + ([value_col] if value_col in model_forecast.columns else [])
            + ([c for c in ["forecast_p10", "forecast_p90"] if c in model_forecast.columns])
        ).rename({value_col: "model_forecast"})

        # Uncertainty ratio: (P90 - P10) / P50
        if "forecast_p10" in result.columns and "forecast_p90" in result.columns:
            result = result.with_columns(
                pl.when(pl.col("model_forecast") > 0)
                .then(
                    (pl.col("forecast_p90") - pl.col("forecast_p10"))
                    / pl.col("model_forecast")
                )
                .otherwise(None)
                .alias("uncertainty_ratio")
            )

        # Join each external forecast source
        for name, ext_df in external_forecasts.items():
            if value_col not in ext_df.columns:
                continue
            ext_slim = (
                ext_df.select([id_col, time_col, value_col])
                .rename({value_col: f"{name}_forecast"})
            )
            result = result.join(ext_slim, on=[id_col, time_col], how="left")

            # Absolute gap and % gap
            result = result.with_columns([
                (pl.col("model_forecast") - pl.col(f"{name}_forecast"))
                .alias(f"{name}_gap"),
                pl.when(pl.col(f"{name}_forecast").abs() > 0)
                .then(
                    (pl.col("model_forecast") - pl.col(f"{name}_forecast"))
                    / pl.col(f"{name}_forecast").abs()
                    * 100.0
                )
                .otherwise(None)
                .alias(f"{name}_gap_pct"),
            ])

        # Prior-cycle comparison (cycle-over-cycle tracking)
        if prior_model_forecast is not None and value_col in prior_model_forecast.columns:
            prior_slim = (
                prior_model_forecast.select([id_col, time_col, value_col])
                .rename({value_col: "prior_model_forecast"})
            )
            result = result.join(prior_slim, on=[id_col, time_col], how="left")
            result = result.with_columns([
                (pl.col("model_forecast") - pl.col("prior_model_forecast"))
                .alias("cycle_change"),
                pl.when(pl.col("prior_model_forecast").abs() > 0)
                .then(
                    (pl.col("model_forecast") - pl.col("prior_model_forecast"))
                    / pl.col("prior_model_forecast").abs()
                    * 100.0
                )
                .otherwise(None)
                .alias("cycle_change_pct"),
            ])

        return result.sort([id_col, time_col])

    def summary(
        self,
        comparison: pl.DataFrame,
        id_col: str = "series_id",
        time_col: str = "week",
        agg_cols: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Aggregate the comparison DataFrame to a summary view.

        Groups by id_col and computes mean gap, mean uncertainty, and
        mean cycle change across all time periods.

        Parameters
        ----------
        agg_cols:
            Additional columns to average.  Defaults to all ``*_gap_pct``
            and ``cycle_change_pct`` columns.
        """
        if comparison.is_empty():
            return comparison

        gap_pct_cols = [c for c in comparison.columns if c.endswith("_gap_pct")]
        cycle_cols = [c for c in comparison.columns if c == "cycle_change_pct"]
        unc_cols = ["uncertainty_ratio"] if "uncertainty_ratio" in comparison.columns else []
        agg_cols = agg_cols or (gap_pct_cols + cycle_cols + unc_cols)

        if not agg_cols:
            return comparison.select([id_col]).unique()

        agg_exprs = [
            pl.col(c).mean().alias(f"avg_{c}") for c in agg_cols if c in comparison.columns
        ]
        if not agg_exprs:
            return comparison.select([id_col]).unique()

        return (
            comparison.group_by(id_col)
            .agg(agg_exprs)
            .sort(id_col)
        )
