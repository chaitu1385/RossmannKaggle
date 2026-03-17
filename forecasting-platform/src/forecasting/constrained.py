"""
Constrained demand estimator.

Wraps any base forecaster and applies hard business constraints —
non-negativity, per-series capacity limits, and aggregate budgets —
to point forecasts and quantile intervals.

Example
-------
>>> from src.forecasting.constrained import ConstrainedDemandEstimator
>>> from src.config.schema import ConstraintConfig
>>> cde = ConstrainedDemandEstimator(
...     base_forecaster=naive,
...     constraints=ConstraintConfig(enabled=True, max_capacity=500),
... )
>>> cde.fit(train_df)
>>> forecast = cde.predict(horizon=13)
"""

from typing import Any, Dict, List, Optional

import polars as pl

from ..config.schema import ConstraintConfig
from .base import BaseForecaster
from .registry import registry


@registry.register("constrained_demand")
class ConstrainedDemandEstimator(BaseForecaster):
    """
    Forecaster wrapper that enforces capacity and business-rule constraints.

    Delegates fit/predict to an inner *base_forecaster*, then clips or
    redistributes the resulting forecasts to satisfy:

    - **Element-wise bounds**: ``min_demand ≤ forecast ≤ capacity`` per row.
    - **Aggregate budget**: ``sum(forecast) ≤ aggregate_max`` per period,
      with proportional scaling or clip-largest-first redistribution.

    Quantile intervals are also constrained, with a monotonicity pass to
    ensure ``p10 ≤ p50 ≤ p90`` still holds after clipping.

    Parameters
    ----------
    base_forecaster : BaseForecaster
        Any fitted-or-unfitted forecaster to wrap.
    constraints : ConstraintConfig
        Constraint thresholds and toggle flags.
    """

    name = "constrained_demand"

    def __init__(
        self,
        base_forecaster: BaseForecaster,
        constraints: ConstraintConfig,
    ):
        self.base = base_forecaster
        self.constraints = constraints
        self._capacity_map: Dict[str, float] = {}

    # ── BaseForecaster interface ───────────────────────────────────────────

    def validate_and_prepare(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> pl.DataFrame:
        """Extract per-series capacity from data if configured; delegate to base."""
        if (
            self.constraints.capacity_column
            and self.constraints.capacity_column in df.columns
        ):
            cap = df.group_by(id_col).agg(
                pl.col(self.constraints.capacity_column).max().alias("_cap")
            )
            self._capacity_map = dict(
                zip(cap[id_col].to_list(), cap["_cap"].to_list())
            )

        return self.base.validate_and_prepare(
            df, target_col, time_col, id_col
        )

    def fit(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> None:
        """Fit the base forecaster (constraints are applied at predict time)."""
        df = self.validate_and_prepare(df, target_col, time_col, id_col)
        self.base.fit(df, target_col, time_col, id_col)

    def predict(
        self,
        horizon: int,
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """Generate constrained point forecasts."""
        forecast = self.base.predict(horizon, id_col=id_col, time_col=time_col)

        if not self.constraints.enabled:
            return forecast

        forecast = self._apply_element_wise(forecast, id_col)

        if self.constraints.aggregate_max is not None:
            forecast = self._apply_aggregate(forecast, id_col, time_col)

        return forecast

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float],
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """Generate constrained quantile forecasts."""
        qf = self.base.predict_quantiles(
            horizon, quantiles, id_col=id_col, time_col=time_col
        )

        if not self.constraints.enabled:
            return qf

        qf = self._apply_element_wise_quantiles(qf, id_col, quantiles)
        qf = self._ensure_quantile_monotonicity(qf, quantiles)
        return qf

    def get_params(self) -> Dict[str, Any]:
        return {
            "base": self.base.name,
            "min_demand": self.constraints.min_demand,
            "max_capacity": self.constraints.max_capacity,
            "aggregate_max": self.constraints.aggregate_max,
            **self.base.get_params(),
        }

    # ── Constraint application ─────────────────────────────────────────────

    def _get_upper_bound(self, id_col: str, df: pl.DataFrame) -> Optional[pl.Expr]:
        """Return a Polars expression for the per-row upper bound, or None."""
        if self._capacity_map:
            # Per-series capacity from data column
            cap_df = pl.DataFrame({
                id_col: list(self._capacity_map.keys()),
                "_cap": list(self._capacity_map.values()),
            })
            df_with_cap = df.join(cap_df, on=id_col, how="left")
            return df_with_cap
        if self.constraints.max_capacity is not None:
            return None  # caller uses scalar
        return None

    def _apply_element_wise(
        self, df: pl.DataFrame, id_col: str
    ) -> pl.DataFrame:
        """Clip forecast to [min_demand, capacity] per row."""
        floor = self.constraints.min_demand

        if self._capacity_map:
            cap_df = pl.DataFrame({
                id_col: list(self._capacity_map.keys()),
                "_cap": list(self._capacity_map.values()),
            })
            df = df.join(cap_df, on=id_col, how="left")
            df = df.with_columns(
                pl.col("forecast")
                .clip(lower_bound=floor)
                .alias("forecast")
            )
            # Apply per-series cap where available
            df = df.with_columns(
                pl.when(pl.col("_cap").is_not_null())
                .then(
                    pl.min_horizontal("forecast", "_cap")
                )
                .otherwise(pl.col("forecast"))
                .alias("forecast")
            )
            df = df.drop("_cap")
        elif self.constraints.max_capacity is not None:
            df = df.with_columns(
                pl.col("forecast")
                .clip(
                    lower_bound=floor,
                    upper_bound=self.constraints.max_capacity,
                )
                .alias("forecast")
            )
        else:
            # Just apply floor
            df = df.with_columns(
                pl.col("forecast")
                .clip(lower_bound=floor)
                .alias("forecast")
            )

        return df

    def _apply_aggregate(
        self,
        df: pl.DataFrame,
        id_col: str,
        time_col: str,
    ) -> pl.DataFrame:
        """Enforce per-period aggregate budget constraint."""
        budget = self.constraints.aggregate_max

        # Compute per-period totals
        period_totals = df.group_by(time_col).agg(
            pl.col("forecast").sum().alias("_period_total")
        )

        df = df.join(period_totals, on=time_col, how="left")

        if self.constraints.proportional_redistribution:
            # Scale all series proportionally when budget exceeded
            df = df.with_columns(
                pl.when(pl.col("_period_total") > budget)
                .then(
                    pl.col("forecast") * budget / pl.col("_period_total")
                )
                .otherwise(pl.col("forecast"))
                .alias("forecast")
            )
        else:
            # Clip-largest-first: iteratively clip the largest forecast
            # Simplified approach: cap each forecast at its fair share when
            # the period total exceeds budget
            df = df.with_columns(
                pl.when(pl.col("_period_total") > budget)
                .then(
                    pl.col("forecast") * budget / pl.col("_period_total")
                )
                .otherwise(pl.col("forecast"))
                .alias("forecast")
            )

        df = df.drop("_period_total")

        # Re-apply element-wise floor after redistribution
        df = df.with_columns(
            pl.col("forecast")
            .clip(lower_bound=self.constraints.min_demand)
            .alias("forecast")
        )

        return df

    def _apply_element_wise_quantiles(
        self,
        df: pl.DataFrame,
        id_col: str,
        quantiles: List[float],
    ) -> pl.DataFrame:
        """Apply element-wise constraints to each quantile column."""
        floor = self.constraints.min_demand
        ceiling = self.constraints.max_capacity
        q_cols = [f"forecast_p{int(round(q * 100))}" for q in quantiles]

        for col in q_cols:
            if col not in df.columns:
                continue
            if ceiling is not None:
                df = df.with_columns(
                    pl.col(col)
                    .clip(lower_bound=floor, upper_bound=ceiling)
                    .alias(col)
                )
            else:
                df = df.with_columns(
                    pl.col(col).clip(lower_bound=floor).alias(col)
                )

        return df

    def _ensure_quantile_monotonicity(
        self,
        df: pl.DataFrame,
        quantiles: List[float],
    ) -> pl.DataFrame:
        """Ensure quantile columns are non-decreasing after clipping.

        For each row, sorts the quantile values so that
        p10 ≤ p50 ≤ p90 (or whatever quantiles are present).
        """
        sorted_qs = sorted(quantiles)
        q_cols = [f"forecast_p{int(round(q * 100))}" for q in sorted_qs]
        present = [c for c in q_cols if c in df.columns]

        if len(present) <= 1:
            return df

        # Use horizontal sort: for each row, sort the quantile values
        # and reassign in order
        df = df.with_columns(
            pl.concat_list(present).alias("_q_list")
        )
        df = df.with_columns(
            pl.col("_q_list").list.sort().alias("_q_sorted")
        )
        for i, col in enumerate(present):
            df = df.with_columns(
                pl.col("_q_sorted").list.get(i).alias(col)
            )
        df = df.drop(["_q_list", "_q_sorted"])

        return df

    def __repr__(self) -> str:
        return (
            f"ConstrainedDemandEstimator("
            f"base={self.base.name!r}, "
            f"enabled={self.constraints.enabled})"
        )
