"""
Weighted ensemble forecaster.

Combines multiple base forecasters by computing a weighted average of their
point forecasts and quantile intervals.  Weights are typically derived from
backtest WMAPE via ``ChampionSelector.compute_ensemble_weights()``.

Instead of picking a single winning model (winner-take-all champion selection),
this provides soft blending: lower-WMAPE models receive proportionally higher
weight while every model still contributes.

Example
-------
>>> from src.forecasting.ensemble import WeightedEnsembleForecaster
>>> ensemble = WeightedEnsembleForecaster(
...     forecasters=[lgbm, arima],
...     weights={"lgbm_direct": 0.7, "auto_arima": 0.3},
... )
>>> ensemble.fit(train_df)
>>> forecast = ensemble.predict(horizon=13)
"""

from typing import Any, Dict, List, Optional

import polars as pl

from .base import BaseForecaster
from .registry import registry


@registry.register("weighted_ensemble")
class WeightedEnsembleForecaster(BaseForecaster):
    """
    Weighted average of multiple forecasters.

    Point forecasts and quantile intervals are blended using the provided
    weights.  Weights need not sum to exactly 1 — they are normalised
    internally.  Models not present in ``weights`` receive zero weight and
    are excluded from the blend.

    Parameters
    ----------
    forecasters:
        Fitted or unfitted ``BaseForecaster`` instances to blend.
    weights:
        Mapping from ``forecaster.name`` to a non-negative blend weight.
        E.g. ``{"lgbm_direct": 0.7, "auto_arima": 0.3}``.
    """

    name = "weighted_ensemble"

    def __init__(
        self,
        forecasters: List[BaseForecaster],
        weights: Dict[str, float],
    ):
        if not forecasters:
            raise ValueError("WeightedEnsembleForecaster requires at least one forecaster.")

        self._forecasters = forecasters

        # Normalize weights so they sum to 1 over *active* models
        raw = {f.name: max(0.0, weights.get(f.name, 0.0)) for f in forecasters}
        total = sum(raw.values())
        if total <= 0:
            # Uniform fallback
            n = len(forecasters)
            self._weights: Dict[str, float] = {f.name: 1.0 / n for f in forecasters}
        else:
            self._weights = {k: v / total for k, v in raw.items()}

    # ── BaseForecaster interface ───────────────────────────────────────────

    def fit(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> None:
        """Fit all sub-forecasters on the same training data."""
        for forecaster in self._forecasters:
            forecaster.fit(df, target_col=target_col, time_col=time_col, id_col=id_col)

    def predict(
        self,
        horizon: int,
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """
        Weighted average point forecast.

        Stacks each model's weighted predictions and sums across models
        for each (series, week) pair.  Because weights sum to 1 this is
        equivalent to a weighted mean.
        """
        weighted_parts: List[pl.DataFrame] = []
        for f in self._forecasters:
            w = self._weights.get(f.name, 0.0)
            if w <= 0:
                continue
            preds = f.predict(horizon, id_col=id_col, time_col=time_col)
            weighted_parts.append(
                preds.with_columns((pl.col("forecast") * w).alias("forecast"))
            )

        if not weighted_parts:
            return pl.DataFrame(
                schema={id_col: pl.Utf8, time_col: pl.Date, "forecast": pl.Float64}
            )

        return (
            pl.concat(weighted_parts, how="vertical_relaxed")
            .group_by([id_col, time_col])
            .agg(pl.col("forecast").sum())
            .sort([id_col, time_col])
        )

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float],
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """
        Weighted blend of each sub-forecaster's quantile predictions.

        Each quantile column is independently blended with the same weights
        as the point forecast.
        """
        # Collect per-model quantile DataFrames
        q_cols = [f"forecast_p{int(round(q * 100))}" for q in quantiles]

        weighted_parts: List[pl.DataFrame] = []
        for f in self._forecasters:
            w = self._weights.get(f.name, 0.0)
            if w <= 0:
                continue
            qdf = f.predict_quantiles(horizon, quantiles, id_col=id_col, time_col=time_col)
            # Scale every quantile column by the weight
            scaled = qdf.with_columns([
                (pl.col(c) * w).alias(c) for c in q_cols if c in qdf.columns
            ])
            weighted_parts.append(scaled)

        if not weighted_parts:
            schema: Dict[str, Any] = {id_col: pl.Utf8, time_col: pl.Date}
            for c in q_cols:
                schema[c] = pl.Float64
            return pl.DataFrame(schema=schema)

        stacked = pl.concat(weighted_parts, how="vertical_relaxed")
        return (
            stacked
            .group_by([id_col, time_col])
            .agg([pl.col(c).sum() for c in q_cols if c in stacked.columns])
            .sort([id_col, time_col])
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "models": [f.name for f in self._forecasters],
            "weights": self._weights,
        }

    def __repr__(self) -> str:
        w_str = ", ".join(f"{k}={v:.3f}" for k, v in self._weights.items())
        return f"WeightedEnsembleForecaster({w_str})"
