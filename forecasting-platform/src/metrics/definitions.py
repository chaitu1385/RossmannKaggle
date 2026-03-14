"""
Forecast accuracy metric definitions.

Primary metrics:
  - WMAPE  (Weighted Mean Absolute Percentage Error)
  - Normalized Bias  (directional error — over vs. under forecasting)

Secondary metrics (available for deeper analysis):
  - MAPE, MAE, RMSE

All functions operate on Polars Series for vectorised computation.
"""

from typing import Callable, Dict, List, Optional

import polars as pl

# ── Metric functions ──────────────────────────────────────────────────────────

def wmape(actual: pl.Series, forecast: pl.Series) -> float:
    """
    Weighted Mean Absolute Percentage Error.

    WMAPE = Σ|actual - forecast| / Σ|actual|

    - Volume-weighted: high-movers naturally dominate.
    - Handles zeros better than MAPE (no per-row division).
    - Range: [0, ∞), lower is better.
    """
    abs_error = (actual - forecast).abs().sum()
    abs_actual = actual.abs().sum()
    if abs_actual == 0:
        return float("inf")
    return float(abs_error / abs_actual)


def normalized_bias(actual: pl.Series, forecast: pl.Series) -> float:
    """
    Normalized Bias — measures systematic over/under-forecasting.

    Bias = Σ(forecast - actual) / Σ|actual|

    - Positive → over-forecasting (forecast > actual on average)
    - Negative → under-forecasting (forecast < actual on average)
    - Range: (-∞, +∞), closer to 0 is better.
    """
    signed_error = (forecast - actual).sum()
    abs_actual = actual.abs().sum()
    if abs_actual == 0:
        return 0.0
    return float(signed_error / abs_actual)


def mape(actual: pl.Series, forecast: pl.Series) -> float:
    """
    Mean Absolute Percentage Error.

    MAPE = mean(|actual - forecast| / |actual|)

    Excluded where actual == 0 to avoid division by zero.
    """
    mask = actual != 0
    if mask.sum() == 0:
        return float("inf")
    pct_errors = ((actual - forecast).abs() / actual.abs()).filter(mask)
    return float(pct_errors.mean())


def mae(actual: pl.Series, forecast: pl.Series) -> float:
    """Mean Absolute Error."""
    return float((actual - forecast).abs().mean())


def rmse(actual: pl.Series, forecast: pl.Series) -> float:
    """Root Mean Squared Error."""
    return float(((actual - forecast) ** 2).mean() ** 0.5)


# ── Registry ──────────────────────────────────────────────────────────────────

METRIC_REGISTRY: Dict[str, Callable[[pl.Series, pl.Series], float]] = {
    "wmape": wmape,
    "normalized_bias": normalized_bias,
    "mape": mape,
    "mae": mae,
    "rmse": rmse,
}


def compute_all_metrics(
    actual: pl.Series,
    forecast: pl.Series,
    metric_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute multiple metrics at once.

    Parameters
    ----------
    actual, forecast:
        Polars Series of equal length.
    metric_names:
        Which metrics to compute.  Defaults to all registered.

    Returns
    -------
    Dict mapping metric name → computed value.
    """
    if metric_names is None:
        metric_names = list(METRIC_REGISTRY.keys())

    results = {}
    for name in metric_names:
        fn = METRIC_REGISTRY.get(name)
        if fn is None:
            raise KeyError(
                f"Unknown metric {name!r}. Available: {list(METRIC_REGISTRY.keys())}"
            )
        results[name] = fn(actual, forecast)
    return results
