"""
CostEstimator — compute cost tracking per pipeline run.

Integrates with ``MetricsEmitter`` timer data to estimate cloud compute
costs based on a configurable per-second rate.

Usage
-----
>>> from src.observability.cost import CostEstimator
>>> estimator = CostEstimator(cost_per_second=0.0001)
>>> estimator.record_model("lgbm_direct", fit_seconds=12.5, predict_seconds=3.2)
>>> estimator.record_model("auto_arima", fit_seconds=45.0, predict_seconds=1.0)
>>> estimate = estimator.build_estimate(run_id="abc123", series_count=1200)
>>> print(estimate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CostEstimate:
    """
    Cost estimate for a single pipeline run.

    Attributes
    ----------
    run_id:
        Pipeline run identifier.
    total_seconds:
        Total wall-clock seconds across all models.
    model_seconds:
        Per-model breakdown of compute time.
    series_count:
        Number of series processed.
    cost_per_second:
        Cloud cost rate (e.g. $/second for the compute tier).
    """
    run_id: str = ""
    total_seconds: float = 0.0
    model_seconds: Dict[str, float] = field(default_factory=dict)
    series_count: int = 0
    cost_per_second: float = 0.0

    @property
    def estimated_cost(self) -> float:
        """Estimated cloud cost for this run."""
        return self.total_seconds * self.cost_per_second

    @property
    def cost_per_series(self) -> float:
        """Average cost per series (0 if no series)."""
        if self.series_count == 0:
            return 0.0
        return self.estimated_cost / self.series_count

    def as_dict(self) -> Dict[str, Any]:
        """Serializable dict for manifest / audit log inclusion."""
        return {
            "run_id": self.run_id,
            "total_seconds": round(self.total_seconds, 3),
            "estimated_cost": round(self.estimated_cost, 6),
            "cost_per_series": round(self.cost_per_series, 6),
            "series_count": self.series_count,
            "model_seconds": {
                k: round(v, 3) for k, v in self.model_seconds.items()
            },
        }

    def __str__(self) -> str:
        return (
            f"CostEstimate(run_id={self.run_id}, "
            f"total={self.total_seconds:.1f}s, "
            f"cost=${self.estimated_cost:.4f}, "
            f"series={self.series_count})"
        )


class CostEstimator:
    """
    Track compute time per model and estimate cloud costs.

    Parameters
    ----------
    cost_per_second:
        Cloud cost rate in $/second.  Set to 0 for logging-only mode.
    """

    def __init__(self, cost_per_second: float = 0.0):
        self.cost_per_second = cost_per_second
        self._model_times: Dict[str, float] = {}

    def record_model(
        self,
        model_name: str,
        fit_seconds: float = 0.0,
        predict_seconds: float = 0.0,
    ) -> None:
        """
        Record compute time for a model.

        Parameters
        ----------
        model_name:
            Forecaster name.
        fit_seconds:
            Wall-clock time for ``fit()``.
        predict_seconds:
            Wall-clock time for ``predict()``.
        """
        total = fit_seconds + predict_seconds
        self._model_times[model_name] = (
            self._model_times.get(model_name, 0.0) + total
        )

    def record_from_emitter(self, emitter) -> None:
        """
        Import timer metrics from a ``MetricsEmitter`` instance.

        Looks for metrics matching ``*model_fit_duration_seconds``
        and ``*model_predict_duration_seconds``.
        """
        for record in emitter.recorded:
            if record["type"] != "timer":
                continue
            name = record["metric"]
            model = record.get("model", "")
            if "model_fit" in name or "model_predict" in name:
                self._model_times[model] = (
                    self._model_times.get(model, 0.0) + record["value"]
                )

    def build_estimate(
        self,
        run_id: str = "",
        series_count: int = 0,
    ) -> CostEstimate:
        """
        Build a ``CostEstimate`` from recorded model times.

        Parameters
        ----------
        run_id:
            Pipeline run identifier.
        series_count:
            Number of series processed.
        """
        total = sum(self._model_times.values())
        return CostEstimate(
            run_id=run_id,
            total_seconds=total,
            model_seconds=dict(self._model_times),
            series_count=series_count,
            cost_per_second=self.cost_per_second,
        )

    def reset(self) -> None:
        """Clear all recorded times."""
        self._model_times.clear()
