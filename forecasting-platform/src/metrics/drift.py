"""
Forecast Drift & Monitoring
(Design document §12 / Phase 2)

Detects when forecast performance has degraded relative to a historical
baseline.  Three orthogonal signals are monitored:

  1. Accuracy drift   — WMAPE of the recent window vs a historical baseline.
  2. Bias drift       — Normalised bias shift from a previously neutral baseline.
  3. Volume anomaly   — Actuals volume z-score vs trailing mean (data staleness,
                        demand shocks, or upstream pipeline breaks).

All computations are Polars-native (no Spark, no pandas) and operate on the
standard MetricStore schema (see ``src/metrics/store.py``).

Usage
-----
>>> from src.metrics.drift import ForecastDriftDetector, DriftConfig
>>> detector = ForecastDriftDetector(DriftConfig())
>>> alerts = detector.detect(metrics_df)
>>> for alert in alerts:
...     print(alert)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import polars as pl

# ── Severity ───────────────────────────────────────────────────────────────────

class DriftSeverity(str, Enum):
    WARNING  = "warning"    # degraded but within tolerable range
    CRITICAL = "critical"   # severe degradation requiring immediate action


# ── Alert dataclass ────────────────────────────────────────────────────────────

@dataclass
class DriftAlert:
    """A single drift event for one series / metric."""
    series_id:       str
    metric:          str             # "accuracy", "bias", "volume"
    severity:        DriftSeverity
    current_value:   float
    baseline_value:  float
    message:         str

    def __str__(self) -> str:
        return (
            f"[{self.severity.value.upper()}] {self.series_id} | {self.metric}: "
            f"current={self.current_value:.4f} baseline={self.baseline_value:.4f} — {self.message}"
        )


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class DriftConfig:
    """
    Configuration for drift detection thresholds and window sizes.

    Parameters
    ----------
    baseline_weeks:
        Number of trailing weeks used to establish the baseline metric value.
        Default: 26 (6 months).
    recent_weeks:
        Number of trailing weeks used to compute the current metric value.
        Must be < ``baseline_weeks``.  Default: 8.
    accuracy_warning_ratio:
        Alert at WARNING when ``current_wmape / baseline_wmape`` exceeds this.
        Default: 1.25 (25 % relative degradation).
    accuracy_critical_ratio:
        Alert at CRITICAL when ratio exceeds this.  Default: 1.50.
    bias_warning_threshold:
        Alert at WARNING when |normalised_bias| exceeds this.
        Default: 0.10 (10 % net over/under-forecast).
    bias_critical_threshold:
        Alert at CRITICAL when |normalised_bias| exceeds this.  Default: 0.25.
    volume_warning_zscore:
        Alert at WARNING when volume z-score |z| exceeds this.  Default: 2.0.
    volume_critical_zscore:
        Alert at CRITICAL when |z| exceeds this.  Default: 3.0.
    min_baseline_periods:
        Minimum data points required to compute a valid baseline.  Skips series
        with insufficient history.  Default: 4.
    """

    baseline_weeks:          int   = 26
    recent_weeks:            int   = 8
    accuracy_warning_ratio:  float = 1.25
    accuracy_critical_ratio: float = 1.50
    bias_warning_threshold:  float = 0.10
    bias_critical_threshold: float = 0.25
    volume_warning_zscore:   float = 2.0
    volume_critical_zscore:  float = 3.0
    min_baseline_periods:    int   = 4


# ── Detector ───────────────────────────────────────────────────────────────────

class ForecastDriftDetector:
    """
    Detects accuracy, bias, and volume drift in forecast performance.

    Parameters
    ----------
    config:
        ``DriftConfig`` controlling thresholds and window sizes.
    """

    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect(self, metrics_df: pl.DataFrame) -> List[DriftAlert]:
        """
        Run all drift detectors on a metric history DataFrame.

        Parameters
        ----------
        metrics_df:
            DataFrame with at least columns:
            ``[series_id, target_week, actual, forecast, wmape, normalized_bias]``.
            Matches the ``MetricStore`` schema.

        Returns
        -------
        List of ``DriftAlert`` objects sorted by severity (CRITICAL first).
        """
        if metrics_df.is_empty():
            return []

        alerts: List[DriftAlert] = []
        alerts.extend(self.detect_accuracy_drift(metrics_df))
        alerts.extend(self.detect_bias_drift(metrics_df))
        alerts.extend(self.detect_volume_anomaly(metrics_df))

        # Sort: CRITICAL first, then WARNING; within each severity by series_id
        severity_order = {DriftSeverity.CRITICAL: 0, DriftSeverity.WARNING: 1}
        alerts.sort(key=lambda a: (severity_order[a.severity], a.series_id))
        return alerts

    def detect_accuracy_drift(self, metrics_df: pl.DataFrame) -> List[DriftAlert]:
        """
        Compare per-series WMAPE in the recent window vs the baseline window.

        A series is flagged when:
            current_wmape / baseline_wmape > accuracy_warning_ratio   → WARNING
            current_wmape / baseline_wmape > accuracy_critical_ratio  → CRITICAL
        """
        cfg = self.config
        alerts: List[DriftAlert] = []

        for series_id, group in self._iter_series(metrics_df):
            baseline_df, recent_df = self._split_windows(group)
            if len(baseline_df) < cfg.min_baseline_periods or recent_df.is_empty():
                continue

            baseline_wmape = self._compute_wmape(baseline_df)
            recent_wmape   = self._compute_wmape(recent_df)

            if baseline_wmape <= 0:
                continue

            ratio = recent_wmape / baseline_wmape

            if ratio > cfg.accuracy_critical_ratio:
                alerts.append(DriftAlert(
                    series_id=series_id,
                    metric="accuracy",
                    severity=DriftSeverity.CRITICAL,
                    current_value=recent_wmape,
                    baseline_value=baseline_wmape,
                    message=(
                        f"WMAPE degraded {ratio:.1%} above baseline "
                        f"(threshold: {cfg.accuracy_critical_ratio:.0%}). "
                        "Model re-training recommended."
                    ),
                ))
            elif ratio > cfg.accuracy_warning_ratio:
                alerts.append(DriftAlert(
                    series_id=series_id,
                    metric="accuracy",
                    severity=DriftSeverity.WARNING,
                    current_value=recent_wmape,
                    baseline_value=baseline_wmape,
                    message=(
                        f"WMAPE degraded {ratio:.1%} above baseline "
                        f"(threshold: {cfg.accuracy_warning_ratio:.0%}). "
                        "Monitor closely."
                    ),
                ))

        return alerts

    def detect_bias_drift(self, metrics_df: pl.DataFrame) -> List[DriftAlert]:
        """
        Flag series whose recent normalised bias exceeds the alert thresholds.

        Unlike accuracy drift, bias is compared to zero (neutral), not to a
        historical baseline — a bias that was always 0.20 is fine; one that
        recently jumped to 0.20 from near-zero needs investigation.
        """
        cfg = self.config
        alerts: List[DriftAlert] = []

        for series_id, group in self._iter_series(metrics_df):
            baseline_df, recent_df = self._split_windows(group)
            if len(baseline_df) < cfg.min_baseline_periods or recent_df.is_empty():
                continue

            baseline_bias = self._compute_bias(baseline_df)
            recent_bias   = self._compute_bias(recent_df)

            # Alert on the recent absolute bias level
            abs_bias = abs(recent_bias)
            direction = "over-forecasting" if recent_bias > 0 else "under-forecasting"

            if abs_bias > cfg.bias_critical_threshold:
                alerts.append(DriftAlert(
                    series_id=series_id,
                    metric="bias",
                    severity=DriftSeverity.CRITICAL,
                    current_value=recent_bias,
                    baseline_value=baseline_bias,
                    message=(
                        f"Critical {direction}: normalised bias = {recent_bias:+.3f} "
                        f"(threshold: ±{cfg.bias_critical_threshold:.2f}). "
                        "Review forecast inputs immediately."
                    ),
                ))
            elif abs_bias > cfg.bias_warning_threshold:
                alerts.append(DriftAlert(
                    series_id=series_id,
                    metric="bias",
                    severity=DriftSeverity.WARNING,
                    current_value=recent_bias,
                    baseline_value=baseline_bias,
                    message=(
                        f"Systematic {direction}: normalised bias = {recent_bias:+.3f} "
                        f"(threshold: ±{cfg.bias_warning_threshold:.2f}). "
                        "Monitor trend."
                    ),
                ))

        return alerts

    def detect_volume_anomaly(self, metrics_df: pl.DataFrame) -> List[DriftAlert]:
        """
        Detect unusual actuals volume using a z-score against the baseline window.

        Alerts when the recent mean actuals is more than ``volume_*_zscore``
        standard deviations from the baseline mean.  This catches demand shocks,
        pipeline breaks, or data ingestion failures.
        """
        cfg = self.config
        alerts: List[DriftAlert] = []

        for series_id, group in self._iter_series(metrics_df):
            baseline_df, recent_df = self._split_windows(group)
            if len(baseline_df) < cfg.min_baseline_periods or recent_df.is_empty():
                continue

            baseline_actuals = baseline_df["actual"].drop_nulls()
            recent_actuals   = recent_df["actual"].drop_nulls()

            if len(baseline_actuals) < 2 or len(recent_actuals) == 0:
                continue

            baseline_mean = float(baseline_actuals.mean())
            baseline_std  = float(baseline_actuals.std())
            recent_mean   = float(recent_actuals.mean())

            if baseline_std <= 0:
                continue

            z_score = (recent_mean - baseline_mean) / baseline_std
            abs_z   = abs(z_score)
            direction = "spike" if z_score > 0 else "drop"

            if abs_z > cfg.volume_critical_zscore:
                alerts.append(DriftAlert(
                    series_id=series_id,
                    metric="volume",
                    severity=DriftSeverity.CRITICAL,
                    current_value=recent_mean,
                    baseline_value=baseline_mean,
                    message=(
                        f"Volume {direction}: recent mean={recent_mean:.2f} vs "
                        f"baseline mean={baseline_mean:.2f} (z={z_score:+.2f}, "
                        f"threshold: ±{cfg.volume_critical_zscore:.1f}). "
                        "Verify upstream data."
                    ),
                ))
            elif abs_z > cfg.volume_warning_zscore:
                alerts.append(DriftAlert(
                    series_id=series_id,
                    metric="volume",
                    severity=DriftSeverity.WARNING,
                    current_value=recent_mean,
                    baseline_value=baseline_mean,
                    message=(
                        f"Volume {direction}: recent mean={recent_mean:.2f} vs "
                        f"baseline mean={baseline_mean:.2f} (z={z_score:+.2f}, "
                        f"threshold: ±{cfg.volume_warning_zscore:.1f}). "
                        "Monitor demand pattern."
                    ),
                ))

        return alerts

    # ------------------------------------------------------------------
    # Convenience: summary report
    # ------------------------------------------------------------------

    def summary(self, metrics_df: pl.DataFrame) -> pl.DataFrame:
        """
        Return a Polars DataFrame summary of all drift alerts.

        Columns: series_id, metric, severity, current_value,
                 baseline_value, message.
        """
        alerts = self.detect(metrics_df)
        if not alerts:
            return pl.DataFrame(schema={
                "series_id":      pl.Utf8,
                "metric":         pl.Utf8,
                "severity":       pl.Utf8,
                "current_value":  pl.Float64,
                "baseline_value": pl.Float64,
                "message":        pl.Utf8,
            })

        return pl.DataFrame({
            "series_id":      [a.series_id      for a in alerts],
            "metric":         [a.metric         for a in alerts],
            "severity":       [a.severity.value for a in alerts],
            "current_value":  [a.current_value  for a in alerts],
            "baseline_value": [a.baseline_value for a in alerts],
            "message":        [a.message        for a in alerts],
        })

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _iter_series(self, df: pl.DataFrame):
        """Yield ``(series_id, group_df)`` sorted by ``target_week``."""
        for series_id in df["series_id"].unique().sort().to_list():
            group = (
                df.filter(pl.col("series_id") == series_id)
                .sort("target_week")
            )
            yield series_id, group

    def _split_windows(
        self, group: pl.DataFrame
    ):
        """
        Split a single-series DataFrame into (baseline_df, recent_df).

        ``recent_df``   = last ``recent_weeks`` rows.
        ``baseline_df`` = rows before the recent window, up to ``baseline_weeks``
                          earlier.
        """
        cfg = self.config
        n = len(group)

        recent_start  = max(0, n - cfg.recent_weeks)
        baseline_end  = recent_start
        baseline_start = max(0, recent_start - cfg.baseline_weeks)

        recent_df   = group.slice(recent_start, n - recent_start)
        baseline_df = group.slice(baseline_start, baseline_end - baseline_start)
        return baseline_df, recent_df

    @staticmethod
    def _compute_wmape(df: pl.DataFrame) -> float:
        """Compute WMAPE from actual/forecast columns."""
        actual   = df["actual"].drop_nulls()
        forecast = df["forecast"].drop_nulls()
        # align lengths
        min_len = min(len(actual), len(forecast))
        if min_len == 0:
            return 0.0
        actual   = actual[:min_len]
        forecast = forecast[:min_len]
        abs_actual = actual.abs().sum()
        if abs_actual == 0:
            return 0.0
        return float((actual - forecast).abs().sum() / abs_actual)

    @staticmethod
    def _compute_bias(df: pl.DataFrame) -> float:
        """Compute normalised bias from actual/forecast columns."""
        actual   = df["actual"].drop_nulls()
        forecast = df["forecast"].drop_nulls()
        min_len  = min(len(actual), len(forecast))
        if min_len == 0:
            return 0.0
        actual   = actual[:min_len]
        forecast = forecast[:min_len]
        abs_actual = actual.abs().sum()
        if abs_actual == 0:
            return 0.0
        return float((forecast - actual).sum() / abs_actual)
