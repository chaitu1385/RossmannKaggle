"""
Deep Data Profiler — comprehensive dataset profiling beyond basic stats.

Extends existing ``DataQualityAnalyzer`` and ``ForecastabilityAnalyzer``
with distribution profiling, temporal pattern detection, correlation
analysis, complementary completeness checks, and anomaly scanning.

All functions accept Polars DataFrames.

Usage::

    from src.profiler import (
        profile_distributions, profile_temporal_patterns,
        profile_correlations, profile_completeness, profile_anomalies,
        run_deep_profile,
    )

    report = run_deep_profile(df, date_col="week", metric_cols=["quantity"])
"""

from __future__ import annotations

import logging
import math
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Distribution profiling
# ---------------------------------------------------------------------------

def profile_distributions(
    df: pl.DataFrame,
    numeric_cols: Optional[Sequence[str]] = None,
) -> list[dict]:
    """Profile distributions of numeric columns.

    Args:
        df: Polars DataFrame.
        numeric_cols: Columns to profile. If None, auto-detects numeric columns.

    Returns:
        List of dicts per column with: ``name``, ``n_values``, ``n_unique``,
        ``mean``, ``median``, ``std``, ``skewness``, ``kurtosis``,
        ``percentiles``, ``iqr``, ``n_outliers_iqr``, ``shape``,
        ``recommended_transform``.
    """
    if numeric_cols is None:
        numeric_cols = [
            c for c in df.columns
            if df[c].dtype in (
                pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            )
        ]

    results = []
    for col in numeric_cols:
        if col not in df.columns:
            continue

        vals = df[col].drop_nulls().cast(pl.Float64)
        n = vals.len()
        if n == 0:
            results.append({"name": col, "n_values": 0, "shape": "empty"})
            continue

        arr = vals.to_numpy()
        mean = float(vals.mean())
        std = float(vals.std()) if n > 1 else 0.0
        median = float(vals.median())
        n_unique = vals.n_unique()

        # Percentiles
        pcts = {}
        for p in [1, 5, 25, 75, 95, 99]:
            pcts[f"p{p}"] = float(vals.quantile(p / 100))

        iqr = pcts["p75"] - pcts["p25"]

        # Outliers (IQR method)
        lower = pcts["p25"] - 1.5 * iqr
        upper = pcts["p75"] + 1.5 * iqr
        n_outliers = int(vals.filter((vals < lower) | (vals > upper)).len())

        # Skewness and kurtosis via numpy
        skewness = float(_skewness(arr)) if n > 2 else 0.0
        kurtosis = float(_kurtosis(arr)) if n > 3 else 0.0

        shape = _classify_shape(arr, skewness, kurtosis, n_unique)
        transform = _recommend_transform(arr, skewness)

        results.append({
            "name": col,
            "n_values": n,
            "n_unique": n_unique,
            "mean": round(mean, 4),
            "median": round(median, 4),
            "std": round(std, 4),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
            "percentiles": {k: round(v, 4) for k, v in pcts.items()},
            "iqr": round(iqr, 4),
            "n_outliers_iqr": n_outliers,
            "shape": shape,
            "recommended_transform": transform,
        })

    return results


# ---------------------------------------------------------------------------
# Temporal patterns
# ---------------------------------------------------------------------------

def profile_temporal_patterns(
    df: pl.DataFrame,
    date_col: str,
    metric_cols: Optional[Sequence[str]] = None,
    freq: str = "1w",
) -> dict:
    """Profile temporal patterns: gaps, day-of-week, monthly, trend, seasonality.

    Args:
        df: Polars DataFrame sorted by date.
        date_col: Date column.
        metric_cols: Metric columns to analyze. Auto-detects if None.
        freq: Expected frequency ("1d", "1w", "1mo").

    Returns:
        dict with ``date_range``, ``expected_periods``, ``actual_periods``,
        ``coverage_pct``, ``gaps``, ``day_of_week_pattern``,
        ``monthly_pattern``, ``trend``, ``seasonality_detected``.
    """
    if date_col not in df.columns:
        return {"error": f"Column '{date_col}' not found"}

    dates = df[date_col].drop_nulls().unique().sort()
    if dates.is_empty():
        return {"error": "No dates found"}

    min_date = dates[0]
    max_date = dates[-1]
    actual_periods = dates.len()

    # Estimate expected periods
    if freq.endswith("w"):
        delta_days = (max_date - min_date).days
        expected_periods = delta_days // 7 + 1
    elif freq.endswith("mo"):
        months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1
        expected_periods = months
    else:  # daily
        expected_periods = (max_date - min_date).days + 1

    coverage = actual_periods / expected_periods if expected_periods > 0 else 0.0

    # Gap detection
    gaps = []
    if actual_periods > 1:
        diffs = dates.diff().drop_nulls()
        for i, d in enumerate(diffs):
            gap_days = d.days if hasattr(d, "days") else int(d.total_seconds() / 86400)
            expected_gap = 7 if freq.endswith("w") else (30 if freq.endswith("mo") else 1)
            if gap_days > expected_gap * 1.5:
                gaps.append({
                    "start": str(dates[i]),
                    "end": str(dates[i + 1]),
                    "gap_days": gap_days,
                })

    # Day-of-week pattern (only for daily data)
    dow_pattern = {}
    if not freq.endswith("mo"):
        try:
            dow = df.with_columns(pl.col(date_col).dt.weekday().alias("_dow"))
            dow_counts = dow.group_by("_dow").agg(pl.count().alias("n")).sort("_dow")
            dow_pattern = dict(zip(
                dow_counts["_dow"].to_list(),
                dow_counts["n"].to_list(),
            ))
        except Exception:
            logger.debug("Failed to compute day-of-week pattern", exc_info=True)

    # Monthly pattern
    monthly_pattern = {}
    try:
        mo = df.with_columns(pl.col(date_col).dt.month().alias("_month"))
        if metric_cols:
            first_metric = metric_cols[0]
            if first_metric in df.columns:
                mo_agg = (
                    mo.group_by("_month")
                    .agg(pl.col(first_metric).mean().alias("mean"))
                    .sort("_month")
                )
                monthly_pattern = dict(zip(
                    mo_agg["_month"].to_list(),
                    [round(v, 2) for v in mo_agg["mean"].to_list()],
                ))
    except Exception:
        logger.debug("Failed to compute monthly pattern", exc_info=True)

    # Trend detection
    trend = "insufficient_data"
    if metric_cols and actual_periods >= 4:
        first_metric = metric_cols[0]
        if first_metric in df.columns:
            sorted_df = df.sort(date_col)
            values = sorted_df[first_metric].drop_nulls().cast(pl.Float64).to_numpy()
            if len(values) >= 4:
                trend = _detect_trend(values)

    # Seasonality detection
    seasonality = False
    if monthly_pattern and len(monthly_pattern) >= 6:
        vals = list(monthly_pattern.values())
        mean_v = sum(vals) / len(vals)
        if mean_v > 0:
            cv = (sum((v - mean_v) ** 2 for v in vals) / len(vals)) ** 0.5 / mean_v
            seasonality = cv > 0.20

    return {
        "date_range": {"min": str(min_date), "max": str(max_date)},
        "expected_periods": expected_periods,
        "actual_periods": actual_periods,
        "coverage_pct": round(coverage * 100, 1),
        "gaps": gaps,
        "day_of_week_pattern": dow_pattern,
        "monthly_pattern": monthly_pattern,
        "trend": trend,
        "seasonality_detected": seasonality,
    }


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

def profile_correlations(
    df: pl.DataFrame,
    numeric_cols: Optional[Sequence[str]] = None,
    threshold: float = 0.5,
) -> list[dict]:
    """Find strong correlations between numeric columns.

    Args:
        df: Polars DataFrame.
        numeric_cols: Columns to check.  Auto-detects if None.
        threshold: Minimum |correlation| to report.

    Returns:
        List of dicts with ``col_a``, ``col_b``, ``correlation``,
        ``strength``, ``direction``.
    """
    if numeric_cols is None:
        numeric_cols = [
            c for c in df.columns
            if df[c].dtype in (
                pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            )
        ]

    if len(numeric_cols) < 2:
        return []

    # Use numpy for correlation matrix
    mat = df.select(numeric_cols).drop_nulls().to_numpy()
    if mat.shape[0] < 3:
        return []

    corr = np.corrcoef(mat.T)
    results = []

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            r = corr[i, j]
            if np.isnan(r):
                continue
            if abs(r) >= threshold:
                strength = (
                    "strong" if abs(r) >= 0.7
                    else "moderate"
                )
                results.append({
                    "col_a": numeric_cols[i],
                    "col_b": numeric_cols[j],
                    "correlation": round(float(r), 4),
                    "abs_correlation": round(abs(float(r)), 4),
                    "strength": strength,
                    "direction": "positive" if r > 0 else "negative",
                })

    results.sort(key=lambda x: x["abs_correlation"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Completeness profiling
# ---------------------------------------------------------------------------

def profile_completeness(df: pl.DataFrame) -> list[dict]:
    """Profile completeness of every column.

    Args:
        df: Polars DataFrame.

    Returns:
        List of dicts per column with: ``name``, ``null_count``,
        ``null_pct``, ``status``, ``zero_count``, ``empty_string_count``,
        ``constant``.
    """
    total = len(df)
    results = []

    for col in df.columns:
        null_count = df[col].null_count()
        null_pct = round(null_count / total * 100, 2) if total > 0 else 0

        if null_pct == 0:
            status = "COMPLETE"
        elif null_pct < 5:
            status = "GOOD"
        elif null_pct < 20:
            status = "WARNING"
        else:
            status = "CRITICAL"

        # Zero count (numeric)
        zero_count = 0
        dtype = df[col].dtype
        if dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
                     pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            zero_count = df.filter(pl.col(col) == 0).height

        # Empty string count
        empty_str = 0
        if dtype in (pl.Utf8, pl.Categorical):
            empty_str = df.filter(
                (pl.col(col) == "") | (pl.col(col).str.strip_chars() == "")
            ).height

        # Constant check
        n_unique = df[col].drop_nulls().n_unique()
        constant = n_unique <= 1

        results.append({
            "name": col,
            "null_count": null_count,
            "null_pct": null_pct,
            "status": status,
            "zero_count": zero_count,
            "empty_string_count": empty_str,
            "constant": constant,
        })

    return results


# ---------------------------------------------------------------------------
# Anomaly scanning
# ---------------------------------------------------------------------------

def profile_anomalies(
    df: pl.DataFrame,
    date_col: Optional[str] = None,
    metric_cols: Optional[Sequence[str]] = None,
    window: int = 14,
    threshold: float = 2.0,
) -> dict:
    """Scan for anomalies in metric columns using rolling statistics.

    Args:
        df: Polars DataFrame.
        date_col: Date column for time-series ordering.
        metric_cols: Metric columns to scan.  Auto-detects if None.
        window: Rolling window size.
        threshold: Z-score multiplier for anomaly bounds.

    Returns:
        dict with ``metrics_scanned``, ``total_anomalies``, ``by_metric``.
    """
    if metric_cols is None:
        metric_cols = [
            c for c in df.columns
            if df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
            and c != date_col
        ]

    sorted_df = df.sort(date_col) if date_col and date_col in df.columns else df
    by_metric: list[dict] = []
    total = 0

    for col in metric_cols:
        if col not in sorted_df.columns:
            continue

        vals = sorted_df[col].cast(pl.Float64)
        if vals.len() < window + 1:
            continue

        rolling_mean = vals.rolling_mean(window_size=window)
        rolling_std = vals.rolling_std(window_size=window)

        anomalies = []
        for i in range(window, vals.len()):
            m = rolling_mean[i]
            s = rolling_std[i]
            v = vals[i]
            if m is None or s is None or s == 0 or v is None:
                continue
            z = abs((v - m) / s)
            if z > threshold:
                row_info = {"index": i, "value": float(v), "z_score": round(float(z), 2)}
                if date_col and date_col in sorted_df.columns:
                    row_info["date"] = str(sorted_df[date_col][i])
                anomalies.append(row_info)

        by_metric.append({
            "column": col,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies[:20],
        })
        total += len(anomalies)

    return {
        "metrics_scanned": len(metric_cols),
        "total_anomalies": total,
        "by_metric": by_metric,
    }


# ---------------------------------------------------------------------------
# Convenience orchestrator
# ---------------------------------------------------------------------------

def run_deep_profile(
    df: pl.DataFrame,
    date_col: Optional[str] = None,
    metric_cols: Optional[Sequence[str]] = None,
    freq: str = "1w",
) -> dict:
    """Run all profiling checks in one call.

    Args:
        df: Polars DataFrame.
        date_col: Date column.
        metric_cols: Metric columns.
        freq: Expected date frequency.

    Returns:
        dict with ``distributions``, ``temporal``, ``correlations``,
        ``completeness``, ``anomalies``, ``summary``.
    """
    distributions = profile_distributions(df, numeric_cols=metric_cols)
    completeness = profile_completeness(df)
    correlations = profile_correlations(df, numeric_cols=metric_cols)
    anomalies = profile_anomalies(df, date_col=date_col, metric_cols=metric_cols)

    temporal = None
    if date_col:
        temporal = profile_temporal_patterns(df, date_col, metric_cols, freq)

    # Summary
    critical_cols = [c for c in completeness if c["status"] == "CRITICAL"]
    high_corr = [c for c in correlations if c["abs_correlation"] >= 0.9]

    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "critical_completeness_cols": len(critical_cols),
        "high_correlation_pairs": len(high_corr),
        "total_anomalies": anomalies["total_anomalies"],
    }
    if temporal:
        summary["coverage_pct"] = temporal.get("coverage_pct", 0)
        summary["trend"] = temporal.get("trend", "unknown")
        summary["seasonality"] = temporal.get("seasonality_detected", False)

    return {
        "distributions": distributions,
        "temporal": temporal,
        "correlations": correlations,
        "completeness": completeness,
        "anomalies": anomalies,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _skewness(arr: np.ndarray) -> float:
    """Compute sample skewness."""
    n = len(arr)
    if n < 3:
        return 0.0
    m = arr.mean()
    s = arr.std(ddof=1)
    if s == 0:
        return 0.0
    return float((n / ((n - 1) * (n - 2))) * np.sum(((arr - m) / s) ** 3))


def _kurtosis(arr: np.ndarray) -> float:
    """Compute excess kurtosis."""
    n = len(arr)
    if n < 4:
        return 0.0
    m = arr.mean()
    s = arr.std(ddof=1)
    if s == 0:
        return 0.0
    k = np.mean(((arr - m) / s) ** 4)
    return float(k - 3.0)


def _classify_shape(
    arr: np.ndarray, skewness: float, kurtosis: float, n_unique: int,
) -> str:
    """Classify distribution shape."""
    if n_unique <= 2:
        return "binary"
    if n_unique <= 5:
        return "categorical"
    if abs(skewness) < 0.5 and abs(kurtosis) < 1.0:
        return "normal"
    if skewness > 1.0:
        return "right-skewed"
    if skewness < -1.0:
        return "left-skewed"
    if kurtosis > 3.0:
        return "heavy-tailed"
    if abs(skewness) >= 0.5:
        return "moderately-skewed"
    return "other"


def _recommend_transform(arr: np.ndarray, skewness: float) -> Optional[str]:
    """Recommend a variance-stabilizing transform."""
    if abs(skewness) < 1.0:
        return None
    if np.min(arr) >= 0:
        if skewness > 1.0:
            return "log" if np.min(arr) > 0 else "sqrt"
    return None


def _detect_trend(values: np.ndarray) -> str:
    """Detect trend via linear fit."""
    n = len(values)
    x = np.arange(n, dtype=float)
    try:
        coeffs = np.polyfit(x, values, 1)
        slope = coeffs[0]
        mean_v = np.mean(values)
        if mean_v == 0:
            return "stable"
        rel_slope = abs(slope * n) / abs(mean_v)
        if rel_slope < 0.05:
            return "stable"
        return "increasing" if slope > 0 else "decreasing"
    except Exception:
        return "unknown"
