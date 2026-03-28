"""
Layer 4 — Simpson's Paradox Detection for forecasting data.

Detects directional reversals between segment-level and aggregate trends.
Critical for hierarchical forecasting — e.g. a metric improving at the
aggregate level while deteriorating for every individual segment.

All functions accept Polars DataFrames.

Usage:
    from src.validation.simpsons_paradox import (
        check_simpsons_paradox, check_simpsons_multi_segment,
        generate_paradox_report,
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import polars as pl

# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def check_simpsons_paradox(
    df: pl.DataFrame,
    segment_col: str,
    value_col: str,
    weight_col: Optional[str] = None,
    period_col: Optional[str] = None,
    period_a=None,
    period_b=None,
) -> dict:
    """
    Check for Simpson's Paradox across a single segment dimension.

    Compares the direction of change between two periods at the aggregate
    level vs each segment.  A paradox exists when the aggregate direction
    differs from the direction observed in the *majority* of segments.

    Args:
        df: Polars DataFrame.
        segment_col: Column defining segments (e.g. "region", "product").
        value_col: Metric column.
        weight_col: Optional weight column for weighted averages.
        period_col: Column identifying time periods.
        period_a: Earlier period value.
        period_b: Later period value.

    Returns:
        dict with ``paradox_detected`` (bool), ``aggregate_direction``,
        ``segment_directions``, ``reversal_segments``, ``severity``.
    """
    if period_col and period_a is not None and period_b is not None:
        df_a = df.filter(pl.col(period_col) == period_a)
        df_b = df.filter(pl.col(period_col) == period_b)
    elif period_col:
        # Auto-detect two most recent periods
        periods = df[period_col].unique().sort()
        if periods.len() < 2:
            return _no_paradox("Insufficient periods")
        period_a = periods[-2]
        period_b = periods[-1]
        df_a = df.filter(pl.col(period_col) == period_a)
        df_b = df.filter(pl.col(period_col) == period_b)
    else:
        # Split in half if no period column
        mid = df.height // 2
        df_a = df.head(mid)
        df_b = df.tail(df.height - mid)

    # Aggregate direction
    agg_a = _weighted_mean(df_a, value_col, weight_col)
    agg_b = _weighted_mean(df_b, value_col, weight_col)

    if agg_a is None or agg_b is None:
        return _no_paradox("Cannot compute aggregate means")

    agg_dir = _direction(agg_a, agg_b)

    # Segment directions
    segment_dirs: list[dict] = []
    reversals: list[str] = []

    segments = df[segment_col].unique().to_list()
    for seg in segments:
        seg_a = df_a.filter(pl.col(segment_col) == seg)
        seg_b = df_b.filter(pl.col(segment_col) == seg)

        mean_a = _weighted_mean(seg_a, value_col, weight_col)
        mean_b = _weighted_mean(seg_b, value_col, weight_col)

        if mean_a is None or mean_b is None:
            continue

        seg_dir = _direction(mean_a, mean_b)
        segment_dirs.append({
            "segment": seg,
            "mean_a": round(mean_a, 4),
            "mean_b": round(mean_b, 4),
            "direction": seg_dir,
        })

        if seg_dir != agg_dir and seg_dir != "flat":
            reversals.append(seg)

    # Paradox if majority of segments reverse
    n_segments = len(segment_dirs)
    paradox = len(reversals) > n_segments / 2 if n_segments > 0 else False

    severity = "BLOCKER" if paradox else "PASS"

    return {
        "paradox_detected": paradox,
        "aggregate_direction": agg_dir,
        "aggregate_a": round(agg_a, 4),
        "aggregate_b": round(agg_b, 4),
        "segment_directions": segment_dirs,
        "reversal_segments": reversals,
        "reversal_rate": round(len(reversals) / n_segments, 2) if n_segments else 0,
        "severity": severity,
    }


def check_simpsons_multi_segment(
    df: pl.DataFrame,
    segment_columns: Sequence[str],
    value_col: str,
    weight_col: Optional[str] = None,
    period_col: Optional[str] = None,
    period_a=None,
    period_b=None,
) -> dict:
    """Run paradox check across multiple segment dimensions.

    Args:
        df: Polars DataFrame.
        segment_columns: List of segment columns to check.
        value_col: Metric column.
        weight_col: Weight column.
        period_col: Period column.
        period_a, period_b: Period values.

    Returns:
        dict with ``any_paradox``, ``results`` (per-segment-column).
    """
    results: Dict[str, dict] = {}
    any_paradox = False

    for seg_col in segment_columns:
        if seg_col not in df.columns:
            continue
        result = check_simpsons_paradox(
            df, seg_col, value_col,
            weight_col=weight_col,
            period_col=period_col,
            period_a=period_a,
            period_b=period_b,
        )
        results[seg_col] = result
        if result["paradox_detected"]:
            any_paradox = True

    return {
        "any_paradox": any_paradox,
        "results": results,
        "severity": "BLOCKER" if any_paradox else "PASS",
    }


def weighted_vs_unweighted(
    df: pl.DataFrame,
    value_col: str,
    weight_col: str,
    segment_col: Optional[str] = None,
) -> dict:
    """Compare weighted vs unweighted means to detect mix-shift effects.

    Args:
        df: Polars DataFrame.
        value_col: Metric column.
        weight_col: Weight column.
        segment_col: Optional segment column for per-segment comparison.

    Returns:
        dict with ``weighted_mean``, ``unweighted_mean``, ``gap``,
        ``gap_pct``, ``segment_detail``.
    """
    w_mean = _weighted_mean(df, value_col, weight_col)
    u_mean = df[value_col].mean()

    if w_mean is None or u_mean is None:
        return {"weighted_mean": None, "unweighted_mean": None,
                "gap": None, "gap_pct": None, "segment_detail": []}

    gap = w_mean - u_mean
    gap_pct = (gap / u_mean * 100) if u_mean != 0 else 0.0

    segment_detail = []
    if segment_col and segment_col in df.columns:
        for seg in df[segment_col].unique().to_list():
            sub = df.filter(pl.col(segment_col) == seg)
            sw = _weighted_mean(sub, value_col, weight_col)
            su = sub[value_col].mean()
            segment_detail.append({
                "segment": seg,
                "weighted_mean": round(sw, 4) if sw else None,
                "unweighted_mean": round(su, 4) if su else None,
            })

    return {
        "weighted_mean": round(w_mean, 4),
        "unweighted_mean": round(u_mean, 4),
        "gap": round(gap, 4),
        "gap_pct": round(gap_pct, 2),
        "segment_detail": segment_detail,
    }


def suggest_segments_to_check(
    df: pl.DataFrame,
    value_col: str,
    max_segments: int = 5,
) -> list[str]:
    """Suggest columns worth checking for Simpson's Paradox.

    Heuristic: categorical or low-cardinality columns (≤50 unique) that
    are likely segment dimensions.

    Args:
        df: Polars DataFrame.
        value_col: The metric column (excluded from suggestions).
        max_segments: Maximum suggestions.

    Returns:
        List of column names.
    """
    candidates = []
    for col in df.columns:
        if col == value_col:
            continue
        dtype = df[col].dtype
        if dtype in (pl.Utf8, pl.Categorical):
            candidates.append((col, df[col].n_unique()))
        elif dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                       pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            uniq = df[col].n_unique()
            if uniq <= 50:
                candidates.append((col, uniq))

    # Sort by cardinality ascending (smaller = more interesting)
    candidates.sort(key=lambda x: x[1])
    return [c[0] for c in candidates[:max_segments]]


def generate_paradox_report(
    df: pl.DataFrame,
    segment_columns: Sequence[str],
    value_col: str,
    weight_col: Optional[str] = None,
    period_col: Optional[str] = None,
    period_a=None,
    period_b=None,
) -> dict:
    """Generate a full paradox report with recommendations.

    Args:
        df: Polars DataFrame.
        segment_columns: Segment columns to check.
        value_col: Metric column.
        weight_col: Weight column.
        period_col: Period column.
        period_a, period_b: Period values.

    Returns:
        dict with ``summary``, ``multi_segment_results``,
        ``recommendations``.
    """
    multi = check_simpsons_multi_segment(
        df, segment_columns, value_col,
        weight_col=weight_col,
        period_col=period_col,
        period_a=period_a,
        period_b=period_b,
    )

    affected = [
        col for col, r in multi["results"].items()
        if r["paradox_detected"]
    ]

    recommendations = []
    if affected:
        recommendations.append(
            "DO NOT rely on aggregate-level conclusions. "
            "Break down by the following dimensions: "
            + ", ".join(affected)
        )
        recommendations.append(
            "Check for mix-shift effects — the composition of segments "
            "may have changed between periods."
        )
        if weight_col:
            recommendations.append(
                "Compare weighted vs unweighted means to quantify the "
                "mix-shift impact."
            )
    else:
        recommendations.append(
            "No paradox detected. Aggregate conclusions appear sound."
        )

    return {
        "summary": {
            "paradox_found": multi["any_paradox"],
            "affected_dimensions": affected,
            "dimensions_checked": list(multi["results"].keys()),
        },
        "multi_segment_results": multi,
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _weighted_mean(
    df: pl.DataFrame, value_col: str, weight_col: Optional[str],
) -> Optional[float]:
    """Compute (optionally weighted) mean."""
    if df.is_empty() or value_col not in df.columns:
        return None
    if weight_col and weight_col in df.columns:
        valid = df.filter(
            pl.col(value_col).is_not_null() & pl.col(weight_col).is_not_null()
        )
        if valid.is_empty():
            return None
        total_w = valid[weight_col].sum()
        if total_w == 0:
            return None
        return float(
            (valid[value_col].cast(pl.Float64) * valid[weight_col].cast(pl.Float64)).sum()
            / total_w
        )
    return float(df[value_col].mean())


def _direction(a: float, b: float) -> str:
    """'up', 'down', or 'flat'."""
    if b > a * 1.001:
        return "up"
    elif b < a * 0.999:
        return "down"
    return "flat"


def _no_paradox(message: str) -> dict:
    return {
        "paradox_detected": False,
        "aggregate_direction": None,
        "aggregate_a": None,
        "aggregate_b": None,
        "segment_directions": [],
        "reversal_segments": [],
        "reversal_rate": 0,
        "severity": "PASS",
        "message": message,
    }
