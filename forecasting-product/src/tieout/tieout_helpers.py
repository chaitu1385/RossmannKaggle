"""
Dual-path data integrity verification helpers.

All profiling uses Polars DataFrames (the platform standard).
Comparisons work between any two profiles, regardless of how the
data was loaded (CSV, Parquet, DuckDB, etc.).

Tolerances:
    - Row/distinct counts: exact (0)
    - Numeric sums: 0.01% relative
    - Claim-level spot checks: 0.1% relative
    - Absolute floor: 0.01 (for near-zero values)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

_ROW_TOL = 0           # exact match
_NUMERIC_TOL = 0.0001  # 0.01% relative
_CLAIM_TOL = 0.001     # 0.1% relative
_ABS_FLOOR = 0.01      # absolute floor for near-zero


# ---------------------------------------------------------------------------
# Source reading
# ---------------------------------------------------------------------------

def read_source_direct(
    path: str,
    **kwargs,
) -> pl.DataFrame:
    """Read a data file directly with Polars (independent of pipeline).

    Supports CSV, Parquet, JSON, and Excel.

    Args:
        path: File path.
        **kwargs: Passed to the appropriate Polars reader.

    Returns:
        Polars DataFrame.

    Raises:
        ValueError: If file format is not supported.
        FileNotFoundError: If file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path, **kwargs)
    elif suffix == ".parquet":
        return pl.read_parquet(path, **kwargs)
    elif suffix in (".json", ".jsonl", ".ndjson"):
        return pl.read_ndjson(path, **kwargs)
    elif suffix in (".xlsx", ".xls"):
        return pl.read_excel(path, **kwargs)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            "Supported: .csv, .parquet, .json, .jsonl, .ndjson, .xlsx, .xls"
        )


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def profile_dataframe(
    df: pl.DataFrame,
    label: str = "source",
) -> dict:
    """Compute a structural + aggregation profile of a DataFrame.

    Args:
        df: Polars DataFrame to profile.
        label: Label for the profile (e.g. "source", "duckdb").

    Returns:
        dict with keys: ``label``, ``row_count``, ``columns``,
        ``null_counts``, ``numeric_sums``, ``distinct_counts``,
        ``date_ranges``.
    """
    if df.is_empty():
        return {
            "label": label,
            "row_count": 0,
            "columns": list(df.columns),
            "null_counts": {},
            "numeric_sums": {},
            "distinct_counts": {},
            "date_ranges": {},
            "warning": "EMPTY_DATAFRAME",
        }

    null_counts = {}
    numeric_sums = {}
    distinct_counts = {}
    date_ranges = {}

    for col in df.columns:
        # Null counts
        null_counts[col] = df[col].null_count()

        # Distinct counts
        distinct_counts[col] = df[col].n_unique()

        # Type-specific profiling
        dtype = df[col].dtype
        if dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
                     pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            s = df[col].drop_nulls().sum()
            numeric_sums[col] = float(s) if s is not None else 0.0

        elif dtype in (pl.Date, pl.Datetime):
            dates = df[col].drop_nulls()
            if dates.len() > 0:
                date_ranges[col] = {
                    "min": str(dates.min()),
                    "max": str(dates.max()),
                }

    return {
        "label": label,
        "row_count": len(df),
        "columns": list(df.columns),
        "null_counts": null_counts,
        "numeric_sums": numeric_sums,
        "distinct_counts": distinct_counts,
        "date_ranges": date_ranges,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _compare_exact(
    check: str, metric: str, src_val: Any, db_val: Any,
) -> dict:
    """Compare two values for exact equality."""
    status = "PASS" if src_val == db_val else "FAIL"
    return {
        "check": check,
        "metric": metric,
        "source": src_val,
        "target": db_val,
        "status": status,
    }


def _compare_within_tolerance(
    check: str,
    metric: str,
    src_val: float,
    db_val: float,
    tol: float = _NUMERIC_TOL,
    abs_floor: float = _ABS_FLOOR,
) -> dict:
    """Compare two numeric values within a relative tolerance.

    Uses absolute floor for near-zero values.
    """
    # Handle None/NaN
    if src_val is None and db_val is None:
        return {"check": check, "metric": metric, "source": src_val,
                "target": db_val, "status": "PASS"}
    if src_val is None or db_val is None:
        return {"check": check, "metric": metric, "source": src_val,
                "target": db_val, "status": "FAIL"}

    # Both zero
    if src_val == 0 and db_val == 0:
        return {"check": check, "metric": metric, "source": src_val,
                "target": db_val, "status": "PASS", "diff": 0}

    # Absolute floor for near-zero
    if abs(src_val) < abs_floor and abs(db_val) < abs_floor:
        diff = abs(src_val - db_val)
        status = "PASS" if diff < abs_floor else "WARN"
        return {"check": check, "metric": metric, "source": src_val,
                "target": db_val, "status": status, "diff": round(diff, 6)}

    # Relative comparison
    denom = max(abs(src_val), abs(db_val))
    rel_diff = abs(src_val - db_val) / denom if denom > 0 else 0

    if rel_diff <= tol:
        status = "PASS"
    elif rel_diff <= tol * 10:
        status = "WARN"
    else:
        status = "FAIL"

    return {
        "check": check, "metric": metric,
        "source": round(src_val, 6), "target": round(db_val, 6),
        "status": status, "rel_diff": round(rel_diff, 6),
    }


def compare_profiles(
    source_profile: dict,
    target_profile: dict,
) -> list[dict]:
    """Compare two profiles and return a list of check results.

    Runs Tier 1 (structural) and Tier 2 (aggregation) comparisons.

    Args:
        source_profile: Profile from ``profile_dataframe(..., label="source")``.
        target_profile: Profile from ``profile_dataframe(..., label="target")``.

    Returns:
        List of check result dicts, each with ``check``, ``metric``,
        ``source``, ``target``, ``status`` ("PASS"/"WARN"/"FAIL").
    """
    results: list[dict] = []

    # Guard: empty data
    if source_profile.get("warning") == "EMPTY_DATAFRAME":
        results.append({
            "check": "guard", "metric": "source_empty",
            "source": 0, "target": target_profile["row_count"],
            "status": "FAIL",
        })
        return results

    if target_profile.get("warning") == "EMPTY_DATAFRAME":
        results.append({
            "check": "guard", "metric": "target_empty",
            "source": source_profile["row_count"], "target": 0,
            "status": "FAIL",
        })
        return results

    # --- Tier 1: Structural ---

    # Row count
    results.append(_compare_exact(
        "tier1", "row_count",
        source_profile["row_count"], target_profile["row_count"],
    ))

    # Column count
    results.append(_compare_exact(
        "tier1", "column_count",
        len(source_profile["columns"]), len(target_profile["columns"]),
    ))

    # Column names
    src_cols = set(source_profile["columns"])
    tgt_cols = set(target_profile["columns"])
    missing = src_cols - tgt_cols
    extra = tgt_cols - src_cols
    if missing or extra:
        results.append({
            "check": "tier1", "metric": "column_names",
            "source": sorted(missing) if missing else [],
            "target": sorted(extra) if extra else [],
            "status": "FAIL" if missing else "WARN",
            "detail": f"Missing: {sorted(missing)}, Extra: {sorted(extra)}",
        })
    else:
        results.append({
            "check": "tier1", "metric": "column_names",
            "source": "match", "target": "match", "status": "PASS",
        })

    # Null counts per column
    common_cols = src_cols & tgt_cols
    for col in sorted(common_cols):
        src_nulls = source_profile["null_counts"].get(col, 0)
        tgt_nulls = target_profile["null_counts"].get(col, 0)
        results.append(_compare_exact(
            "tier1", f"nulls.{col}", src_nulls, tgt_nulls,
        ))

    # --- Tier 2: Aggregation ---

    # Numeric sums
    src_sums = source_profile.get("numeric_sums", {})
    tgt_sums = target_profile.get("numeric_sums", {})
    for col in sorted(set(src_sums) & set(tgt_sums)):
        results.append(_compare_within_tolerance(
            "tier2", f"sum.{col}", src_sums[col], tgt_sums[col],
        ))

    # Handle asymmetric numeric columns
    for col in sorted(set(src_sums) - set(tgt_sums)):
        results.append({
            "check": "tier2", "metric": f"sum.{col}",
            "source": src_sums[col], "target": "N/A",
            "status": "WARN", "detail": "Column not numeric in target",
        })

    # Distinct counts
    src_dist = source_profile.get("distinct_counts", {})
    tgt_dist = target_profile.get("distinct_counts", {})
    for col in sorted(set(src_dist) & set(tgt_dist)):
        results.append(_compare_exact(
            "tier2", f"distinct.{col}", src_dist[col], tgt_dist[col],
        ))

    # Date ranges
    src_dates = source_profile.get("date_ranges", {})
    tgt_dates = target_profile.get("date_ranges", {})
    for col in sorted(set(src_dates) & set(tgt_dates)):
        results.append(_compare_exact(
            "tier2", f"date_min.{col}",
            src_dates[col]["min"], tgt_dates[col]["min"],
        ))
        results.append(_compare_exact(
            "tier2", f"date_max.{col}",
            src_dates[col]["max"], tgt_dates[col]["max"],
        ))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_tieout_table(results: list[dict]) -> str:
    """Format tie-out results as a Markdown table.

    Args:
        results: Output from ``compare_profiles()``.

    Returns:
        Markdown-formatted table string.
    """
    lines = [
        "| Check | Metric | Source | Target | Status |",
        "|-------|--------|--------|--------|--------|",
    ]
    for r in results:
        status_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(r["status"], "?")
        lines.append(
            f"| {r['check']} | {r['metric']} | {r['source']} | {r['target']} "
            f"| {status_icon} {r['status']} |"
        )
    return "\n".join(lines)


def overall_status(results: list[dict]) -> str:
    """Compute overall gate decision from tie-out results.

    Returns:
        ``"PASS"``, ``"WARN"``, or ``"FAIL"``.
    """
    statuses = {r["status"] for r in results}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def gate_decision(results: list[dict]) -> dict:
    """Return a structured gate decision.

    Returns:
        dict with ``status``, ``action``, ``summary``.
    """
    status = overall_status(results)
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    warned = sum(1 for r in results if r["status"] == "WARN")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    actions = {
        "PASS": "PROCEED — data integrity verified",
        "WARN": "PROCEED WITH CAUTION — review warnings before analysis",
        "FAIL": "HALT — data integrity issues must be resolved",
    }

    return {
        "status": status,
        "action": actions[status],
        "total_checks": total,
        "passed": passed,
        "warned": warned,
        "failed": failed,
        "summary": f"{passed}/{total} passed, {warned} warnings, {failed} failures",
    }


# ---------------------------------------------------------------------------
# Data quality extensions
# ---------------------------------------------------------------------------

def check_null_concentration(
    df: pl.DataFrame,
    warn_threshold: float = 0.5,
    fail_threshold: float = 0.95,
) -> list[dict]:
    """Flag columns with high null concentration.

    Args:
        df: Polars DataFrame.
        warn_threshold: Null rate above this → WARN.
        fail_threshold: Null rate above this → FAIL.

    Returns:
        List of check result dicts.
    """
    results = []
    total = len(df)
    if total == 0:
        return results

    for col in df.columns:
        null_rate = df[col].null_count() / total
        if null_rate >= fail_threshold:
            status = "FAIL"
        elif null_rate >= warn_threshold:
            status = "WARN"
        else:
            status = "PASS"

        if status != "PASS":
            results.append({
                "check": "null_concentration",
                "column": col,
                "null_rate": round(null_rate, 4),
                "null_count": df[col].null_count(),
                "status": status,
            })

    return results


def check_outliers(
    series: pl.Series,
    method: str = "iqr",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
) -> dict:
    """Detect outliers in a numeric series.

    Args:
        series: Polars numeric Series.
        method: ``"iqr"`` or ``"zscore"``.
        iqr_multiplier: IQR fence multiplier.
        z_threshold: Z-score threshold.

    Returns:
        dict with ``method``, ``outlier_count``, ``outlier_rate``,
        ``bounds``, ``status``.
    """
    vals = series.drop_nulls().cast(pl.Float64)

    if vals.len() < 3:
        return {"method": method, "outlier_count": 0, "outlier_rate": 0,
                "status": "PASS", "detail": "Too few values"}

    if method == "iqr":
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        outliers = vals.filter((vals < lower) | (vals > upper))
        bounds = {"lower": round(lower, 4), "upper": round(upper, 4)}
    else:  # zscore
        mean = vals.mean()
        std = vals.std()
        if std == 0:
            return {"method": method, "outlier_count": 0, "outlier_rate": 0,
                    "status": "PASS", "bounds": {}}
        z = ((vals - mean) / std).abs()
        outliers = vals.filter(z > z_threshold)
        bounds = {"z_threshold": z_threshold}

    outlier_count = outliers.len()
    outlier_rate = outlier_count / vals.len()

    if outlier_rate > 0.05:
        status = "FAIL"
    elif outlier_count > 0:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "method": method,
        "outlier_count": outlier_count,
        "outlier_rate": round(outlier_rate, 4),
        "bounds": bounds,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def run_full_tieout(
    source_path: str,
    target_df: pl.DataFrame,
    source_label: str = "source",
    target_label: str = "pipeline",
) -> dict:
    """Run a complete tie-out: read source, profile both, compare, gate.

    Args:
        source_path: Path to the raw source file.
        target_df: DataFrame as loaded by the pipeline.
        source_label: Label for the source profile.
        target_label: Label for the target profile.

    Returns:
        dict with ``source_profile``, ``target_profile``, ``results``,
        ``gate``, ``table`` (formatted markdown).
    """
    source_df = read_source_direct(source_path)
    source_profile = profile_dataframe(source_df, label=source_label)
    target_profile = profile_dataframe(target_df, label=target_label)

    results = compare_profiles(source_profile, target_profile)
    gate = gate_decision(results)
    table = format_tieout_table(results)

    # Also run null concentration on target
    null_checks = check_null_concentration(target_df)

    return {
        "source_profile": source_profile,
        "target_profile": target_profile,
        "results": results,
        "null_checks": null_checks,
        "gate": gate,
        "table": table,
    }
