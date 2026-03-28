"""
User-friendly error translation layer for the Forecasting Platform.

Translates cryptic Python/Polars exceptions into plain-English messages
with actionable suggestions.  Integrates with ``StructuredLogger`` from
``src.observability`` to log every translated error.

Usage::

    from src.errors import friendly_error, safe_execute, suggest_column

    try:
        df = pl.read_csv("missing.csv")
    except Exception as exc:
        result = friendly_error(exc, context="loading data")
        print(result["message"])
        print(result["suggestion"])

    # Wrap any callable with automatic error translation
    result, info = safe_execute(pl.read_csv, "data.csv", context="ingestion")
"""

from __future__ import annotations

import difflib
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import polars as pl
    _PL_AVAILABLE = True
except ImportError:
    _PL_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error pattern definitions
# ---------------------------------------------------------------------------

_POLARS_PATTERNS = [
    {
        "keywords": ["ColumnNotFoundError", "not found: ", "column", "not found"],
        "error_type": "missing_column",
        "message": "A column name in your operation was not recognized.",
        "suggestion": (
            "Check column names for typos.  Polars column names are case-sensitive.\n"
            "  df.columns  — list available columns\n"
            "  df.schema   — show column names and types"
        ),
    },
    {
        "keywords": ["SchemaError", "schema", "type mismatch", "expected .* got"],
        "error_type": "schema_error",
        "message": "There is a data type mismatch in your operation.",
        "suggestion": (
            "A column has a different type than expected.  Common fixes:\n"
            "  df.with_columns(pl.col('x').cast(pl.Float64))  — cast types\n"
            "  df.with_columns(pl.col('date').str.to_date())   — parse dates"
        ),
    },
    {
        "keywords": ["ComputeError", "overflow", "divide by zero", "cannot compute"],
        "error_type": "compute_error",
        "message": "A computation failed (possible overflow, divide-by-zero, or type issue).",
        "suggestion": (
            "Check for:\n"
            "  - Division by zero: use pl.when(denom != 0).then(...)\n"
            "  - Null values in aggregations: .drop_nulls() or .fill_null(0)\n"
            "  - Integer overflow: cast to Float64 before large computations"
        ),
    },
    {
        "keywords": ["ShapeError", "shape", "length mismatch", "lengths don't match"],
        "error_type": "shape_error",
        "message": "DataFrames or Series have incompatible shapes.",
        "suggestion": (
            "Ensure DataFrames have the same number of rows when combining:\n"
            "  df.height  — check row count\n"
            "  Use .join() instead of .hstack() to align by key"
        ),
    },
    {
        "keywords": ["NoDataError", "empty", "no data"],
        "error_type": "empty_data",
        "message": "The data source appears to be empty.",
        "suggestion": (
            "The file exists but contains no data.  Check that:\n"
            "  - The file was fully downloaded\n"
            "  - The file is not corrupted\n"
            "  - You are pointing to the correct file path"
        ),
    },
]

_FORECAST_PATTERNS = [
    {
        "keywords": ["horizon", "invalid horizon", "forecast horizon"],
        "error_type": "invalid_horizon",
        "message": "The forecast horizon is invalid or missing.",
        "suggestion": (
            "Ensure the horizon is a positive integer:\n"
            "  config['horizon'] = 12  — number of periods to forecast\n"
            "  Check that it does not exceed your available history."
        ),
    },
    {
        "keywords": ["target column", "target_col", "missing target"],
        "error_type": "missing_target",
        "message": "The target column for forecasting was not found.",
        "suggestion": (
            "Verify your config specifies the correct target column:\n"
            "  config['target_col'] = 'quantity'\n"
            "  df.columns  — see available columns"
        ),
    },
    {
        "keywords": ["frequency", "freq", "irregular", "infer_freq"],
        "error_type": "frequency_error",
        "message": "The time series frequency could not be determined or is irregular.",
        "suggestion": (
            "Ensure your date column has a regular frequency:\n"
            "  - Resample or aggregate to a consistent frequency\n"
            "  - Fill gaps with pl.DataFrame.upsample()\n"
            "  - Specify frequency explicitly: config['freq'] = '1w'"
        ),
    },
    {
        "keywords": ["insufficient history", "not enough data", "too few observations"],
        "error_type": "insufficient_history",
        "message": "Not enough historical data for the requested model or horizon.",
        "suggestion": (
            "Most models need at least 2× the horizon in history.  Options:\n"
            "  - Reduce the horizon\n"
            "  - Use a simpler model (naive, moving average)\n"
            "  - Aggregate to a coarser frequency (daily → weekly)"
        ),
    },
]

_IO_PATTERNS = [
    {
        "keywords": ["FileNotFoundError", "No such file", "not found"],
        "error_type": "file_not_found",
        "message": "The requested file was not found.",
        "suggestion": (
            "Check the file path for typos.  Common locations:\n"
            "  data/   — raw data files\n"
            "  working/ — intermediate outputs\n"
            "  outputs/ — final deliverables"
        ),
    },
    {
        "keywords": ["PermissionError", "Permission denied", "locked"],
        "error_type": "permission_error",
        "message": "Permission denied when accessing this file.",
        "suggestion": (
            "The file may be locked by another process.\n"
            "  - Close any programs that may have the file open\n"
            "  - Check file permissions"
        ),
    },
    {
        "keywords": ["unsupported file", "cannot read", "unsupported format"],
        "error_type": "unsupported_format",
        "message": "This file format is not supported.",
        "suggestion": (
            "Supported formats: CSV, Parquet, Excel (.xlsx).  Convert other formats to CSV first."
        ),
    },
]


# ---------------------------------------------------------------------------
# Core translator
# ---------------------------------------------------------------------------

def friendly_error(
    exception: BaseException,
    context: Optional[str] = None,
    run_id: Optional[str] = None,
) -> dict:
    """Translate an exception into a user-friendly error message.

    Args:
        exception: Any Python exception.
        context: Description of what was happening (e.g. 'loading data').
        run_id: Pipeline run ID for correlation.

    Returns:
        dict with error_type, message, suggestion, technical, run_id.
    """
    exc_type = type(exception).__name__
    exc_msg = str(exception)
    tb = traceback.format_exception(type(exception), exception, exception.__traceback__)
    technical = "".join(tb)
    prefix = f"While {context}: " if context else ""

    # Check isinstance first for common types
    if isinstance(exception, FileNotFoundError):
        result = _build_result(
            "file_not_found",
            f"{prefix}File not found: {exc_msg}",
            "Check the file path.  Use pathlib.Path to verify existence.",
            technical, run_id,
        )
        _log_error(result)
        return result

    if isinstance(exception, PermissionError):
        result = _build_result(
            "permission_error",
            f"{prefix}Permission denied: {exc_msg}",
            "Close other programs using the file and check permissions.",
            technical, run_id,
        )
        _log_error(result)
        return result

    if isinstance(exception, (ImportError, ModuleNotFoundError)):
        module = _extract_module_name(exc_msg)
        pip_cmd = f"pip install {module}" if module else "pip install <package>"
        result = _build_result(
            "import_error",
            f"{prefix}Missing package: {exc_msg}",
            f"Install the missing package:\n  {pip_cmd}",
            technical, run_id,
        )
        _log_error(result)
        return result

    # Pattern matching against all pattern lists
    combined = exc_type + " " + exc_msg
    for patterns in (_POLARS_PATTERNS, _FORECAST_PATTERNS, _IO_PATTERNS):
        for pat in patterns:
            if any(kw.lower() in combined.lower() for kw in pat["keywords"]):
                result = _build_result(
                    pat["error_type"],
                    f"{prefix}{pat['message']}",
                    pat["suggestion"],
                    technical, run_id,
                )
                _log_error(result)
                return result

    # Generic fallback
    result = _build_result(
        "unknown_error",
        f"{prefix}Unexpected error: {exc_type}: {exc_msg}",
        "Read the error above for clues.  Check inputs and try again.",
        technical, run_id,
    )
    _log_error(result)
    return result


# ---------------------------------------------------------------------------
# Safe execution wrapper
# ---------------------------------------------------------------------------

def safe_execute(
    func: Callable,
    *args: Any,
    fallback: Any = None,
    context: Optional[str] = None,
    run_id: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Any, dict]:
    """Execute a callable with friendly error handling.

    Args:
        func: Callable to execute.
        *args: Positional arguments for func.
        fallback: Value returned if func raises.
        context: Description for error message.
        run_id: Pipeline run ID.
        **kwargs: Keyword arguments for func.

    Returns:
        (result, info) where info has source, status, error (or None).
    """
    try:
        result = func(*args, **kwargs)
        info = {"source": func.__name__, "status": "ok", "error": None}

        # Warn on empty DataFrame results
        if _PL_AVAILABLE and isinstance(result, pl.DataFrame) and result.is_empty():
            info["warning"] = "Operation returned an empty DataFrame."
        return (result, info)

    except Exception as exc:
        error_info = friendly_error(exc, context=context, run_id=run_id)
        return (
            fallback,
            {"source": func.__name__, "status": "error", "error": error_info},
        )


# ---------------------------------------------------------------------------
# Empty DataFrame checker
# ---------------------------------------------------------------------------

def check_empty_dataframe(df, label: str = "result") -> dict:
    """Check if a DataFrame is empty and return a structured warning.

    Args:
        df: Polars (or pandas) DataFrame.
        label: Human-readable label.

    Returns:
        dict with status ('PASS' or 'WARN'), message, details.
    """
    row_count = len(df) if hasattr(df, "__len__") else 0
    columns = list(df.columns) if hasattr(df, "columns") else []

    if row_count == 0:
        return {
            "status": "WARN",
            "message": (
                f"'{label}' returned 0 rows.  Data may be filtered too "
                "aggressively, or the source may be empty."
            ),
            "details": {"label": label, "row_count": 0, "columns": columns},
        }

    return {
        "status": "PASS",
        "message": f"'{label}' returned {row_count:,} rows.",
        "details": {"label": label, "row_count": row_count, "columns": columns},
    }


# ---------------------------------------------------------------------------
# Column suggestion
# ---------------------------------------------------------------------------

def suggest_column(
    target: str,
    available_columns: List[str],
    n: int = 3,
) -> dict:
    """Suggest closest matching column names for a misspelled column.

    Args:
        target: Column name that was not found.
        available_columns: Valid column names.
        n: Max suggestions.

    Returns:
        dict with target, suggestions, best_match.
    """
    matches = difflib.get_close_matches(target, available_columns, n=n, cutoff=0.4)
    return {
        "target": target,
        "suggestions": matches,
        "best_match": matches[0] if matches else None,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_result(
    error_type: str, message: str, suggestion: str,
    technical: str, run_id: Optional[str],
) -> dict:
    return {
        "error_type": error_type,
        "message": message,
        "suggestion": suggestion,
        "technical": technical,
        "run_id": run_id,
    }


def _log_error(result: dict) -> None:
    """Log translated error at WARNING level."""
    logger.warning(
        "Error translated: type=%s run_id=%s msg=%s",
        result["error_type"],
        result.get("run_id", ""),
        result["message"][:200],
    )


def _extract_module_name(msg: str) -> Optional[str]:
    """Extract module name from ImportError message."""
    for prefix in ("No module named ", "cannot import name "):
        if prefix in msg:
            name = msg.split(prefix, 1)[1].strip().strip("'\"")
            return name.split(".")[0]
    return None
