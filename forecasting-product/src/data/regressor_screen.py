"""
Regressor variance screening — pre-training quality checks for external features.

Runs after DataValidator and feature join but before model fit.  Checks:
  1. Near-zero variance (constant or near-constant columns).
  2. Highly correlated feature pairs (redundant information).
  3. (Optional) Mutual information with target (no predictive signal).

Produces a ``RegressorScreenReport`` summarising dropped features, warnings,
and per-column statistics.
"""

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional

import polars as pl

from ..config.schema import RegressorScreenConfig

logger = logging.getLogger(__name__)

_HAS_SKLEARN = False
try:
    from sklearn.feature_selection import mutual_info_regression
    _HAS_SKLEARN = True
except ImportError:
    pass


@dataclass
class RegressorScreenReport:
    """Result of pre-training regressor screening."""

    screened_columns: List[str] = field(default_factory=list)
    dropped_columns: List[str] = field(default_factory=list)
    low_variance_columns: List[str] = field(default_factory=list)
    high_correlation_pairs: List[Dict[str, Any]] = field(default_factory=list)
    low_mi_columns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    per_column_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def screen_regressors(
    df: pl.DataFrame,
    feature_columns: List[str],
    target_col: str = "quantity",
    config: Optional[RegressorScreenConfig] = None,
) -> RegressorScreenReport:
    """
    Screen external regressors for quality before model training.

    Parameters
    ----------
    df:
        Model-ready DataFrame containing target and feature columns.
    feature_columns:
        List of feature column names to screen.
    target_col:
        Name of the target column (for mutual information check).
    config:
        Screening thresholds.  Uses defaults if *None*.

    Returns
    -------
    RegressorScreenReport with screening results and per-column statistics.
    """
    if config is None:
        config = RegressorScreenConfig(enabled=True)

    report = RegressorScreenReport(screened_columns=list(feature_columns))

    # Filter to columns actually present in the DataFrame
    present_cols = [c for c in feature_columns if c in df.columns]
    if not present_cols:
        return report

    # --- 1. Variance check ---------------------------------------------------
    _check_variance(df, present_cols, config, report)

    # --- 2. Pairwise correlation ----------------------------------------------
    # Only check columns that survived variance screening
    surviving = [c for c in present_cols if c not in report.dropped_columns]
    if len(surviving) >= 2:
        _check_correlation(df, surviving, config, report)

    # --- 3. Mutual information (optional) -------------------------------------
    if config.mi_enabled:
        surviving = [c for c in present_cols if c not in report.dropped_columns]
        if surviving and target_col in df.columns:
            _check_mutual_information(df, surviving, target_col, config, report)

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_variance(
    df: pl.DataFrame,
    columns: List[str],
    config: RegressorScreenConfig,
    report: RegressorScreenReport,
) -> None:
    """Flag and optionally drop near-zero-variance columns."""
    for col in columns:
        var = df[col].cast(pl.Float64).var()
        if var is None:
            var = 0.0
        report.per_column_stats.setdefault(col, {})["variance"] = float(var)

        if var < config.variance_threshold:
            report.low_variance_columns.append(col)
            report.dropped_columns.append(col)
            msg = (
                f"dropped {col!r} (variance={var:.2e} "
                f"< threshold={config.variance_threshold:.2e})"
            )
            report.warnings.append(msg)
            logger.info("Regressor screen: %s", msg)


def _check_correlation(
    df: pl.DataFrame,
    columns: List[str],
    config: RegressorScreenConfig,
    report: RegressorScreenReport,
) -> None:
    """Warn on highly correlated feature pairs."""
    for col_a, col_b in combinations(columns, 2):
        corr = df.select(
            pl.corr(col_a, col_b, method="pearson")
        ).item()
        if corr is None:
            continue
        abs_corr = abs(float(corr))
        if abs_corr >= config.correlation_threshold:
            report.high_correlation_pairs.append({
                "col_a": col_a,
                "col_b": col_b,
                "correlation": round(float(corr), 4),
            })
            msg = (
                f"warning: {col_a!r} and {col_b!r} are "
                f"{abs_corr:.0%} correlated — consider dropping one"
            )
            report.warnings.append(msg)
            logger.info("Regressor screen: %s", msg)


def _check_mutual_information(
    df: pl.DataFrame,
    columns: List[str],
    target_col: str,
    config: RegressorScreenConfig,
    report: RegressorScreenReport,
) -> None:
    """Flag features with near-zero mutual information with the target."""
    if not _HAS_SKLEARN:
        msg = "mutual information check skipped — sklearn not installed"
        report.warnings.append(msg)
        logger.warning("Regressor screen: %s", msg)
        return

    # Build numpy arrays (drop nulls)
    sub = df.select([target_col] + columns).drop_nulls()
    if sub.is_empty() or len(sub) < 10:
        return

    y = sub[target_col].to_numpy().astype(float)
    X = sub.select(columns).to_numpy().astype(float)  # noqa: N806

    mi_scores = mutual_info_regression(X, y, random_state=42)

    for col, mi in zip(columns, mi_scores):
        report.per_column_stats.setdefault(col, {})["mi_score"] = float(mi)
        if mi < config.mi_threshold:
            report.low_mi_columns.append(col)
            msg = (
                f"warning: {col!r} has near-zero mutual information "
                f"with target (MI={mi:.4f} < {config.mi_threshold})"
            )
            report.warnings.append(msg)
            logger.info("Regressor screen: %s", msg)
