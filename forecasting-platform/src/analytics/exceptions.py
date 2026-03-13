"""
ExceptionEngine — automated exception flagging for S&OP review queues.

Planners reviewing hundreds of SKUs each week need to know which ones
require attention.  The ExceptionEngine applies configurable business
rules to the ForecastComparator output and produces a boolean flag
column per exception type, plus a summary ``exception_flags`` list
column containing the names of all fired flags.

Exception types
---------------
LARGE_CYCLE_CHANGE    |cycle_change_pct| > threshold (default 20%)
HIGH_UNCERTAINTY      uncertainty_ratio > threshold  (default 0.5)
FIELD_DISAGREE        |<name>_gap_pct| > threshold   (default 25%) for any external
OVERFCAST / UNDERFORECAST  signed cycle bias exceeding separate thresholds
NO_PRIOR              prior_model_forecast is null (new series or first cycle)

Usage
-----
>>> eng = ExceptionEngine()
>>> flagged = eng.flag(comparison_df)
>>> # Filter to actionable exceptions only:
>>> actionable = flagged.filter(pl.col("has_exception"))
"""

from typing import Dict, List, Optional

import polars as pl


# Default thresholds (can all be overridden at construction or call time)
_DEFAULTS = dict(
    cycle_change_pct_threshold=20.0,    # %
    uncertainty_ratio_threshold=0.50,   # (P90-P10)/P50
    field_disagree_pct_threshold=25.0,  # %
    overforecast_pct_threshold=30.0,    # % above external
    underforecast_pct_threshold=-30.0,  # % below external
)


class ExceptionEngine:
    """
    Apply business-rule exception flags to a ForecastComparator output.

    Parameters
    ----------
    cycle_change_pct_threshold:
        Absolute % cycle-over-cycle change that triggers LARGE_CYCLE_CHANGE.
    uncertainty_ratio_threshold:
        (P90-P10)/P50 above which HIGH_UNCERTAINTY fires.
    field_disagree_pct_threshold:
        Absolute % gap vs any external forecast that triggers FIELD_DISAGREE.
    overforecast_pct_threshold:
        Positive % gap vs any external forecast that triggers OVERFORECAST.
    underforecast_pct_threshold:
        Negative % gap (value is negative) vs any external forecast
        that triggers UNDERFORECAST.
    """

    def __init__(
        self,
        cycle_change_pct_threshold: float = _DEFAULTS["cycle_change_pct_threshold"],
        uncertainty_ratio_threshold: float = _DEFAULTS["uncertainty_ratio_threshold"],
        field_disagree_pct_threshold: float = _DEFAULTS["field_disagree_pct_threshold"],
        overforecast_pct_threshold: float = _DEFAULTS["overforecast_pct_threshold"],
        underforecast_pct_threshold: float = _DEFAULTS["underforecast_pct_threshold"],
    ):
        self.cycle_change_pct_threshold = cycle_change_pct_threshold
        self.uncertainty_ratio_threshold = uncertainty_ratio_threshold
        self.field_disagree_pct_threshold = field_disagree_pct_threshold
        self.overforecast_pct_threshold = overforecast_pct_threshold
        self.underforecast_pct_threshold = underforecast_pct_threshold

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def flag(
        self,
        comparison: pl.DataFrame,
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """
        Add exception flag columns to a ForecastComparator output.

        Returns the input DataFrame with additional columns:

          exc_large_cycle_change   — bool
          exc_high_uncertainty     — bool
          exc_field_disagree       — bool
          exc_overforecast         — bool
          exc_underforecast        — bool
          exc_no_prior             — bool (prior_model_forecast is null)
          exception_flags          — list[str] of fired flag names
          has_exception            — bool (any flag fired)
        """
        result = comparison.clone()
        fired: List[str] = []

        # ── LARGE_CYCLE_CHANGE ────────────────────────────────────────────
        if "cycle_change_pct" in result.columns:
            result = result.with_columns(
                (pl.col("cycle_change_pct").abs() > self.cycle_change_pct_threshold)
                .alias("exc_large_cycle_change")
            )
            fired.append("exc_large_cycle_change")
        else:
            result = result.with_columns(pl.lit(False).alias("exc_large_cycle_change"))

        # ── HIGH_UNCERTAINTY ──────────────────────────────────────────────
        if "uncertainty_ratio" in result.columns:
            result = result.with_columns(
                (pl.col("uncertainty_ratio") > self.uncertainty_ratio_threshold)
                .fill_null(False)
                .alias("exc_high_uncertainty")
            )
            fired.append("exc_high_uncertainty")
        else:
            result = result.with_columns(pl.lit(False).alias("exc_high_uncertainty"))

        # ── FIELD_DISAGREE / OVERFORECAST / UNDERFORECAST ─────────────────
        gap_pct_cols = [c for c in result.columns if c.endswith("_gap_pct")]

        if gap_pct_cols:
            # Absolute disagreement with ANY external source
            disagree_expr = pl.lit(False)
            over_expr = pl.lit(False)
            under_expr = pl.lit(False)
            for col in gap_pct_cols:
                disagree_expr = disagree_expr | (
                    pl.col(col).abs() > self.field_disagree_pct_threshold
                )
                over_expr = over_expr | (
                    pl.col(col) > self.overforecast_pct_threshold
                )
                under_expr = under_expr | (
                    pl.col(col) < self.underforecast_pct_threshold
                )
            result = result.with_columns([
                disagree_expr.fill_null(False).alias("exc_field_disagree"),
                over_expr.fill_null(False).alias("exc_overforecast"),
                under_expr.fill_null(False).alias("exc_underforecast"),
            ])
            fired += ["exc_field_disagree", "exc_overforecast", "exc_underforecast"]
        else:
            result = result.with_columns([
                pl.lit(False).alias("exc_field_disagree"),
                pl.lit(False).alias("exc_overforecast"),
                pl.lit(False).alias("exc_underforecast"),
            ])

        # ── NO_PRIOR ──────────────────────────────────────────────────────
        if "prior_model_forecast" in result.columns:
            result = result.with_columns(
                pl.col("prior_model_forecast").is_null().alias("exc_no_prior")
            )
        else:
            result = result.with_columns(pl.lit(False).alias("exc_no_prior"))

        # ── Summary columns ───────────────────────────────────────────────
        flag_cols = [
            "exc_large_cycle_change", "exc_high_uncertainty",
            "exc_field_disagree", "exc_overforecast",
            "exc_underforecast", "exc_no_prior",
        ]

        # has_exception: any flag is True
        any_flag_expr = pl.lit(False)
        for fc in flag_cols:
            any_flag_expr = any_flag_expr | pl.col(fc)

        result = result.with_columns(
            any_flag_expr.alias("has_exception")
        )

        return result.sort([id_col, time_col])

    def exception_summary(
        self,
        flagged: pl.DataFrame,
        id_col: str = "series_id",
    ) -> pl.DataFrame:
        """
        Aggregate flagged DataFrame to one row per series showing:
          - which exception types fired (ever across the forecast horizon)
          - count of exception weeks per type
          - total exception weeks

        Useful for the S&OP review queue — sorted by total exception weeks
        descending so the most-flagged series appear first.
        """
        if flagged.is_empty():
            return flagged

        flag_cols = [c for c in flagged.columns if c.startswith("exc_")]
        if not flag_cols:
            return flagged.select([id_col]).unique()

        agg_exprs = [
            pl.col(fc).sum().alias(f"n_weeks_{fc}") for fc in flag_cols
        ]
        agg_exprs += [
            pl.col("has_exception").sum().alias("total_exception_weeks")
        ]

        return (
            flagged.group_by(id_col)
            .agg(agg_exprs)
            .sort("total_exception_weeks", descending=True)
        )
