"""
Method 4 — Temporal Co-movement
(Design document §3.4 / Phase 2)

Detects SKU transitions by measuring statistical co-movement between an old
SKU's sales and a new SKU's sales in the period surrounding the new SKU's
launch.

Theory
------
If two SKUs are involved in a true demand transition, their sales time series
should be **negatively correlated** in the transition window: as the old SKU
loses sales, the new SKU gains them.  Separately, the magnitude of the old
SKU's decline should closely match the magnitude of the new SKU's rise
(demand conservation).

Three signals are scored:

    correlation_score (0.40)
        Pearson correlation of ΔOld (first differences, inverted sign) and
        ΔNew (first differences) in the post-launch window.  High score if
        the two series move together (positive correlation after sign
        inversion).

    overlap_score (0.35)
        The overlap between the old SKU's active selling period and the new
        SKU's ramp-up window.  A tight temporal overlap increases confidence
        that the two SKUs are directly competing for the same demand.

    volume_match_score (0.25)
        How closely the old SKU's average sales in the pre-window match the
        new SKU's average sales in the post-window (demand conservation at
        the volume level).

Like the curve-fitting method, this method requires external sales history
data and returns an empty list when none is supplied.

Usage
-----
>>> from src.sku_mapping.methods.temporal_comovement import TemporalCovementMethod
>>> method = TemporalCovementMethod(sales_df=weekly_sales_df)
>>> candidates = method.run(product_master)
"""

from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from ..data.schemas import PREDECESSOR_STATUSES, MappingCandidate
from .base import BaseMethod


class TemporalCovementMethod(BaseMethod):
    """
    Temporal co-movement SKU transition discovery (Method 4 / Phase 2).

    Parameters
    ----------
    sales_df:
        Weekly sales history DataFrame with columns
        ``[sku_col, date_col, sales_col]``.
    sku_col:
        SKU identifier column.  Default: ``"sku_id"``.
    date_col:
        Week date column.  Default: ``"week"``.
    sales_col:
        Sales quantity column.  Default: ``"quantity"``.
    window_weeks:
        Half-width of the transition window (weeks).  Default: 13.
    min_data_points:
        Minimum overlapping data points required in the post-launch window
        for both SKUs to attempt scoring.  Default: 4.
    launch_window_days:
        Maximum days between old-SKU launch and new-SKU launch to form a
        candidate pair.  Default: 365.
    min_correlation:
        Minimum inverted-correlation score required to return a candidate.
        Pairs where the two series don't co-move are discarded.
        Default: 0.0 (keep everything that has a positive score).
    """

    name = "temporal"

    # Component weights (must sum to 1.0)
    _W_CORRELATION:  float = 0.40
    _W_OVERLAP:      float = 0.35
    _W_VOLUME_MATCH: float = 0.25

    def __init__(
        self,
        sales_df: Optional[pl.DataFrame] = None,
        sku_col: str = "sku_id",
        date_col: str = "week",
        sales_col: str = "quantity",
        window_weeks: int = 13,
        min_data_points: int = 4,
        launch_window_days: int = 365,
        min_correlation: float = 0.0,
    ):
        self.sales_df = sales_df
        self.sku_col = sku_col
        self.date_col = date_col
        self.sales_col = sales_col
        self.window_weeks = window_weeks
        self.min_data_points = min_data_points
        self.launch_window_days = launch_window_days
        self.min_correlation = min_correlation

        self._index: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
        if sales_df is not None and len(sales_df) > 0:
            self._index = self._build_index(sales_df)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, product_master: pl.DataFrame) -> List[MappingCandidate]:
        """Return temporal co-movement candidates from ``product_master``."""
        if not self._index:
            return []

        old_skus = product_master.filter(
            pl.col("status").is_in(list(PREDECESSOR_STATUSES))
        )
        if old_skus.is_empty():
            return []

        pairs = (
            old_skus.join(
                product_master,
                on=["product_family", "segment"],
                how="inner",
                suffix="_new",
            )
            .filter(pl.col("launch_date") < pl.col("launch_date_new"))
            .filter(
                (pl.col("launch_date_new") - pl.col("launch_date")).dt.total_days()
                <= self.launch_window_days
            )
        )

        if pairs.is_empty():
            return []

        candidates: List[MappingCandidate] = []

        for row in pairs.select(
            ["sku_id", "sku_id_new", "launch_date", "launch_date_new", "eol_date"]
        ).iter_rows(named=True):

            result = self._score_pair(
                old_sku=str(row["sku_id"]),
                new_sku=str(row["sku_id_new"]),
                old_launch=row["launch_date"],
                new_launch=row["launch_date_new"],
                old_eol=row.get("eol_date"),
            )
            if result is not None:
                score, meta = result
                if score > 0.0:
                    candidates.append(
                        MappingCandidate(
                            old_sku=str(row["sku_id"]),
                            new_sku=str(row["sku_id_new"]),
                            method=self.name,
                            method_score=round(score, 4),
                            metadata=meta,
                        )
                    )

        return candidates

    # ------------------------------------------------------------------
    # Private — index
    # ------------------------------------------------------------------

    def _build_index(
        self, sales_df: pl.DataFrame
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Build ``{sku_id: (date_ordinals, sales_values)}`` index."""
        grouped = (
            sales_df
            .with_columns(pl.col(self.sales_col).cast(pl.Float64).fill_null(0.0))
            .sort([self.sku_col, self.date_col])
            .group_by(self.sku_col)
            .agg([
                pl.col(self.date_col).alias("dates"),
                pl.col(self.sales_col).alias("sales"),
            ])
        )

        index: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for row in grouped.iter_rows(named=True):
            sku_id = str(row[self.sku_col])
            raw_dates = row["dates"]
            if not raw_dates:
                continue
            try:
                ordinals = np.array(
                    [d.toordinal() for d in raw_dates], dtype=np.float64
                )
            except AttributeError:
                ordinals = np.arange(len(raw_dates), dtype=np.float64)

            sales = np.array(row["sales"], dtype=np.float64)
            order = np.argsort(ordinals)
            index[sku_id] = (ordinals[order], sales[order])

        return index

    # ------------------------------------------------------------------
    # Private — scoring
    # ------------------------------------------------------------------

    def _score_pair(
        self,
        old_sku: str,
        new_sku: str,
        old_launch: date,
        new_launch: date,
        old_eol: Optional[date],
    ) -> Optional[Tuple[float, dict]]:
        """Score one (old_sku, new_sku) pair using temporal co-movement."""
        if old_sku not in self._index or new_sku not in self._index:
            return None

        old_ord, old_sales = self._index[old_sku]
        new_ord, new_sales = self._index[new_sku]

        try:
            launch_ord = float(new_launch.toordinal())
        except AttributeError:
            return None

        window_ord = float(self.window_weeks * 7)
        pre_start  = launch_ord - window_ord
        post_end   = launch_ord + window_ord

        # ── Minimum data check ───────────────────────────────────────────
        # Both SKUs must have enough post-window data to make any assessment.
        post_old_mask = (old_ord >= launch_ord) & (old_ord <= post_end)
        post_new_mask = (new_ord >= launch_ord) & (new_ord <= post_end)
        if int(post_old_mask.sum()) < self.min_data_points or int(post_new_mask.sum()) < self.min_data_points:
            return None

        # ── Component 1: correlation score ───────────────────────────────
        # Align both series on a shared weekly grid in the post-window

        old_post_t = old_ord[post_old_mask]
        old_post_s = old_sales[post_old_mask]
        new_post_t = new_ord[post_new_mask]
        new_post_s = new_sales[post_new_mask]

        correlation_score = 0.0
        n_overlap = 0

        if len(old_post_t) >= self.min_data_points and len(new_post_t) >= self.min_data_points:
            # Interpolate both onto a common weekly grid
            t_start = max(old_post_t[0], new_post_t[0])
            t_end   = min(old_post_t[-1], new_post_t[-1])

            if t_end > t_start:
                n_pts = min(len(old_post_t), len(new_post_t))
                common_t = np.linspace(t_start, t_end, max(n_pts, 2))
                n_overlap = len(common_t)

                old_aligned = np.interp(common_t, old_post_t, old_post_s)
                new_aligned = np.interp(common_t, new_post_t, new_post_s)

                # First differences (week-over-week changes)
                d_old = np.diff(old_aligned)
                d_new = np.diff(new_aligned)

                if len(d_old) >= 2:
                    # Invert old: old should decline when new grows → negative corr → positive score
                    try:
                        corr = float(np.corrcoef(-d_old, d_new)[0, 1])
                        if not np.isnan(corr):
                            correlation_score = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))
                    except Exception:
                        correlation_score = 0.0

        # ── Component 2: temporal overlap score ──────────────────────────
        # How much of the old SKU's selling life overlaps with the transition window?
        overlap_score = 0.0
        try:
            old_first_ord = float(old_launch.toordinal())
            old_last_ord  = float(old_eol.toordinal()) if old_eol else old_ord[-1]

            # Overlap between [old_first, old_last] and [new_launch, post_end]
            overlap_start = max(old_first_ord, launch_ord)
            overlap_end   = min(old_last_ord, post_end)

            if overlap_end > overlap_start:
                overlap_score = float(
                    np.clip((overlap_end - overlap_start) / window_ord, 0.0, 1.0)
                )
        except (AttributeError, IndexError):
            overlap_score = 0.5  # neutral fallback

        # ── Component 3: volume match score ──────────────────────────────
        volume_match_score = 0.0
        pre_old_mask = (old_ord >= pre_start) & (old_ord < launch_ord)
        old_pre_sales = old_sales[pre_old_mask]
        new_post_avg = float(np.mean(new_post_s)) if len(new_post_s) > 0 else 0.0
        old_pre_avg  = float(np.mean(old_pre_sales)) if len(old_pre_sales) > 0 else 0.0

        if old_pre_avg > 0 and new_post_avg > 0:
            ratio = min(old_pre_avg, new_post_avg) / max(old_pre_avg, new_post_avg)
            volume_match_score = float(ratio)
        elif old_pre_avg > 0 or new_post_avg > 0:
            volume_match_score = 0.2  # partial credit if one side exists

        # ── Final score ──────────────────────────────────────────────────
        final_score = (
            self._W_CORRELATION  * correlation_score
            + self._W_OVERLAP    * overlap_score
            + self._W_VOLUME_MATCH * volume_match_score
        )

        meta = {
            "correlation_score":   round(correlation_score, 3),
            "overlap_score":       round(overlap_score, 3),
            "volume_match_score":  round(volume_match_score, 3),
            "n_overlap_points":    n_overlap,
            "old_pre_avg":         round(old_pre_avg, 2),
            "new_post_avg":        round(new_post_avg, 2),
        }

        return float(final_score), meta
