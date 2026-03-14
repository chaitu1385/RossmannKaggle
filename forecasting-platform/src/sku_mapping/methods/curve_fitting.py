"""
Method 2 — Demand Curve Fitting
(Design document §3.3 / Phase 2)

Detects SKU transitions by analysing the demand shape in the period
surrounding a new SKU's launch date.

Theory
------
When an old SKU is replaced by a new one, two things happen simultaneously:

  1. Old SKU demand **declines** — a downward trend starting around T0
     (new SKU launch date).
  2. New SKU demand **ramps up** — an upward trend starting at T0.
  3. **Demand conservation** — if the transition is clean, the combined
     old + new demand should stay roughly constant (low coefficient of
     variation) during the transition window.

This method fits linear trends to each SKU's sales in a ±``window_weeks``
window around T0 and scores four complementary signals:

    decline_score   (0.30) – old SKU has a negative slope post-T0
    ramp_score      (0.30) – new SKU has a positive slope post-T0
    complement_score(0.25) – combined demand is stable (low CV) post-T0
    scale_score     (0.15) – old SKU pre-launch avg ≈ new SKU post-launch avg

Unlike the attribute and naming methods this method requires external
sales history data.  When ``sales_df`` is None or empty the method
returns an empty candidate list — the other methods still run and can
independently discover the same pair.

Usage
-----
>>> from src.sku_mapping.methods.curve_fitting import CurveFittingMethod
>>> method = CurveFittingMethod(sales_df=weekly_sales_df)
>>> candidates = method.run(product_master)
"""

from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from ..data.schemas import PREDECESSOR_STATUSES, MappingCandidate
from .base import BaseMethod


class CurveFittingMethod(BaseMethod):
    """
    Demand curve-fitting SKU transition discovery (Method 2 / Phase 2).

    Parameters
    ----------
    sales_df:
        Weekly sales history.  Must contain at least:
        ``[sku_col, date_col, sales_col]``.  Rows for SKUs not present
        in the product master are silently ignored.
    sku_col:
        Column name for SKU identifier.  Default: ``"sku_id"``.
    date_col:
        Column name for the week date.  Default: ``"week"``.
    sales_col:
        Column name for the sales quantity.  Default: ``"quantity"``.
    window_weeks:
        Half-width (in weeks) of the transition window centred on the
        new SKU's launch date.  Pre-window = [T0 - W, T0);
        Post-window = [T0, T0 + W].  Default: 13.
    min_data_points:
        Minimum data points required in the post-window for the old SKU
        to attempt curve fitting.  Default: 4.
    launch_window_days:
        Maximum days between old-SKU launch and new-SKU launch to
        consider a pair at all.  Default: 365.
    """

    name = "curve"

    # Component weights (must sum to 1.0)
    _W_DECLINE:     float = 0.30
    _W_RAMP:        float = 0.30
    _W_COMPLEMENT:  float = 0.25
    _W_SCALE:       float = 0.15

    def __init__(
        self,
        sales_df: Optional[pl.DataFrame] = None,
        sku_col: str = "sku_id",
        date_col: str = "week",
        sales_col: str = "quantity",
        window_weeks: int = 13,
        min_data_points: int = 4,
        launch_window_days: int = 365,
    ):
        self.sales_df = sales_df
        self.sku_col = sku_col
        self.date_col = date_col
        self.sales_col = sales_col
        self.window_weeks = window_weeks
        self.min_data_points = min_data_points
        self.launch_window_days = launch_window_days

        # {sku_id: (date_ordinals_np, sales_np)} — built lazily on first run
        self._index: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
        if sales_df is not None and len(sales_df) > 0:
            self._index = self._build_index(sales_df)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, product_master: pl.DataFrame) -> List[MappingCandidate]:
        """
        Return demand-curve candidates from ``product_master``.

        Returns an empty list if no sales history was supplied.
        """
        if not self._index:
            return []

        # ── Build candidate pairs ────────────────────────────────────────
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

        # ── Score each pair ──────────────────────────────────────────────
        candidates: List[MappingCandidate] = []

        for row in pairs.select(
            ["sku_id", "sku_id_new", "launch_date", "launch_date_new"]
        ).iter_rows(named=True):

            result = self._score_pair(
                old_sku=str(row["sku_id"]),
                new_sku=str(row["sku_id_new"]),
                new_launch=row["launch_date_new"],
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
        """
        Build a per-SKU sales index: ``{sku_id: (ordinals, sales)}``.

        Dates are converted to integer ordinals (``date.toordinal()``)
        so that arithmetic in ``_score_pair`` works without datetime
        dependencies.
        """
        # Sort dates within each SKU group, then collect as lists
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
            sales_list = row["sales"]

            if not raw_dates:
                continue

            # Convert dates to ordinals — handles datetime.date and polars Date
            try:
                ordinals = np.array(
                    [d.toordinal() for d in raw_dates], dtype=np.float64
                )
            except AttributeError:
                # Fallback: use positional index if toordinal() not available
                ordinals = np.arange(len(raw_dates), dtype=np.float64)

            sales = np.array(sales_list, dtype=np.float64)

            # Sort by ordinal (in case group_by reordered)
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
        new_launch: date,
    ) -> Optional[Tuple[float, dict]]:
        """
        Score one (old_sku, new_sku) pair.

        Returns ``(final_score, metadata_dict)`` or ``None`` if there
        is insufficient data to compute any component.
        """
        if old_sku not in self._index or new_sku not in self._index:
            return None

        old_ord, old_sales = self._index[old_sku]
        new_ord, new_sales = self._index[new_sku]

        # Convert launch date to ordinal
        try:
            launch_ord = float(new_launch.toordinal())
        except AttributeError:
            return None

        window_ord = float(self.window_weeks * 7)

        # Define windows (ordinal boundaries)
        pre_start  = launch_ord - window_ord
        pre_end    = launch_ord
        post_start = launch_ord
        post_end   = launch_ord + window_ord

        # Slice data into windows
        old_pre_mask  = (old_ord >= pre_start) & (old_ord < pre_end)
        old_post_mask = (old_ord >= post_start) & (old_ord <= post_end)
        new_post_mask = (new_ord >= post_start) & (new_ord <= post_end)

        old_pre_sales  = old_sales[old_pre_mask]
        old_post_t     = old_ord[old_post_mask] - post_start
        old_post_sales = old_sales[old_post_mask]
        new_post_t     = new_ord[new_post_mask] - post_start
        new_post_sales = new_sales[new_post_mask]

        # Need at least some post-window data for the old SKU
        if len(old_post_sales) < self.min_data_points:
            return None

        # ── Component 1: decline score ───────────────────────────────────
        decline_score = 0.0
        if len(old_post_sales) >= 2:
            slope, _ = np.polyfit(old_post_t, old_post_sales, 1)
            ref = float(np.mean(old_pre_sales)) if len(old_pre_sales) > 0 else float(np.mean(old_post_sales))
            if ref > 0:
                # Normalised slope: fraction of baseline lost per window
                decline_score = float(np.clip(-slope * window_ord / ref, 0.0, 1.0))

        # ── Component 2: ramp score ──────────────────────────────────────
        ramp_score = 0.0
        if len(new_post_sales) >= 2:
            slope, _ = np.polyfit(new_post_t, new_post_sales, 1)
            ref = max(float(np.mean(new_post_sales)), 1e-6)
            ramp_score = float(np.clip(slope * window_ord / ref, 0.0, 1.0))

        # ── Component 3: complementarity ────────────────────────────────
        complement_score = 0.0
        if len(old_post_sales) >= 2 and len(new_post_sales) >= 2:
            old_scale = max(
                float(np.max(old_pre_sales)) if len(old_pre_sales) > 0
                else float(np.max(old_post_sales)),
                1e-6,
            )
            new_scale = max(float(np.max(new_post_sales)), 1e-6)

            old_norm = old_post_sales / old_scale
            new_norm = new_post_sales / new_scale

            # Build a common time grid over the overlap of both windows
            t_start = max(old_post_t[0] if len(old_post_t) else 0.0,
                          new_post_t[0] if len(new_post_t) else 0.0)
            t_end   = min(old_post_t[-1] if len(old_post_t) else window_ord,
                          new_post_t[-1] if len(new_post_t) else window_ord)

            if t_end > t_start:
                n_pts = min(len(old_post_t), len(new_post_t), 10)
                common_t = np.linspace(t_start, t_end, max(n_pts, 2))
                old_interp = np.interp(common_t, old_post_t, old_norm)
                new_interp = np.interp(common_t, new_post_t, new_norm)

                combined      = old_interp + new_interp
                combined_mean = float(np.mean(combined))
                combined_std  = float(np.std(combined))

                if combined_mean > 0:
                    cv = combined_std / combined_mean
                    complement_score = float(np.clip(1.0 - cv, 0.0, 1.0))

        # ── Component 4: scale similarity ───────────────────────────────
        scale_score = 0.0
        old_peak = (
            float(np.max(old_pre_sales)) if len(old_pre_sales) > 0
            else float(np.max(old_post_sales)) if len(old_post_sales) > 0
            else 0.0
        )
        new_plateau = float(np.max(new_post_sales)) if len(new_post_sales) > 0 else 0.0

        if old_peak > 0 and new_plateau > 0:
            scale_score = float(
                min(old_peak, new_plateau) / max(old_peak, new_plateau)
            )

        # ── Final weighted score ─────────────────────────────────────────
        final_score = (
            self._W_DECLINE    * decline_score
            + self._W_RAMP     * ramp_score
            + self._W_COMPLEMENT * complement_score
            + self._W_SCALE    * scale_score
        )

        meta = {
            "decline_score":    round(decline_score, 3),
            "ramp_score":       round(ramp_score, 3),
            "complement_score": round(complement_score, 3),
            "scale_score":      round(scale_score, 3),
            "old_pre_weeks":    int(old_pre_mask.sum()),
            "old_post_weeks":   int(old_post_mask.sum()),
            "new_post_weeks":   int(new_post_mask.sum()),
        }

        return float(final_score), meta
