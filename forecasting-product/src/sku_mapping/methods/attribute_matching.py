"""
Method 1 — Attribute-Based Matching
(Design document §3.1)

For each (old_sku × new_sku) pair inside the same product_family × segment
bucket, score how well the attributes align.  Fully vectorised using Polars.
"""

from typing import Dict, List

import polars as pl

from ..data.schemas import PREDECESSOR_STATUSES, MappingCandidate
from .base import BaseMethod


class AttributeMatchingMethod(BaseMethod):
    """Attribute-based SKU transition discovery (Method 1)."""

    name = "attribute"

    # Scoring weights from the design document
    _BASE_SCORE: float = 0.30
    _WEIGHTS: Dict[str, float] = {
        "price_tier": 0.20,
        "form_factor": 0.20,
        "product_category": 0.10,
        "launch_gap_lt_90d": 0.10,
    }

    def __init__(self, launch_window_days: int = 365):
        """
        Parameters
        ----------
        launch_window_days:
            Maximum number of days between the old SKU's launch date and the
            new SKU's launch date for a pair to be considered at all.
            Default: 365 days (1 year). The design doc references 6 months
            relative to EOL; using launch-to-launch we allow a full year so
            that transitions like annual product refreshes are included.
        """
        self.launch_window_days = launch_window_days

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, product_master: pl.DataFrame) -> List[MappingCandidate]:
        """
        Return attribute-based candidates from ``product_master``.

        Steps
        -----
        1. Split into predecessor pool (Discontinued / Declining) and
           successor pool (Active / Planned).
        2. Inner-join on ``product_family × segment`` to form candidate pairs
           (a Polars cross-join scoped to the same bucket).
        3. Filter pairs by date ordering and launch-window.
        4. Filter pairs by country overlap.
        5. Score all pairs vectorised; keep score > 0.
        """
        old_skus = product_master.filter(
            pl.col("status").is_in(list(PREDECESSOR_STATUSES))
        )
        # Any SKU can be a successor — including Discontinued ones that are
        # themselves replaced later (middle-of-chain generations).
        new_skus = product_master

        if old_skus.is_empty():
            return []

        # Join on (product_family, segment) to scope pairs within the same bucket.
        # Polars join on multiple keys returns both sides; suffix "_new" on right.
        pairs = old_skus.join(
            new_skus,
            on=["product_family", "segment"],
            how="inner",
            suffix="_new",
        )

        # Guard: old must have launched earlier than new
        pairs = pairs.filter(pl.col("launch_date") < pl.col("launch_date_new"))

        # Guard: launch window
        pairs = pairs.filter(
            (pl.col("launch_date_new") - pl.col("launch_date")).dt.total_days()
            <= self.launch_window_days
        )

        if pairs.is_empty():
            return []

        # Guard: at least one country in common
        pairs = pairs.filter(
            pl.col("country")
            .list.set_intersection(pl.col("country_new"))
            .list.len()
            > 0
        )

        if pairs.is_empty():
            return []

        # Vectorised scoring
        pairs = self._add_scores(pairs)

        # Keep only pairs where the score exceeds the base (i.e. at least one
        # attribute matched on top of the structural criteria).
        pairs = pairs.filter(pl.col("_method_score") > 0)

        return self._to_candidates(pairs)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _add_scores(self, pairs: pl.DataFrame) -> pl.DataFrame:
        """Compute the attribute score for every row in one vectorised pass."""

        base = pl.lit(self._BASE_SCORE)

        # Price tier match (both columns must be non-null)
        price_match = (
            pl.col("price_tier").is_not_null()
            & pl.col("price_tier_new").is_not_null()
            & (pl.col("price_tier") == pl.col("price_tier_new"))
        ).cast(pl.Float64) * self._WEIGHTS["price_tier"]

        # Form factor match
        form_match = (
            pl.col("form_factor").is_not_null()
            & pl.col("form_factor_new").is_not_null()
            & (pl.col("form_factor") == pl.col("form_factor_new"))
        ).cast(pl.Float64) * self._WEIGHTS["form_factor"]

        # Product category match (already filtered to same family, but
        # finer-grained categories within a family may differ)
        cat_match = (
            pl.col("product_category") == pl.col("product_category_new")
        ).cast(pl.Float64) * self._WEIGHTS["product_category"]

        # Launch gap < 90 days bonus
        gap_days = (
            (pl.col("launch_date_new") - pl.col("launch_date")).dt.total_days()
        )
        gap_bonus = (gap_days < 90).cast(pl.Float64) * self._WEIGHTS["launch_gap_lt_90d"]

        score = (base + price_match + form_match + cat_match + gap_bonus).clip(
            lower_bound=0.0, upper_bound=1.0
        )

        return pairs.with_columns(score.alias("_method_score"))

    def _to_candidates(self, scored: pl.DataFrame) -> List[MappingCandidate]:
        candidates: List[MappingCandidate] = []

        for row in scored.select(
            ["sku_id", "sku_id_new", "_method_score",
             "price_tier", "price_tier_new",
             "form_factor", "form_factor_new",
             "launch_date", "launch_date_new"]
        ).iter_rows(named=True):
            candidates.append(
                MappingCandidate(
                    old_sku=row["sku_id"],
                    new_sku=row["sku_id_new"],
                    method=self.name,
                    method_score=round(row["_method_score"], 4),
                    metadata={
                        "price_tier_match": row["price_tier"] == row["price_tier_new"],
                        "form_factor_match": row["form_factor"] == row["form_factor_new"],
                        "launch_gap_days": (
                            row["launch_date_new"] - row["launch_date"]
                        ).days,
                    },
                )
            )

        return candidates
