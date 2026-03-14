"""
Candidate Fusion & Scoring
(Design document §4)

Aggregates candidates from all discovery methods into a ranked mapping table
with unified confidence scores, mapping types, and proportion estimates.
"""

import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import polars as pl

from ..data.schemas import MappingCandidate, MappingRecord
from .bayesian_proportions import BayesianProportionEstimator

# ── Weight tables ─────────────────────────────────────────────────────────────

# Original four-method weights (Phase 2+, design doc §4.1)
_FULL_WEIGHTS: Dict[str, float] = {
    "attribute": 0.30,
    "temporal":  0.30,
    "naming":    0.15,
    "curve":     0.25,
}

# Phase 1 normalised weights (attribute + naming only).
# Derived as: w_i / sum(available w_i)
_PHASE1_WEIGHTS: Dict[str, float] = {
    "attribute": _FULL_WEIGHTS["attribute"]
    / (_FULL_WEIGHTS["attribute"] + _FULL_WEIGHTS["naming"]),   # ≈ 0.667
    "naming": _FULL_WEIGHTS["naming"]
    / (_FULL_WEIGHTS["attribute"] + _FULL_WEIGHTS["naming"]),   # ≈ 0.333
}

# Multi-method bonus applied when ≥2 methods agree (Phase 1 version of the
# design doc's "3+ methods" rule, adjusted for having only 2 methods available)
_MULTI_METHOD_BONUS: float = 0.10

# Confidence thresholds (design doc §4.1 table)
_CONFIDENCE_THRESHOLDS: List[Tuple[float, str]] = [
    (0.75, "High"),
    (0.50, "Medium"),
    (0.30, "Low"),
    (0.00, "Very Low"),
]


def _classify_confidence(score: float) -> str:
    for threshold, label in _CONFIDENCE_THRESHOLDS:
        if score >= threshold:
            return label
    return "Very Low"


class CandidateFusion:
    """
    Fuses candidates from multiple methods into final ``MappingRecord`` objects.

    Phase behaviour
    ---------------
    In Phase 1 (attribute + naming only), the method weights are normalised
    so that the fused score still spans [0, 1].  When Phase 2 methods are
    added, pass ``weights=_FULL_WEIGHTS`` or leave the default to use all
    registered methods with their original doc weights.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        multi_method_bonus: float = _MULTI_METHOD_BONUS,
        min_confidence: str = "Very Low",
        bayesian_proportions: bool = False,
        bayesian_concentration: float = 0.5,
    ):
        """
        Parameters
        ----------
        weights:
            Override the default method weights.  Keys must match method
            ``name`` values.  Defaults to Phase 1 weights.
        multi_method_bonus:
            Added to the fused score when ≥2 methods agree.
        min_confidence:
            Minimum confidence level to include in the output.
            E.g. ``"Low"`` discards Very Low candidates.
        bayesian_proportions:
            When True, use Dirichlet-Bayes weighting instead of equal-split
            for multi-mapped pairs (Phase 2).  Default: False (Phase 1 compat).
        bayesian_concentration:
            Dirichlet concentration parameter α for Bayesian proportion
            estimation.  Only used when ``bayesian_proportions=True``.
            Default: 0.5.
        """
        self.weights = weights or _PHASE1_WEIGHTS
        self.multi_method_bonus = multi_method_bonus
        self.min_confidence = min_confidence
        self.bayesian_proportions = bayesian_proportions
        self._bayesian_estimator = (
            BayesianProportionEstimator(concentration=bayesian_concentration)
            if bayesian_proportions else None
        )

        _order = ["High", "Medium", "Low", "Very Low"]
        self._min_rank = _order.index(min_confidence) if min_confidence in _order else 3

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fuse(
        self,
        candidates: List[MappingCandidate],
        product_master: pl.DataFrame,
    ) -> List[MappingRecord]:
        """
        Aggregate ``candidates`` into a list of ``MappingRecord`` objects.

        Steps
        -----
        1. Group candidates by (old_sku, new_sku) pair.
        2. Compute weighted fused score with optional multi-method bonus.
        3. Determine mapping types (1-to-1, 1-to-Many, Many-to-1, Many-to-Many).
        4. Estimate proportions (equal split fallback in Phase 1).
        5. Look up lifecycle metadata from the product master.
        6. Discard pairs below ``min_confidence``.
        """
        if not candidates:
            return []

        # Build lookup: sku_id → row dict from product master
        sku_lookup: Dict[str, dict] = {
            row["sku_id"]: row
            for row in product_master.iter_rows(named=True)
        }

        # ── Step 1: group by pair ────────────────────────────────────────────
        pair_groups: Dict[Tuple[str, str], List[MappingCandidate]] = defaultdict(list)
        for c in candidates:
            pair_groups[(c.old_sku, c.new_sku)].append(c)

        # ── Step 2: compute fused scores ─────────────────────────────────────
        proto_records: List[MappingRecord] = []

        for (old_sku, new_sku), group in pair_groups.items():
            score_by_method: Dict[str, float] = {
                c.method: c.method_score for c in group
            }
            methods_matched = sorted(score_by_method.keys())

            # Normalise weights to whatever methods are actually present
            available_weight = sum(
                self.weights.get(m, 0.0) for m in methods_matched
            )
            if available_weight == 0.0:
                continue

            raw_score = sum(
                self.weights.get(m, 0.0) * s
                for m, s in score_by_method.items()
            ) / available_weight

            # Multi-method bonus
            if len(methods_matched) >= 2:
                raw_score = min(1.0, raw_score + self.multi_method_bonus)

            confidence_level = _classify_confidence(raw_score)

            # Filter by minimum confidence
            conf_order = ["High", "Medium", "Low", "Very Low"]
            if conf_order.index(confidence_level) > self._min_rank:
                continue

            old_meta = sku_lookup.get(old_sku, {})
            new_meta = sku_lookup.get(new_sku, {})

            proto_records.append(
                MappingRecord(
                    mapping_id=f"MAP-{uuid.uuid4().hex[:8].upper()}",
                    old_sku=old_sku,
                    new_sku=new_sku,
                    mapping_type="1-to-1",    # placeholder; updated below
                    proportion=1.0,           # placeholder; updated below
                    confidence_score=round(raw_score, 4),
                    confidence_level=confidence_level,
                    methods_matched=methods_matched,
                    transition_start_week=new_meta.get("launch_date"),
                    transition_end_week=old_meta.get("eol_date"),
                    old_sku_lifecycle_stage=old_meta.get("status", "Unknown"),
                    notes=None,
                )
            )

        # ── Step 3+4: mapping types and proportions ──────────────────────────
        proto_records = self._assign_mapping_types_and_proportions(proto_records)

        # ── Step 5 (Phase 2): Bayesian proportion refinement ─────────────────
        if self._bayesian_estimator is not None:
            proto_records = self._bayesian_estimator.estimate(proto_records)

        # Sort: highest confidence first, then by score descending
        conf_rank = {"High": 0, "Medium": 1, "Low": 2, "Very Low": 3}
        proto_records.sort(
            key=lambda r: (conf_rank[r.confidence_level], -r.confidence_score)
        )

        return proto_records

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _assign_mapping_types_and_proportions(
        self, records: List[MappingRecord]
    ) -> List[MappingRecord]:
        """
        Determine mapping type for each record by counting how many new SKUs
        each old SKU maps to (and vice versa), then assign equal-split
        proportions as a Phase 1 fallback.
        """
        old_to_news: Dict[str, List[str]] = defaultdict(list)
        new_to_olds: Dict[str, List[str]] = defaultdict(list)

        for r in records:
            old_to_news[r.old_sku].append(r.new_sku)
            new_to_olds[r.new_sku].append(r.old_sku)

        for r in records:
            n_new = len(old_to_news[r.old_sku])
            n_old = len(new_to_olds[r.new_sku])

            if n_new == 1 and n_old == 1:
                r.mapping_type = "1-to-1"
                r.proportion = 1.0

            elif n_new > 1 and n_old == 1:
                r.mapping_type = "1-to-Many"
                r.proportion = round(1.0 / n_new, 4)
                r.notes = (
                    "Proportion estimated (equal split). "
                    "Planner review required."
                )

            elif n_new == 1 and n_old > 1:
                r.mapping_type = "Many-to-1"
                r.proportion = round(1.0 / n_old, 4)
                r.notes = (
                    "Many-to-1 mapping. "
                    "Proportion estimated (equal split). "
                    "Planner review required."
                )

            else:
                r.mapping_type = "Many-to-Many"
                r.proportion = round(1.0 / n_new, 4)
                r.notes = (
                    "Complex many-to-many transition. "
                    "Manual review required. "
                    "Proportions are equal-split estimates only."
                )

        return records
