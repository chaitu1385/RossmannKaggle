"""
Bayesian Demand-Proportion Estimator
(Design document §4.2 / Phase 2)

Replaces the equal-split fallback for multi-mapped SKU pairs with a
Dirichlet-Bayes weighted allocation that uses mapping confidence scores
as evidence.

Theory
------
For any group of N candidate pairs that compete for the same demand pool
(e.g. one old SKU splitting demand across three new SKUs), the equal-split
prior p_i = 1/N is updated with the observed confidence score s_i:

    posterior(i) = (α/N  +  s_i) / (α  +  ΣS)

Where:
    α  = concentration parameter (controls prior strength)
    N  = number of candidates in the allocation group
    s_i = confidence score of the i-th candidate ∈ [0, 1]
    ΣS = sum of all scores in the group

Properties
----------
- When all s_i are equal: posterior = prior = 1/N (uninformative data).
- High-confidence pairs attract more demand share.
- When α → 0: pure score-proportional allocation.
- When α → ∞: pure equal-split (prior dominates evidence).
- Default α = 0.5: weakly informative, allows scores to influence allocation
  while guarding against extreme concentrations on noisy high scores.
- Outputs always sum to 1.0 (normalised with rounding correction).

Allocation groups by mapping type
----------------------------------
1-to-1     : no change (proportion = 1.0 by definition).
1-to-Many  : group = all (old, *) records for the same old_sku.
             proportion = fraction of old's demand going to each new SKU.
Many-to-1  : group = all (*, new) records for the same new_sku.
             proportion = fraction of new's demand inherited from each old SKU.
Many-to-Many: group per old_sku (same as 1-to-Many within each old SKU's basket).
"""

from collections import defaultdict
from typing import Dict, List

from ..data.schemas import MappingRecord


class BayesianProportionEstimator:
    """
    Update ``proportion`` fields of ``MappingRecord`` objects using
    Dirichlet-Bayes weighting on confidence scores.

    Parameters
    ----------
    concentration:
        Dirichlet concentration parameter α.  Values between 0.1 and 2.0
        are typical.  Default: 0.5.
    """

    def __init__(self, concentration: float = 0.5):
        if concentration < 0:
            raise ValueError(f"concentration must be >= 0, got {concentration}")
        self.concentration = concentration

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def estimate(self, records: List[MappingRecord]) -> List[MappingRecord]:
        """
        Update proportions in-place for all multi-mapped records.

        Parameters
        ----------
        records:
            Output of ``CandidateFusion._assign_mapping_types_and_proportions``.
            ``mapping_type`` and ``confidence_score`` must already be set.

        Returns
        -------
        The same list with updated ``proportion`` fields.
        """
        if not records:
            return records

        # Build index structures
        old_groups: Dict[str, List[MappingRecord]] = defaultdict(list)  # old_sku → records
        new_groups: Dict[str, List[MappingRecord]] = defaultdict(list)  # new_sku → records

        for r in records:
            old_groups[r.old_sku].append(r)
            new_groups[r.new_sku].append(r)

        updated_ids = set()

        for r in records:
            if id(r) in updated_ids:
                continue

            if r.mapping_type == "1-to-1":
                r.proportion = 1.0
                updated_ids.add(id(r))

            elif r.mapping_type in ("1-to-Many", "Many-to-Many"):
                # Proportion = old_sku's demand share going to each new_sku.
                # Group: all records with this old_sku.
                group = old_groups[r.old_sku]
                self._update_group(group)
                for g in group:
                    updated_ids.add(id(g))

            elif r.mapping_type == "Many-to-1":
                # Proportion = share of new_sku's demand inherited from each old.
                # Group: all records with this new_sku.
                group = new_groups[r.new_sku]
                self._update_group(group)
                for g in group:
                    updated_ids.add(id(g))

        return records

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _update_group(self, group: List[MappingRecord]) -> None:
        """Apply Dirichlet-Bayes update to a single allocation group (in-place)."""
        n = len(group)
        if n == 0:
            return
        if n == 1:
            group[0].proportion = 1.0
            return

        scores = [r.confidence_score for r in group]
        proportions = self._dirichlet_posterior(scores)

        for record, proportion in zip(group, proportions):
            record.proportion = proportion
            suffix = f"Bayesian proportion (α={self.concentration:.2g})."
            if record.notes:
                record.notes = f"{record.notes} {suffix}"
            else:
                record.notes = suffix

    def _dirichlet_posterior(self, scores: List[float]) -> List[float]:
        """
        Compute normalised Dirichlet-Bayes posterior proportions.

        Formula:  p_i = (α/N + s_i) / (α + ΣS)

        Returns a list of floats that sum to exactly 1.0 (with rounding
        correction applied to the highest-weight item).
        """
        n = len(scores)
        alpha = self.concentration
        alpha_prior = alpha / n       # per-item prior pseudo-count
        total_score = sum(scores)
        denominator = alpha + total_score

        if denominator <= 0:
            # Degenerate: all scores zero → equal split
            return [round(1.0 / n, 6)] * n

        raw = [(alpha_prior + s) / denominator for s in scores]

        # Normalise (floating-point safety)
        raw_sum = sum(raw)
        if raw_sum > 0:
            raw = [x / raw_sum for x in raw]

        # Round to 4 decimal places
        proportions = [round(p, 4) for p in raw]

        # Correct rounding drift on the highest item
        rounding_error = round(1.0 - sum(proportions), 4)
        if rounding_error != 0.0:
            max_idx = proportions.index(max(proportions))
            proportions[max_idx] = round(proportions[max_idx] + rounding_error, 4)

        return proportions
