"""Data classes and Polars schema definitions for the SKU mapping pipeline."""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import polars as pl

# ── Input schema ──────────────────────────────────────────────────────────────

#: Column types used when creating or validating a product master DataFrame.
PRODUCT_MASTER_SCHEMA: Dict[str, pl.DataType] = {
    "sku_id": pl.Utf8,
    "sku_description": pl.Utf8,
    "product_family": pl.Utf8,
    "product_category": pl.Utf8,
    "form_factor": pl.Utf8,      # nullable
    "price_tier": pl.Utf8,       # nullable
    "country": pl.List(pl.Utf8), # parsed from comma-separated CSV field
    "segment": pl.Utf8,
    "launch_date": pl.Date,
    "eol_date": pl.Date,         # nullable
    "status": pl.Utf8,
}

#: Valid status values in the product master.
VALID_STATUSES = {"Active", "Discontinued", "Planned", "Declining"}

#: Statuses that mark a SKU as a candidate *predecessor* in a transition.
PREDECESSOR_STATUSES = {"Discontinued", "Declining"}

#: Statuses that mark a SKU as a candidate *successor* in a transition.
SUCCESSOR_STATUSES = {"Active", "Planned"}


# ── Intermediate representation ───────────────────────────────────────────────

@dataclass
class MappingCandidate:
    """A single method's candidate mapping between an old and a new SKU."""

    old_sku: str
    new_sku: str
    method: str
    method_score: float
    metadata: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"MappingCandidate({self.old_sku!r} → {self.new_sku!r}, "
            f"method={self.method!r}, score={self.method_score:.3f})"
        )


# ── Output schema ─────────────────────────────────────────────────────────────

MAPPING_TYPES = frozenset({"1-to-1", "1-to-Many", "Many-to-1", "Many-to-Many"})
CONFIDENCE_LEVELS = frozenset({"High", "Medium", "Low", "Very Low"})
VALIDATION_STATUSES = frozenset({"Pending Review", "Confirmed", "Modified", "Rejected"})


@dataclass
class MappingRecord:
    """
    One row in the final mapping table (Section 5 of the design document).

    This is the unit that flows into the forecasting platform's transition
    management module after planner validation.
    """

    mapping_id: str
    old_sku: str
    new_sku: str
    mapping_type: str                       # 1-to-1 | 1-to-Many | Many-to-1 | Many-to-Many
    proportion: float                       # fraction of old demand attributed to this new SKU
    confidence_score: float                 # fused score in [0, 1]
    confidence_level: str                   # High | Medium | Low | Very Low
    methods_matched: List[str]              # which methods produced this candidate
    transition_start_week: Optional[date]   # estimated start of transition
    transition_end_week: Optional[date]     # estimated end of transition (nullable)
    old_sku_lifecycle_stage: str            # lifecycle stage of the old SKU
    validation_status: str = "Pending Review"
    validated_by: Optional[str] = None
    validation_date: Optional[date] = None
    notes: Optional[str] = None
