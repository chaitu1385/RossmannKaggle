"""
Method 3 — Naming Convention Parsing
(Design document §3.2)

Parses SKU IDs and descriptions for generation markers, strips them to
obtain a "base name", then uses rapidfuzz string similarity to find
predecessor-successor pairs.

Scoring formula (from the design document):
  score = similarity × 0.5 + generation_marker_detected × 0.3 + same_family × 0.2
"""

import re
from typing import List, Optional, Tuple

import polars as pl
from rapidfuzz import fuzz

from ..data.schemas import MappingCandidate, PREDECESSOR_STATUSES, SUCCESSOR_STATUSES
from .base import BaseMethod


# ── Generation-marker regex patterns ─────────────────────────────────────────
# Each tuple: (compiled regex, marker_type_label)
# All patterns are applied to the *lowercased* name/description.
# They are tried in order; first match wins.

_MARKER_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # "MkII", "Mk2", "Mk III", "Mark 2" etc.
    (re.compile(r'\s+mk\s*(?:ii|iii|iv|v|vi|\d+)$'), "mk"),
    # "Gen2", "Gen 3"
    (re.compile(r'\s+gen\s*\d+$'), "gen"),
    # "v2", "v 3"
    (re.compile(r'\s+v\s*\d+$'), "version"),
    # "Plus"
    (re.compile(r'\s+plus$'), "plus"),
    # "Pro" — only when trailing
    (re.compile(r'\s+pro$'), "pro"),
    # "Lite", "Sport"
    (re.compile(r'\s+(?:lite|sport)$'), "variant"),
    # Roman numerals: " II", " III", " IV" etc.
    (re.compile(r'\s+(?:ii|iii|iv|v|vi)$'), "roman"),
    # Trailing standalone digit: "SoundMax 2", "SoundMax 3"
    (re.compile(r'\s+\d+$'), "trailing_number"),
    # Year suffix in SKU ID (hyphen-separated): "-2024", "-2025"
    (re.compile(r'-20\d{2}$'), "year_suffix"),
    # Generation in SKU ID: "-MkII", "-Gen2", "-v2"
    (re.compile(r'-mk(?:ii|iii|iv|v|vi|\d+)$'), "mk_id"),
    (re.compile(r'-gen\d+$'), "gen_id"),
    (re.compile(r'-v\d+$'), "version_id"),
]


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def _extract_base_and_marker(name: str) -> Tuple[str, Optional[str]]:
    """
    Return (base_name, marker_value) where the generation marker has been
    stripped from the end of ``name``.

    Returns the actual matched marker text (e.g. ``"-mkii"``, ``"-2022"``,
    ``"gen 2"``) so that two SKUs with *different* year suffixes
    (``-2022`` vs ``-2023``) are correctly recognised as a progression
    rather than being considered identical markers.

    If no marker is detected, marker_value is ``None`` and base_name equals
    the normalised input.
    """
    normalized = _normalize(name)
    for pattern, _label in _MARKER_PATTERNS:
        m = pattern.search(normalized)
        if m:
            base = normalized[: m.start()].strip()
            marker_value = m.group(0).strip()
            return base, marker_value
    return normalized, None


# ── Method implementation ─────────────────────────────────────────────────────

class NamingConventionMethod(BaseMethod):
    """Naming-convention-based SKU transition discovery (Method 3)."""

    name = "naming"

    def __init__(
        self,
        min_base_similarity: float = 0.70,
        use_description: bool = True,
        use_sku_id: bool = True,
    ):
        """
        Parameters
        ----------
        min_base_similarity:
            Minimum rapidfuzz ``WRatio`` (0-100 → mapped to 0-1) on the base
            name before the pair is considered.  Default: 0.70.
        use_description:
            Parse the human-readable ``sku_description`` field.
        use_sku_id:
            Parse the ``sku_id`` field.
        """
        self.min_base_similarity = min_base_similarity
        self.use_description = use_description
        self.use_sku_id = use_sku_id

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, product_master: pl.DataFrame) -> List[MappingCandidate]:
        """Discover naming-convention-based candidates."""
        # Add normalised name columns via Polars, then extract base/marker in Python
        enriched = self._add_normalised_columns(product_master)

        old_skus = enriched.filter(
            pl.col("status").is_in(list(PREDECESSOR_STATUSES))
        )
        # Any SKU can be a successor — including Discontinued mid-generation
        # SKUs that are themselves predecessors to a later generation.
        new_skus = enriched

        if old_skus.is_empty() or new_skus.is_empty():
            return []

        # Scope cross-join to same product_family to keep pair count manageable
        pairs = old_skus.join(
            new_skus,
            on="product_family",
            how="inner",
            suffix="_new",
        ).filter(pl.col("launch_date") < pl.col("launch_date_new"))

        if pairs.is_empty():
            return []

        return self._score_pairs(pairs)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _add_normalised_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add Polars-computed normalised name columns."""
        exprs = []

        if self.use_description and "sku_description" in df.columns:
            exprs.append(
                pl.col("sku_description")
                .str.to_lowercase()
                .str.replace_all(r'\s+', ' ')
                .str.strip_chars()
                .alias("_norm_desc")
            )
        else:
            exprs.append(pl.lit(None).cast(pl.Utf8).alias("_norm_desc"))

        if self.use_sku_id and "sku_id" in df.columns:
            exprs.append(
                pl.col("sku_id")
                .str.to_lowercase()
                .str.strip_chars()
                .alias("_norm_id")
            )
        else:
            exprs.append(pl.lit(None).cast(pl.Utf8).alias("_norm_id"))

        return df.with_columns(exprs)

    def _score_pairs(self, pairs: pl.DataFrame) -> List[MappingCandidate]:
        candidates: List[MappingCandidate] = []

        for row in pairs.iter_rows(named=True):
            candidate = self._score_one_pair(row)
            if candidate is not None:
                candidates.append(candidate)

        return candidates

    def _score_one_pair(self, row: dict) -> Optional[MappingCandidate]:
        """
        Compute the naming score for a single (old, new) pair.

        Tries both description and SKU-ID fields and takes the higher score.
        """
        best_score: Optional[float] = None
        best_meta: dict = {}

        for field_suffix, norm_col in [("desc", "_norm_desc"), ("id", "_norm_id")]:
            old_text = row.get(norm_col)
            new_text = row.get(f"{norm_col}_new")

            if not old_text or not new_text:
                continue

            old_base, old_marker = _extract_base_and_marker(old_text)
            new_base, new_marker = _extract_base_and_marker(new_text)

            # Use fuzz.ratio (simple edit-distance ratio) rather than WRatio
            # to avoid token-set scoring inflating similarity between unrelated
            # names that happen to share a numeric token (e.g. "alpha 100" vs
            # "wh 100 mkii" would score 93 with WRatio but ~30 with ratio).
            similarity = fuzz.ratio(old_base, new_base) / 100.0

            if similarity < self.min_base_similarity:
                continue

            # Generation marker detected: old has no marker (base product) and
            # new has a marker, OR both have different markers (increment).
            marker_detected = (
                (old_marker is None and new_marker is not None)
                or (
                    old_marker is not None
                    and new_marker is not None
                    and old_marker != new_marker
                )
            )

            # Both SKUs share product_family because it is the join key —
            # Polars merges join-key columns, so no "_new" suffix exists for it.
            same_family = True

            score = (
                similarity * 0.50
                + (0.30 if marker_detected else 0.0)
                + (0.20 if same_family else 0.0)
            )
            score = min(score, 1.0)

            if best_score is None or score > best_score:
                best_score = score
                best_meta = {
                    "field": field_suffix,
                    "old_base": old_base,
                    "new_base": new_base,
                    "old_marker": old_marker,
                    "new_marker": new_marker,
                    "similarity": round(similarity, 4),
                    "marker_detected": marker_detected,
                }

        if best_score is None:
            return None

        return MappingCandidate(
            old_sku=row["sku_id"],
            new_sku=row["sku_id_new"],
            method=self.name,
            method_score=round(best_score, 4),
            metadata=best_meta,
        )
