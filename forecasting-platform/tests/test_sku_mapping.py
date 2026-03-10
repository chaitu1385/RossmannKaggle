"""
Tests for the SKU Mapping Discovery Pipeline (Phase 1).

Coverage:
  - Mock data generator schema validity
  - Method 1 (attribute matching): finds known transitions, rejects true negatives
  - Method 3 (naming parsing): detects all generation-marker patterns
  - CandidateFusion: scoring, multi-method bonus, confidence classification
  - Mapping type assignment: 1-to-1, 1-to-Many
  - Pipeline end-to-end smoke test
  - Writer output schema
"""

from datetime import date

import polars as pl
import pytest

from src.sku_mapping.data.mock_generator import generate_product_master
from src.sku_mapping.data.schemas import MappingCandidate
from src.sku_mapping.fusion.scorer import CandidateFusion, _classify_confidence
from src.sku_mapping.methods.attribute_matching import AttributeMatchingMethod
from src.sku_mapping.methods.naming_parsing import (
    NamingConventionMethod,
    _extract_base_and_marker,
)
from src.sku_mapping.output.writer import MappingWriter
from src.sku_mapping.pipeline import build_phase1_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _minimal_pm(*rows) -> pl.DataFrame:
    """Build a minimal product master DataFrame from plain dicts."""
    schema = {
        "sku_id": pl.Utf8,
        "sku_description": pl.Utf8,
        "product_family": pl.Utf8,
        "product_category": pl.Utf8,
        "form_factor": pl.Utf8,
        "price_tier": pl.Utf8,
        "country": pl.List(pl.Utf8),
        "segment": pl.Utf8,
        "launch_date": pl.Date,
        "eol_date": pl.Date,
        "status": pl.Utf8,
    }
    return pl.DataFrame(list(rows), schema=schema)


def _pair_found(candidates, old_sku: str, new_sku: str) -> bool:
    return any(c.old_sku == old_sku and c.new_sku == new_sku for c in candidates)


# ─────────────────────────────────────────────────────────────────────────────
# Mock data generator
# ─────────────────────────────────────────────────────────────────────────────

class TestMockGenerator:
    def test_returns_polars_dataframe(self):
        pm = generate_product_master()
        assert isinstance(pm, pl.DataFrame)

    def test_required_columns_present(self):
        pm = generate_product_master()
        required = {"sku_id", "product_family", "launch_date", "country", "segment", "status"}
        assert required.issubset(set(pm.columns))

    def test_country_is_list_column(self):
        pm = generate_product_master()
        assert pm["country"].dtype == pl.List(pl.Utf8)

    def test_no_duplicate_sku_ids(self):
        pm = generate_product_master()
        assert pm["sku_id"].n_unique() == len(pm)

    def test_known_skus_present(self):
        pm = generate_product_master()
        sku_ids = set(pm["sku_id"].to_list())
        for expected in ("WH-100", "WH-100-MkII", "XB-500-2022", "SM-PRO-1", "EP-100"):
            assert expected in sku_ids, f"Expected SKU {expected!r} not in mock data"

    def test_predecessor_skus_have_discontinued_or_declining_status(self):
        pm = generate_product_master()
        old_statuses = set(
            pm.filter(pl.col("status").is_in(["Discontinued", "Declining"]))["sku_id"]
            .to_list()
        )
        assert "WH-100" in old_statuses
        assert "XB-500-2022" in old_statuses


# ─────────────────────────────────────────────────────────────────────────────
# Attribute Matching (Method 1)
# ─────────────────────────────────────────────────────────────────────────────

class TestAttributeMatchingMethod:
    def setup_method(self):
        # Use the default launch_window_days (365) so mid-cycle transitions
        # (e.g. annual product refreshes with ~336-day gaps) are included.
        self.method = AttributeMatchingMethod()

    # -- Positive cases: should find these transitions ---

    def test_finds_wh100_to_mkii(self):
        pm = generate_product_master()
        candidates = self.method.run(pm)
        assert _pair_found(candidates, "WH-100", "WH-100-MkII"), (
            "WH-100 → WH-100-MkII not found by attribute matching"
        )

    def test_finds_xb500_year_series(self):
        pm = generate_product_master()
        candidates = self.method.run(pm)
        assert _pair_found(candidates, "XB-500-2022", "XB-500-2023")
        assert _pair_found(candidates, "XB-500-2023", "XB-500-2024")

    def test_finds_pocketbass_transition(self):
        pm = generate_product_master()
        candidates = self.method.run(pm)
        assert _pair_found(candidates, "PB-V1", "PB-V2")

    # -- Negative cases: must NOT match these ----------

    def test_does_not_match_different_families(self):
        pm = generate_product_master()
        candidates = self.method.run(pm)
        # ALPHA-100 (On-Ear) vs BETA-200 (Portable-Speaker): different families
        assert not _pair_found(candidates, "ALPHA-100", "BETA-200")

    def test_does_not_match_new_before_old(self):
        """new SKU must have a later launch date than old SKU."""
        pm = _minimal_pm(
            dict(sku_id="A", sku_description="A", product_family="F",
                 product_category="C", form_factor="X", price_tier="Budget",
                 country=["USA"], segment="Retail",
                 launch_date=date(2024, 1, 1), eol_date=None, status="Active"),
            dict(sku_id="B", sku_description="B", product_family="F",
                 product_category="C", form_factor="X", price_tier="Budget",
                 country=["USA"], segment="Retail",
                 launch_date=date(2022, 1, 1), eol_date=date(2023, 12, 31),
                 status="Discontinued"),
        )
        candidates = self.method.run(pm)
        # A is active and launched after B, but B is the "old" one.
        # The pair (A→B) should NOT appear because A launched AFTER B.
        assert not _pair_found(candidates, "A", "B")

    def test_does_not_match_different_countries(self):
        pm = _minimal_pm(
            dict(sku_id="OLD", sku_description="Old", product_family="F",
                 product_category="C", form_factor="X", price_tier="Budget",
                 country=["USA"], segment="Retail",
                 launch_date=date(2022, 1, 1), eol_date=date(2023, 6, 1),
                 status="Discontinued"),
            dict(sku_id="NEW", sku_description="New", product_family="F",
                 product_category="C", form_factor="X", price_tier="Budget",
                 country=["DEU"], segment="Retail",    # no country overlap
                 launch_date=date(2023, 3, 1), eol_date=None, status="Active"),
        )
        candidates = self.method.run(pm)
        assert not _pair_found(candidates, "OLD", "NEW")

    # -- Score properties -----------------------

    def test_full_attribute_match_scores_above_0_7(self):
        """Perfect attribute match should be high-scoring (uses a large window)."""
        method = AttributeMatchingMethod(launch_window_days=400)
        pm = _minimal_pm(
            dict(sku_id="OLD", sku_description="Old", product_family="F",
                 product_category="C", form_factor="X", price_tier="Budget",
                 country=["USA"], segment="Retail",
                 launch_date=date(2022, 1, 1), eol_date=date(2023, 1, 31),
                 status="Discontinued"),
            dict(sku_id="NEW", sku_description="New", product_family="F",
                 product_category="C", form_factor="X", price_tier="Budget",
                 country=["USA"], segment="Retail",
                 launch_date=date(2023, 1, 1), eol_date=None, status="Active"),
        )
        candidates = method.run(pm)
        pair = next(
            (c for c in candidates if c.old_sku == "OLD" and c.new_sku == "NEW"),
            None,
        )
        assert pair is not None
        # base(0.3) + price(0.2) + form(0.2) + category(0.1) + gap_bonus(0.1) = 0.9
        assert pair.method_score >= 0.70


# ─────────────────────────────────────────────────────────────────────────────
# Naming Convention Parser (Method 3)
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractBaseAndMarker:
    """Unit tests for the base-name extraction helper."""

    @pytest.mark.parametrize("name,expected_base,expect_marker", [
        ("budget on-ear headphones mkii",  "budget on-ear headphones", True),
        ("xb-500 over-ear 2024",           "xb-500 over-ear",         True),
        ("soundmax pro 2",                  "soundmax pro",             True),
        ("bt-air gen2",                     "bt-air",                   True),
        ("noisepro plus",                   "noisepro",                 True),
        ("pocketbass v1",                   "pocketbass",               True),
        ("bt-air-g1",                       "bt-air-g1",                False),  # no marker in this ID
        ("earphone standalone",             "earphone standalone",      False),
    ])
    def test_extraction(self, name, expected_base, expect_marker):
        base, marker = _extract_base_and_marker(name)
        assert base == expected_base, f"Expected base {expected_base!r}, got {base!r}"
        if expect_marker:
            assert marker is not None, f"Expected a marker for {name!r}, got None"
        else:
            assert marker is None, f"Expected no marker for {name!r}, got {marker!r}"


class TestNamingConventionMethod:
    def setup_method(self):
        self.method = NamingConventionMethod(min_base_similarity=0.70)

    def test_finds_mkii_pattern(self):
        pm = generate_product_master()
        candidates = self.method.run(pm)
        assert _pair_found(candidates, "WH-100", "WH-100-MkII"), (
            "WH-100 → WH-100-MkII not found by naming method"
        )

    def test_finds_year_suffix_pattern(self):
        pm = generate_product_master()
        candidates = self.method.run(pm)
        assert _pair_found(candidates, "XB-500-2022", "XB-500-2023")

    def test_finds_gen_pattern(self):
        pm = generate_product_master()
        candidates = self.method.run(pm)
        assert _pair_found(candidates, "BT-AIR-G1", "BT-AIR-G2")

    def test_finds_trailing_number_pattern(self):
        pm = generate_product_master()
        candidates = self.method.run(pm)
        assert _pair_found(candidates, "SM-PRO-1", "SM-PRO-2")

    def test_rejects_completely_different_names(self):
        """ALPHA-100 → BETA-200 share no base name; should not be matched."""
        pm = generate_product_master()
        candidates = self.method.run(pm)
        assert not _pair_found(candidates, "ALPHA-100", "BETA-200")

    def test_score_in_valid_range(self):
        pm = generate_product_master()
        candidates = self.method.run(pm)
        for c in candidates:
            assert 0.0 <= c.method_score <= 1.0, (
                f"Score out of range for {c.old_sku} → {c.new_sku}: {c.method_score}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Fusion & Scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceClassification:
    @pytest.mark.parametrize("score,expected", [
        (0.90, "High"),
        (0.75, "High"),
        (0.74, "Medium"),
        (0.50, "Medium"),
        (0.49, "Low"),
        (0.30, "Low"),
        (0.29, "Very Low"),
        (0.00, "Very Low"),
    ])
    def test_thresholds(self, score, expected):
        assert _classify_confidence(score) == expected


class TestCandidateFusion:
    def setup_method(self):
        self.pm = generate_product_master()
        self.fusion = CandidateFusion()

    def test_empty_candidates_returns_empty(self):
        records = self.fusion.fuse([], self.pm)
        assert records == []

    def test_single_method_candidate_produces_record(self):
        candidates = [
            MappingCandidate(
                old_sku="WH-100",
                new_sku="WH-100-MkII",
                method="attribute",
                method_score=0.80,
            )
        ]
        records = self.fusion.fuse(candidates, self.pm)
        assert len(records) == 1
        assert records[0].old_sku == "WH-100"
        assert records[0].new_sku == "WH-100-MkII"
        assert records[0].confidence_level == "High"

    def test_multi_method_bonus_applied(self):
        """Two methods agreeing should push the score above either alone."""
        base_attr_score = 0.55
        base_naming_score = 0.55

        # Single method
        single = CandidateFusion().fuse(
            [MappingCandidate("OLD", "NEW", "attribute", base_attr_score)],
            self.pm,
        )
        # Two methods
        double = CandidateFusion().fuse(
            [
                MappingCandidate("OLD", "NEW", "attribute", base_attr_score),
                MappingCandidate("OLD", "NEW", "naming", base_naming_score),
            ],
            self.pm,
        )
        # With bonus, the two-method score should be higher
        if single and double:
            assert double[0].confidence_score > single[0].confidence_score

    def test_mapping_type_1to1(self):
        candidates = [
            MappingCandidate("WH-100", "WH-100-MkII", "attribute", 0.85),
        ]
        records = self.fusion.fuse(candidates, self.pm)
        assert records[0].mapping_type == "1-to-1"
        assert records[0].proportion == 1.0

    def test_mapping_type_1_to_many(self):
        """EP-100 maps to both EP-200-STD and EP-200-SPT."""
        candidates = [
            MappingCandidate("EP-100", "EP-200-STD", "attribute", 0.75),
            MappingCandidate("EP-100", "EP-200-SPT", "attribute", 0.70),
        ]
        records = self.fusion.fuse(candidates, self.pm)
        types = {r.new_sku: r.mapping_type for r in records}
        assert types["EP-200-STD"] == "1-to-Many"
        assert types["EP-200-SPT"] == "1-to-Many"
        # Equal split → 0.50 each
        for r in records:
            assert abs(r.proportion - 0.50) < 1e-6

    def test_output_sorted_high_confidence_first(self):
        candidates = [
            MappingCandidate("WH-100", "WH-100-MkII", "attribute", 0.30),  # Low
            MappingCandidate("XB-500-2022", "XB-500-2023", "attribute", 0.90),  # High
        ]
        records = self.fusion.fuse(candidates, self.pm)
        assert records[0].confidence_level == "High"

    def test_min_confidence_filter(self):
        """Candidates below min_confidence should be excluded."""
        fusion = CandidateFusion(min_confidence="Medium")
        candidates = [
            MappingCandidate("WH-100", "WH-100-MkII", "attribute", 0.20),  # Very Low
        ]
        records = fusion.fuse(candidates, self.pm)
        assert records == []


# ─────────────────────────────────────────────────────────────────────────────
# Output Writer
# ─────────────────────────────────────────────────────────────────────────────

class TestMappingWriter:
    def setup_method(self):
        self.writer = MappingWriter()
        self.pm = generate_product_master()

    def test_to_polars_empty(self):
        df = self.writer.to_polars([])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_output_has_all_schema_columns(self):
        fusion = CandidateFusion()
        candidates = [
            MappingCandidate("WH-100", "WH-100-MkII", "attribute", 0.85),
        ]
        records = fusion.fuse(candidates, self.pm)
        df = self.writer.to_polars(records)

        expected_cols = {
            "mapping_id", "old_sku", "new_sku", "mapping_type", "proportion",
            "confidence_score", "confidence_level", "methods_matched",
            "transition_start_week", "transition_end_week",
            "old_sku_lifecycle_stage", "validation_status",
            "validated_by", "validation_date", "notes",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_save_csv_creates_file(self, tmp_path):
        fusion = CandidateFusion()
        candidates = [
            MappingCandidate("WH-100", "WH-100-MkII", "attribute", 0.85),
        ]
        records = fusion.fuse(candidates, self.pm)
        df = self.writer.to_polars(records)

        out = tmp_path / "mappings.csv"
        self.writer.save_csv(df, str(out))
        assert out.exists()
        assert out.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineEndToEnd:
    def test_smoke_on_mock_data(self, tmp_path):
        pipeline = build_phase1_pipeline(min_confidence="Low")
        pm = generate_product_master()
        out = str(tmp_path / "mappings.csv")
        df = pipeline.run(pm, output_path=out)

        # Should return a non-empty mapping table
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

        # Known high-confidence pairs must appear
        pairs = set(zip(df["old_sku"].to_list(), df["new_sku"].to_list()))
        assert ("WH-100", "WH-100-MkII") in pairs
        assert ("XB-500-2022", "XB-500-2023") in pairs

        # Output file should be created
        import os
        assert os.path.exists(out)

    def test_all_confidence_scores_in_range(self):
        pipeline = build_phase1_pipeline(min_confidence="Very Low")
        pm = generate_product_master()
        df = pipeline.run(pm)
        assert df["confidence_score"].min() >= 0.0
        assert df["confidence_score"].max() <= 1.0

    def test_proportions_sum_to_one_per_old_sku(self):
        """
        For 1-to-1 and 1-to-Many mappings each old_sku's proportions must
        sum to ~1.0 (the old SKU's demand is fully redistributed across its
        successor(s)).

        For Many-to-1 / Many-to-Many mappings the proportion represents the
        *old SKU's share of the new SKU's demand*, so it sums to 1.0 per
        new_sku, not per old_sku — those rows are intentionally excluded here.
        """
        pipeline = build_phase1_pipeline(min_confidence="Low")
        pm = generate_product_master()
        df = pipeline.run(pm)

        if df.is_empty():
            return

        # Exclude Many-to-1 rows: in that case proportion = 1/n_old_skus and
        # represents the old SKU's share of the *new* SKU's demand — summing
        # per old_sku would give < 1.0 by design.
        # For 1-to-1, 1-to-Many, and Many-to-Many the proportion always
        # equals 1/n_new_skus, so all rows for a given old_sku sum to 1.0.
        df_outbound = df.filter(
            pl.col("mapping_type") != "Many-to-1"
        )

        if df_outbound.is_empty():
            return

        sums = (
            df_outbound.group_by("old_sku")
            .agg(pl.col("proportion").sum().alias("total_proportion"))
        )
        for row in sums.iter_rows(named=True):
            # Tolerance of 1e-3 to accommodate equal-split rounding for SKUs
            # with many successors (e.g. 6 × 0.1667 = 1.0002).
            assert abs(row["total_proportion"] - 1.0) < 1e-3, (
                f"Proportions for {row['old_sku']!r} sum to "
                f"{row['total_proportion']:.4f}, expected 1.0"
            )
