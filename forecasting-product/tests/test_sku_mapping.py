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


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Curve-fitting method tests
# ─────────────────────────────────────────────────────────────────────────────

from src.sku_mapping.methods.curve_fitting import CurveFittingMethod
from src.sku_mapping.pipeline import build_phase2_pipeline


def _make_sales_df(
    sku_sales: dict,          # {sku_id: list_of_weekly_sales}
    start: date = date(2023, 1, 2),
) -> pl.DataFrame:
    """Build a weekly sales DataFrame for testing."""
    from datetime import timedelta
    rows = []
    for sku_id, sales in sku_sales.items():
        for i, qty in enumerate(sales):
            rows.append({
                "sku_id": sku_id,
                "week": start + timedelta(weeks=i),
                "quantity": float(qty),
            })
    return pl.DataFrame(rows)


def _curve_pm(old_launch, new_launch) -> pl.DataFrame:
    """Minimal product master with one old and one new SKU."""
    return _minimal_pm(
        {
            "sku_id": "OLD-1", "sku_description": "Widget Gen1",
            "product_family": "Widget", "product_category": "Electronics",
            "form_factor": "Standard", "price_tier": "Mid",
            "country": ["US"], "segment": "Consumer",
            "launch_date": old_launch, "eol_date": new_launch,
            "status": "Discontinued",
        },
        {
            "sku_id": "NEW-1", "sku_description": "Widget Gen2",
            "product_family": "Widget", "product_category": "Electronics",
            "form_factor": "Standard", "price_tier": "Mid",
            "country": ["US"], "segment": "Consumer",
            "launch_date": new_launch, "eol_date": None,
            "status": "Active",
        },
    )


class TestCurveFittingMethod:
    """Phase 2: demand curve-fitting SKU discovery tests."""

    def test_no_sales_data_returns_empty(self):
        pm = _curve_pm(date(2023, 1, 1), date(2023, 7, 1))
        method = CurveFittingMethod(sales_df=None)
        assert method.run(pm) == []

    def test_empty_sales_df_returns_empty(self):
        empty = pl.DataFrame({"sku_id": [], "week": [], "quantity": []}).cast({
            "week": pl.Date, "quantity": pl.Float64
        })
        method = CurveFittingMethod(sales_df=empty)
        assert method.run(pm := _curve_pm(date(2023, 1, 1), date(2023, 7, 1))) == []

    def test_detects_clean_transition(self):
        """
        Old SKU declining + new SKU ramping up → should find a candidate.
        """
        new_launch = date(2023, 7, 3)   # week 26
        # Old SKU: 52 weeks of data — first 26 weeks high, then declining
        old_sales = [100.0] * 26 + [100 - i * 6 for i in range(1, 14)]
        # New SKU: starts at new_launch, ramps up
        new_sales = [i * 8 for i in range(1, 14)]

        sales_df = _make_sales_df({
            "OLD-1": old_sales,
            "NEW-1": [0.0] * 26 + new_sales,
        })

        pm = _curve_pm(date(2023, 1, 2), new_launch)
        method = CurveFittingMethod(sales_df=sales_df, window_weeks=13, min_data_points=4)
        candidates = method.run(pm)

        assert len(candidates) == 1
        c = candidates[0]
        assert c.old_sku == "OLD-1"
        assert c.new_sku == "NEW-1"
        assert c.method == "curve"
        assert 0.0 < c.method_score <= 1.0

    def test_decline_score_positive_for_falling_old_sku(self):
        """Old SKU that strictly declines post-launch should get decline_score > 0."""
        new_launch = date(2023, 7, 3)
        old_sales = [100.0] * 26 + [90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 3, 2, 1]
        new_sales = [0.0] * 26 + [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97, 98, 99]

        sales_df = _make_sales_df({"OLD-1": old_sales, "NEW-1": new_sales})
        pm = _curve_pm(date(2023, 1, 2), new_launch)
        method = CurveFittingMethod(sales_df=sales_df, window_weeks=13)
        candidates = method.run(pm)

        assert len(candidates) == 1
        assert candidates[0].metadata["decline_score"] > 0

    def test_ramp_score_positive_for_growing_new_sku(self):
        """New SKU that strictly grows should get ramp_score > 0."""
        new_launch = date(2023, 7, 3)
        old_sales = [100.0] * 26 + [80, 60, 40, 30, 20, 10, 8, 6, 4, 3, 2, 1, 0]
        new_sales = [0.0] * 26 + [10, 20, 35, 50, 65, 78, 86, 91, 95, 97, 98, 99, 100]

        sales_df = _make_sales_df({"OLD-1": old_sales, "NEW-1": new_sales})
        pm = _curve_pm(date(2023, 1, 2), new_launch)
        method = CurveFittingMethod(sales_df=sales_df, window_weeks=13)
        candidates = method.run(pm)

        assert len(candidates) == 1
        assert candidates[0].metadata["ramp_score"] > 0

    def test_insufficient_post_window_data_returns_no_candidate(self):
        """If old SKU has fewer than min_data_points post-launch, no candidate."""
        new_launch = date(2023, 7, 3)
        # Only 2 post-launch data points for old SKU
        old_sales = [100.0] * 26 + [80.0, 60.0]
        new_sales = [0.0] * 26 + [20.0, 40.0]

        sales_df = _make_sales_df({"OLD-1": old_sales, "NEW-1": new_sales})
        pm = _curve_pm(date(2023, 1, 2), new_launch)
        method = CurveFittingMethod(sales_df=sales_df, min_data_points=4)
        # With only 2 points and min=4, should return nothing
        candidates = method.run(pm)
        assert candidates == []

    def test_missing_sku_in_sales_skipped(self):
        """If one of the SKUs is not in the sales data, the pair is skipped."""
        new_launch = date(2023, 7, 3)
        old_sales = [100.0] * 39
        # NEW-1 not in sales_df at all
        sales_df = _make_sales_df({"OLD-1": old_sales})
        pm = _curve_pm(date(2023, 1, 2), new_launch)
        method = CurveFittingMethod(sales_df=sales_df)
        assert method.run(pm) == []

    def test_method_name_is_curve(self):
        method = CurveFittingMethod()
        assert method.name == "curve"

    def test_metadata_keys_present(self):
        """All expected metadata keys are present in the candidate."""
        new_launch = date(2023, 7, 3)
        old_sales = [100.0] * 26 + [90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 3, 2, 1]
        new_sales = [0.0] * 26 + [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97, 98, 99]
        sales_df = _make_sales_df({"OLD-1": old_sales, "NEW-1": new_sales})
        pm = _curve_pm(date(2023, 1, 2), new_launch)
        method = CurveFittingMethod(sales_df=sales_df, window_weeks=13)
        candidates = method.run(pm)
        assert len(candidates) == 1
        meta = candidates[0].metadata
        for key in ("decline_score", "ramp_score", "complement_score", "scale_score",
                    "old_pre_weeks", "old_post_weeks", "new_post_weeks"):
            assert key in meta, f"Missing metadata key: {key}"

    def test_flat_old_sku_gets_low_decline_score(self):
        """An old SKU with flat sales should not get a high decline score."""
        new_launch = date(2023, 7, 3)
        old_sales = [100.0] * 39   # flat throughout
        new_sales = [0.0] * 26 + [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97, 98, 99]
        sales_df = _make_sales_df({"OLD-1": old_sales, "NEW-1": new_sales})
        pm = _curve_pm(date(2023, 1, 2), new_launch)
        method = CurveFittingMethod(sales_df=sales_df, window_weeks=13)
        candidates = method.run(pm)
        if candidates:
            assert candidates[0].metadata["decline_score"] < 0.3


class TestPhase2Pipeline:
    """Phase 2 pipeline smoke tests (curve-fitting integrated)."""

    def test_build_phase2_pipeline_no_sales(self):
        """Phase 2 pipeline without sales data behaves like Phase 1."""
        pipeline = build_phase2_pipeline(sales_df=None, min_confidence="Low")
        pm = generate_product_master()
        df = pipeline.run(pm)
        assert isinstance(df, pl.DataFrame)

    def test_build_phase2_pipeline_with_sales(self):
        """Phase 2 pipeline with sales data runs end-to-end without error."""
        pm = generate_product_master()

        # Build a simple sales dataset for all SKUs
        from datetime import timedelta
        import random
        random.seed(0)
        rows = []
        start = date(2022, 1, 3)
        for sku_row in pm.iter_rows(named=True):
            for w in range(52):
                rows.append({
                    "sku_id": sku_row["sku_id"],
                    "week": start + timedelta(weeks=w),
                    "quantity": float(random.randint(10, 200)),
                })
        sales_df = pl.DataFrame(rows)

        pipeline = build_phase2_pipeline(sales_df=sales_df, min_confidence="Low")
        df = pipeline.run(pm)
        assert isinstance(df, pl.DataFrame)

    def test_phase2_pipeline_has_four_methods(self):
        pipeline = build_phase2_pipeline()
        assert len(pipeline.methods) == 4
        names = {m.name for m in pipeline.methods}
        assert names == {"attribute", "naming", "curve", "temporal"}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Temporal co-movement method tests
# ─────────────────────────────────────────────────────────────────────────────

from src.sku_mapping.methods.temporal_comovement import TemporalCovementMethod


class TestTemporalCovementMethod:
    """Phase 2: temporal co-movement SKU discovery tests."""

    def test_no_sales_returns_empty(self):
        pm = _curve_pm(date(2023, 1, 1), date(2023, 7, 1))
        assert TemporalCovementMethod(sales_df=None).run(pm) == []

    def test_method_name_is_temporal(self):
        assert TemporalCovementMethod().name == "temporal"

    def test_detects_negatively_correlated_pair(self):
        """
        Old SKU declining while new SKU grows → negative raw correlation →
        high correlation_score → candidate found.
        """
        new_launch = date(2023, 7, 3)
        # Old SKU: 26 weeks high then strictly declining
        old_s = [100.0] * 26 + [100 - i * 7 for i in range(1, 14)]
        # New SKU: starts at 0, strictly increasing (anti-correlated with old)
        new_s = [0.0] * 26 + [i * 7 for i in range(1, 14)]

        sales_df = _make_sales_df({"OLD-1": old_s, "NEW-1": new_s})
        pm = _curve_pm(date(2023, 1, 2), new_launch)
        method = TemporalCovementMethod(sales_df=sales_df, window_weeks=13, min_data_points=4)
        candidates = method.run(pm)

        assert len(candidates) == 1
        c = candidates[0]
        assert c.old_sku == "OLD-1"
        assert c.new_sku == "NEW-1"
        assert c.method == "temporal"
        assert 0.0 < c.method_score <= 1.0

    def test_correlation_score_high_for_anti_correlated_pair(self):
        new_launch = date(2023, 7, 3)
        old_s = [100.0] * 26 + [90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 3, 2, 1]
        new_s = [0.0]   * 26 + [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97, 98, 99]

        sales_df = _make_sales_df({"OLD-1": old_s, "NEW-1": new_s})
        pm = _curve_pm(date(2023, 1, 2), new_launch)
        method = TemporalCovementMethod(sales_df=sales_df, window_weeks=13)
        candidates = method.run(pm)

        assert len(candidates) == 1
        assert candidates[0].metadata["correlation_score"] > 0.5

    def test_volume_match_high_for_similar_volumes(self):
        """When old pre-launch avg ≈ new post-launch avg, volume_match_score → 1."""
        new_launch = date(2023, 7, 3)
        # Both at ~100 units
        old_s = [100.0] * 26 + [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35]
        new_s = [0.0]   * 26 + [30, 45, 60, 70, 80, 90, 95, 100, 103, 105, 106, 107, 108]

        sales_df = _make_sales_df({"OLD-1": old_s, "NEW-1": new_s})
        pm = _curve_pm(date(2023, 1, 2), new_launch)
        method = TemporalCovementMethod(sales_df=sales_df, window_weeks=13)
        candidates = method.run(pm)

        assert len(candidates) == 1
        assert candidates[0].metadata["volume_match_score"] > 0.5

    def test_missing_sku_in_sales_skipped(self):
        old_s = [100.0] * 39
        sales_df = _make_sales_df({"OLD-1": old_s})  # NEW-1 missing
        pm = _curve_pm(date(2023, 1, 2), date(2023, 7, 3))
        method = TemporalCovementMethod(sales_df=sales_df)
        assert method.run(pm) == []

    def test_insufficient_overlap_returns_no_candidate(self):
        """Fewer than min_data_points in post-window → skip."""
        new_launch = date(2023, 7, 3)
        old_s = [100.0] * 26 + [80.0, 60.0]   # only 2 post-launch points
        new_s = [0.0]   * 26 + [20.0, 40.0]
        sales_df = _make_sales_df({"OLD-1": old_s, "NEW-1": new_s})
        pm = _curve_pm(date(2023, 1, 2), new_launch)
        method = TemporalCovementMethod(sales_df=sales_df, min_data_points=4)
        assert method.run(pm) == []

    def test_metadata_keys_present(self):
        new_launch = date(2023, 7, 3)
        old_s = [100.0] * 26 + [90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 3, 2, 1]
        new_s = [0.0]   * 26 + [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97, 98, 99]
        sales_df = _make_sales_df({"OLD-1": old_s, "NEW-1": new_s})
        pm = _curve_pm(date(2023, 1, 2), new_launch)
        method = TemporalCovementMethod(sales_df=sales_df, window_weeks=13)
        candidates = method.run(pm)
        assert len(candidates) == 1
        meta = candidates[0].metadata
        for key in ("correlation_score", "overlap_score", "volume_match_score",
                    "n_overlap_points", "old_pre_avg", "new_post_avg"):
            assert key in meta, f"Missing metadata key: {key}"

    def test_phase2_pipeline_has_four_methods(self):
        pipeline = build_phase2_pipeline()
        assert len(pipeline.methods) == 4
        names = {m.name for m in pipeline.methods}
        assert names == {"attribute", "naming", "curve", "temporal"}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Bayesian proportion estimation tests
# ─────────────────────────────────────────────────────────────────────────────

from src.sku_mapping.fusion.bayesian_proportions import BayesianProportionEstimator
from src.sku_mapping.data.schemas import MappingRecord


def _make_record(old_sku, new_sku, mapping_type, confidence_score=0.5, proportion=None):
    return MappingRecord(
        mapping_id="MAP-TEST",
        old_sku=old_sku,
        new_sku=new_sku,
        mapping_type=mapping_type,
        proportion=proportion if proportion is not None else (1.0 if mapping_type == "1-to-1" else 0.5),
        confidence_score=confidence_score,
        confidence_level="Medium",
        methods_matched=["attribute"],
        transition_start_week=None,
        transition_end_week=None,
        old_sku_lifecycle_stage="Discontinued",
    )


class TestBayesianProportionEstimator:
    """Phase 2: Bayesian proportion estimation tests."""

    def test_one_to_one_stays_1(self):
        estimator = BayesianProportionEstimator()
        r = _make_record("OLD", "NEW", "1-to-1", confidence_score=0.8)
        result = estimator.estimate([r])
        assert result[0].proportion == 1.0

    def test_one_to_many_sums_to_one(self):
        """1-to-Many: proportions across all successors must sum to 1.0."""
        estimator = BayesianProportionEstimator(concentration=0.5)
        records = [
            _make_record("OLD", "NEW-A", "1-to-Many", confidence_score=0.9),
            _make_record("OLD", "NEW-B", "1-to-Many", confidence_score=0.3),
            _make_record("OLD", "NEW-C", "1-to-Many", confidence_score=0.6),
        ]
        estimator.estimate(records)
        total = sum(r.proportion for r in records)
        assert abs(total - 1.0) < 1e-3

    def test_one_to_many_high_score_gets_more(self):
        """High-confidence pair should get a larger proportion than low-confidence."""
        estimator = BayesianProportionEstimator(concentration=0.1)  # low alpha → score-dominated
        records = [
            _make_record("OLD", "NEW-A", "1-to-Many", confidence_score=0.9),
            _make_record("OLD", "NEW-B", "1-to-Many", confidence_score=0.1),
        ]
        estimator.estimate(records)
        p_a = next(r.proportion for r in records if r.new_sku == "NEW-A")
        p_b = next(r.proportion for r in records if r.new_sku == "NEW-B")
        assert p_a > p_b

    def test_equal_scores_give_equal_proportions(self):
        """When all confidence scores are equal, proportions should be ~1/n."""
        estimator = BayesianProportionEstimator(concentration=0.5)
        records = [
            _make_record("OLD", "NEW-A", "1-to-Many", confidence_score=0.6),
            _make_record("OLD", "NEW-B", "1-to-Many", confidence_score=0.6),
            _make_record("OLD", "NEW-C", "1-to-Many", confidence_score=0.6),
        ]
        estimator.estimate(records)
        for r in records:
            assert abs(r.proportion - 1/3) < 0.01

    def test_many_to_one_sums_to_one_per_new_sku(self):
        """Many-to-1: proportions for all predecessors of same new_sku sum to 1.0."""
        estimator = BayesianProportionEstimator()
        records = [
            _make_record("OLD-A", "NEW", "Many-to-1", confidence_score=0.8),
            _make_record("OLD-B", "NEW", "Many-to-1", confidence_score=0.4),
        ]
        estimator.estimate(records)
        total = sum(r.proportion for r in records)
        assert abs(total - 1.0) < 1e-3

    def test_high_alpha_approaches_equal_split(self):
        """Very high concentration → proportions approach equal split."""
        estimator = BayesianProportionEstimator(concentration=1000.0)
        records = [
            _make_record("OLD", "NEW-A", "1-to-Many", confidence_score=0.9),
            _make_record("OLD", "NEW-B", "1-to-Many", confidence_score=0.1),
        ]
        estimator.estimate(records)
        p_a = next(r.proportion for r in records if r.new_sku == "NEW-A")
        p_b = next(r.proportion for r in records if r.new_sku == "NEW-B")
        # With very high alpha, both should be close to 0.5
        assert abs(p_a - 0.5) < 0.05
        assert abs(p_b - 0.5) < 0.05

    def test_negative_concentration_raises(self):
        with pytest.raises(ValueError):
            BayesianProportionEstimator(concentration=-0.1)

    def test_empty_records_returns_empty(self):
        estimator = BayesianProportionEstimator()
        assert estimator.estimate([]) == []

    def test_notes_updated_for_multi_mapped(self):
        """Notes should mention Bayesian proportion after estimation."""
        estimator = BayesianProportionEstimator(concentration=0.5)
        records = [
            _make_record("OLD", "NEW-A", "1-to-Many", confidence_score=0.8),
            _make_record("OLD", "NEW-B", "1-to-Many", confidence_score=0.5),
        ]
        estimator.estimate(records)
        for r in records:
            assert "Bayesian" in (r.notes or ""), f"Expected Bayesian in notes: {r.notes}"

    def test_many_to_many_proportions_sum_to_one_per_old(self):
        """For Many-to-Many, proportions per old_sku should sum to 1.0."""
        estimator = BayesianProportionEstimator(concentration=0.5)
        records = [
            _make_record("OLD-A", "NEW-1", "Many-to-Many", confidence_score=0.8),
            _make_record("OLD-A", "NEW-2", "Many-to-Many", confidence_score=0.4),
            _make_record("OLD-B", "NEW-1", "Many-to-Many", confidence_score=0.6),
            _make_record("OLD-B", "NEW-2", "Many-to-Many", confidence_score=0.7),
        ]
        estimator.estimate(records)
        # OLD-A's two successors should sum to 1.0
        old_a = [r.proportion for r in records if r.old_sku == "OLD-A"]
        assert abs(sum(old_a) - 1.0) < 1e-3
        # OLD-B's two successors should sum to 1.0
        old_b = [r.proportion for r in records if r.old_sku == "OLD-B"]
        assert abs(sum(old_b) - 1.0) < 1e-3

    def test_fusion_with_bayesian_enabled(self):
        """CandidateFusion with bayesian_proportions=True runs end-to-end."""
        pm = generate_product_master()
        pipeline = build_phase2_pipeline(min_confidence="Low")
        df = pipeline.run(pm)
        assert isinstance(df, pl.DataFrame)
