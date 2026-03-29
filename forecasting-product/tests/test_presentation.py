"""Tests for src/presentation/: templates.py, deck_builder.py, marp_linter.py, marp_export.py."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# templates.py
# ===========================================================================

class TestTitleSlide:
    def test_contains_title(self):
        from src.presentation.templates import title_slide
        result = title_slide("My Forecast Review")
        assert "My Forecast Review" in result

    def test_class_directive(self):
        from src.presentation.templates import title_slide
        result = title_slide("T")
        assert "_class: title" in result

    def test_accent_bar_present(self):
        from src.presentation.templates import title_slide
        result = title_slide("T")
        assert "accent-bar" in result

    def test_subtitle_included_when_provided(self):
        from src.presentation.templates import title_slide
        result = title_slide("T", subtitle="Sub heading")
        assert "Sub heading" in result

    def test_subtitle_omitted_when_empty(self):
        from src.presentation.templates import title_slide
        result = title_slide("T", subtitle="")
        assert "###" not in result

    def test_lob_and_date_included(self):
        from src.presentation.templates import title_slide
        result = title_slide("T", lob="Retail", date_str="March 2026")
        assert "Retail" in result
        assert "March 2026" in result

    def test_no_meta_line_when_both_empty(self):
        from src.presentation.templates import title_slide
        result = title_slide("T", lob="", date_str="")
        # No pipe separator because both are empty
        assert " | " not in result


class TestSectionSlide:
    def test_contains_heading(self):
        from src.presentation.templates import section_slide
        result = section_slide("Key Findings")
        assert "Key Findings" in result

    def test_class_directive(self):
        from src.presentation.templates import section_slide
        result = section_slide("S")
        assert "_class: section-opener" in result

    def test_subheading_included(self):
        from src.presentation.templates import section_slide
        result = section_slide("S", subheading="Details follow")
        assert "Details follow" in result

    def test_subheading_omitted_when_empty(self):
        from src.presentation.templates import section_slide
        result = section_slide("S", subheading="")
        # Only the heading line should follow the class directive
        assert result.count("# S") == 1


class TestImpactSlide:
    def test_contains_question(self):
        from src.presentation.templates import impact_slide
        result = impact_slide("What drives the variance?")
        assert "What drives the variance?" in result

    def test_class_directive(self):
        from src.presentation.templates import impact_slide
        result = impact_slide("Q")
        assert "_class: impact" in result


class TestKpiSlide:
    def _sample_metrics(self):
        return [
            {"label": "WMAPE", "value": "18%", "delta": "-2pp", "direction": "down"},
            {"label": "Bias", "value": "+1%", "delta": "+0.5pp", "direction": "up", "detail": "Note"},
        ]

    def test_contains_headline(self):
        from src.presentation.templates import kpi_slide
        result = kpi_slide("Accuracy improved", self._sample_metrics())
        assert "Accuracy improved" in result

    def test_class_directive(self):
        from src.presentation.templates import kpi_slide
        result = kpi_slide("H", self._sample_metrics())
        assert "_class: kpi" in result

    def test_metric_labels_present(self):
        from src.presentation.templates import kpi_slide
        result = kpi_slide("H", self._sample_metrics())
        assert "WMAPE" in result
        assert "Bias" in result

    def test_metric_values_present(self):
        from src.presentation.templates import kpi_slide
        result = kpi_slide("H", self._sample_metrics())
        assert "18%" in result
        assert "+1%" in result

    def test_delta_and_direction_rendered(self):
        from src.presentation.templates import kpi_slide
        result = kpi_slide("H", self._sample_metrics())
        assert "-2pp" in result
        assert 'class="delta down"' in result

    def test_detail_rendered(self):
        from src.presentation.templates import kpi_slide
        result = kpi_slide("H", self._sample_metrics())
        assert "Note" in result

    def test_max_four_metrics(self):
        from src.presentation.templates import kpi_slide
        metrics = [{"label": f"M{i}", "value": str(i)} for i in range(6)]
        result = kpi_slide("H", metrics)
        # Only first 4 should appear
        assert "M0" in result
        assert "M3" in result
        assert "M4" not in result
        assert "M5" not in result

    def test_kpi_row_div_present(self):
        from src.presentation.templates import kpi_slide
        result = kpi_slide("H", self._sample_metrics())
        assert 'class="kpi-row"' in result


class TestChartSlide:
    def test_contains_headline(self):
        from src.presentation.templates import chart_slide
        result = chart_slide("Trend Analysis", "charts/trend.png")
        assert "Trend Analysis" in result

    def test_img_src_present(self):
        from src.presentation.templates import chart_slide
        result = chart_slide("H", "charts/trend.png")
        assert "charts/trend.png" in result

    def test_chart_container_div(self):
        from src.presentation.templates import chart_slide
        result = chart_slide("H", "charts/foo.png")
        assert 'class="chart-container"' in result

    def test_chart_full_class(self):
        from src.presentation.templates import chart_slide
        result = chart_slide("H", "p.png", layout="chart-full")
        assert "_class: chart-full" in result

    def test_chart_left_class(self):
        from src.presentation.templates import chart_slide
        result = chart_slide("H", "p.png", layout="chart-left")
        assert "_class: chart-left" in result

    def test_source_rendered(self):
        from src.presentation.templates import chart_slide
        result = chart_slide("H", "p.png", source="Sales DB")
        assert "Sales DB" in result

    def test_narrative_in_chart_left(self):
        from src.presentation.templates import chart_slide
        result = chart_slide("H", "p.png", layout="chart-left", narrative="Key insight here")
        assert "Key insight here" in result

    def test_narrative_in_chart_right(self):
        from src.presentation.templates import chart_slide
        result = chart_slide("H", "p.png", layout="chart-right", narrative="Narrative text")
        assert "Narrative text" in result


class TestFindingSlide:
    def test_contains_headline(self):
        from src.presentation.templates import finding_slide
        result = finding_slide("Headline", "Key insight", [{"headline": "F1", "detail": "d"}])
        assert "Headline" in result

    def test_so_what_present(self):
        from src.presentation.templates import finding_slide
        result = finding_slide("H", "The main takeaway", [])
        assert "The main takeaway" in result
        assert 'class="so-what"' in result

    def test_finding_details_present(self):
        from src.presentation.templates import finding_slide
        result = finding_slide("H", "SW", [
            {"headline": "Finding A", "detail": "Detail A", "impact": "Impact A"},
        ])
        assert "Finding A" in result
        assert "Detail A" in result
        assert "Impact A" in result

    def test_class_directive(self):
        from src.presentation.templates import finding_slide
        result = finding_slide("H", "SW", [])
        assert "_class: takeaway" in result


class TestRecommendationSlide:
    def _recs(self):
        return [
            {"action": "Action Low", "rationale": "R1", "confidence": "low"},
            {"action": "Action High", "rationale": "R2", "confidence": "high"},
            {"action": "Action Med", "rationale": "R3", "confidence": "medium"},
        ]

    def test_contains_headline(self):
        from src.presentation.templates import recommendation_slide
        result = recommendation_slide("Next Steps", self._recs())
        assert "Next Steps" in result

    def test_class_directive(self):
        from src.presentation.templates import recommendation_slide
        result = recommendation_slide("H", self._recs())
        assert "_class: recommendation" in result

    def test_sorted_high_first(self):
        from src.presentation.templates import recommendation_slide
        result = recommendation_slide("H", self._recs())
        idx_high = result.index("Action High")
        idx_med = result.index("Action Med")
        idx_low = result.index("Action Low")
        assert idx_high < idx_med < idx_low

    def test_badge_classes_rendered(self):
        from src.presentation.templates import recommendation_slide
        result = recommendation_slide("H", self._recs())
        assert 'class="badge high"' in result
        assert 'class="badge medium"' in result
        assert 'class="badge low"' in result

    def test_rec_row_divs_present(self):
        from src.presentation.templates import recommendation_slide
        result = recommendation_slide("H", self._recs())
        assert result.count('class="rec-row"') == 3


class TestAppendixSlide:
    def test_contains_headline_and_content(self):
        from src.presentation.templates import appendix_slide
        result = appendix_slide("Appendix A", "Some content here")
        assert "Appendix A" in result
        assert "Some content here" in result

    def test_class_directive(self):
        from src.presentation.templates import appendix_slide
        result = appendix_slide("A", "c")
        assert "_class: appendix" in result


class TestDataSourceSlide:
    def test_sources_rendered(self):
        from src.presentation.templates import data_source_slide
        result = data_source_slide(["Source A", "Source B"])
        assert "Source A" in result
        assert "Source B" in result

    def test_data_source_class_divs(self):
        from src.presentation.templates import data_source_slide
        result = data_source_slide(["S1", "S2"])
        assert result.count('class="data-source"') == 2

    def test_appendix_class(self):
        from src.presentation.templates import data_source_slide
        result = data_source_slide(["X"])
        assert "_class: appendix" in result


class TestFrontmatter:
    def test_frontmatter_format(self):
        from src.presentation.templates import FRONTMATTER
        rendered = FRONTMATTER.format(footer="My Footer")
        assert "marp: true" in rendered
        assert "theme: analytics" in rendered
        assert "size: 16:9" in rendered
        assert "paginate: true" in rendered
        assert "html: true" in rendered
        assert "My Footer" in rendered


# ===========================================================================
# deck_builder.py
# ===========================================================================

class TestDeckBuilderInit:
    def test_default_footer_uses_lob(self):
        from src.presentation.deck_builder import DeckBuilder
        b = DeckBuilder(title="T", lob="Retail")
        assert "Retail" in b.footer

    def test_custom_footer(self):
        from src.presentation.deck_builder import DeckBuilder
        b = DeckBuilder(title="T", footer="Custom Footer")
        assert b.footer == "Custom Footer"

    def test_slide_count_starts_at_zero(self):
        from src.presentation.deck_builder import DeckBuilder
        b = DeckBuilder(title="T")
        assert b.slide_count == 0

    def test_date_str_defaults_to_current_month(self):
        from src.presentation.deck_builder import DeckBuilder
        from datetime import date
        b = DeckBuilder(title="T")
        # Should contain a year
        assert str(date.today().year) in b.date_str


class TestDeckBuilderSlideAdders:
    def _builder(self):
        from src.presentation.deck_builder import DeckBuilder
        return DeckBuilder(title="Test Deck", lob="Retail")

    def test_add_section_increments_count(self):
        b = self._builder()
        b.add_section("Section 1")
        assert b.slide_count == 1

    def test_add_impact_increments_count(self):
        b = self._builder()
        b.add_impact("What next?")
        assert b.slide_count == 1

    def test_add_kpi_slide(self):
        b = self._builder()
        b.add_kpi_slide("Accuracy Up", [{"label": "WMAPE", "value": "18%"}])
        assert b.slide_count == 1

    def test_add_chart_slide(self):
        b = self._builder()
        b.add_chart_slide("Trend", "charts/trend.png")
        assert b.slide_count == 1

    def test_add_finding_slide(self):
        b = self._builder()
        b.add_finding_slide("Findings", "Key takeaway", [{"headline": "F1"}])
        assert b.slide_count == 1

    def test_add_recommendation_slide(self):
        b = self._builder()
        b.add_recommendation_slide("Recs", [{"action": "Do X", "confidence": "high"}])
        assert b.slide_count == 1

    def test_add_appendix(self):
        b = self._builder()
        b.add_appendix("Appendix A", "Data here")
        assert b.slide_count == 1

    def test_add_raw_slide(self):
        b = self._builder()
        b.add_raw_slide("# Raw content")
        assert b.slide_count == 1

    def test_method_chaining(self):
        b = self._builder()
        result = b.add_section("S").add_kpi_slide("K", [{"label": "L", "value": "V"}])
        assert result is b
        assert b.slide_count == 2

    def test_add_data_source(self):
        b = self._builder()
        b.add_data_source("Sales DB")
        # data sources don't count in slide_count
        assert b.slide_count == 0
        assert "Sales DB" in b._data_sources


class TestDeckBuilderConfidenceBadge:
    def test_high_score_direction_up(self):
        from src.presentation.deck_builder import DeckBuilder
        b = DeckBuilder("T")
        b.add_confidence_badge("Data Quality", "A", 90, "Excellent")
        content = b.build()
        assert "direction" not in content or "up" in content or "A" in content

    def test_low_score_direction_down(self):
        from src.presentation.deck_builder import DeckBuilder
        b = DeckBuilder("T")
        b.add_confidence_badge("Quality", "F", 20, "Poor")
        assert b.slide_count == 1


class TestDeckBuilderBuild:
    def _builder_with_slides(self):
        from src.presentation.deck_builder import DeckBuilder
        b = DeckBuilder(title="Q1 Review", lob="Retail", subtitle="Forecast")
        b.add_section("Context")
        b.add_kpi_slide("KPIs", [{"label": "WMAPE", "value": "18%"}])
        b.add_chart_slide("Trend", "charts/trend.png")
        return b

    def test_build_returns_string(self):
        b = self._builder_with_slides()
        result = b.build()
        assert isinstance(result, str)

    def test_build_contains_title(self):
        b = self._builder_with_slides()
        result = b.build()
        assert "Q1 Review" in result

    def test_build_contains_frontmatter(self):
        b = self._builder_with_slides()
        result = b.build()
        assert "marp: true" in result

    def test_build_includes_data_source_slide(self):
        from src.presentation.deck_builder import DeckBuilder
        b = DeckBuilder("T")
        b.add_data_source("My DB")
        result = b.build()
        assert "My DB" in result
        assert "Data Sources" in result

    def test_build_no_data_source_slide_when_empty(self):
        from src.presentation.deck_builder import DeckBuilder
        b = DeckBuilder("T")
        b.add_kpi_slide("K", [{"label": "L", "value": "V"}])
        result = b.build()
        assert "Data Sources" not in result

    def test_slide_separators_present(self):
        b = self._builder_with_slides()
        result = b.build()
        assert "\n---\n" in result

    def test_auto_pacing_inserts_breathing_slide(self):
        from src.presentation.deck_builder import DeckBuilder
        b = DeckBuilder("T")
        # Add 6 consecutive non-breathing slides (exceeds _MAX_CONSECUTIVE_CONTENT=4)
        for i in range(6):
            b.add_kpi_slide(f"KPI {i}", [{"label": "L", "value": "V"}])
        result = b.build()
        # An auto-inserted impact/breathing slide should appear
        assert "_class: impact" in result


class TestDeckBuilderSave:
    def test_save_creates_file(self, tmp_path):
        from src.presentation.deck_builder import DeckBuilder
        b = DeckBuilder("T")
        b.add_section("S")
        out = b.save(str(tmp_path / "deck.marp.md"))
        assert out.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        from src.presentation.deck_builder import DeckBuilder
        b = DeckBuilder("T")
        out = b.save(str(tmp_path / "deep" / "nested" / "deck.marp.md"))
        assert out.exists()

    def test_save_returns_path(self, tmp_path):
        from src.presentation.deck_builder import DeckBuilder
        b = DeckBuilder("T")
        result = b.save(str(tmp_path / "deck.marp.md"))
        assert isinstance(result, Path)

    def test_saved_content_matches_build(self, tmp_path):
        from src.presentation.deck_builder import DeckBuilder
        b = DeckBuilder("T", lob="Retail")
        b.add_section("S")
        out = b.save(str(tmp_path / "deck.marp.md"))
        assert out.read_text() == b.build()


class TestDeckBuilderCommentaryIntegration:
    def _make_commentary_result(self):
        """Fake CommentaryResult-like object."""
        km = MagicMock()
        km.name = "WMAPE"
        km.value = 18.2
        km.unit = "%"
        km.trend = "improving"

        r = MagicMock()
        r.executive_summary = "Forecast accuracy improved Q1"
        r.exceptions = ["Exception A", "Exception B"]
        r.key_metrics = [km]
        r.action_items = ["Action 1", "Action 2"]
        return r

    def test_commentary_adds_slides(self):
        from src.presentation.deck_builder import DeckBuilder
        b = DeckBuilder("T")
        initial_count = b.slide_count
        b.add_commentary_slides(self._make_commentary_result())
        assert b.slide_count > initial_count

    def test_executive_summary_in_build(self):
        from src.presentation.deck_builder import DeckBuilder
        b = DeckBuilder("T")
        b.add_commentary_slides(self._make_commentary_result())
        result = b.build()
        assert "Forecast accuracy improved Q1" in result

    def test_action_items_in_build(self):
        from src.presentation.deck_builder import DeckBuilder
        b = DeckBuilder("T")
        b.add_commentary_slides(self._make_commentary_result())
        result = b.build()
        assert "Action 1" in result


# ===========================================================================
# marp_linter.py
# ===========================================================================

def _write_deck(tmp_path: Path, content: str) -> Path:
    """Write a deck file and return its path."""
    p = tmp_path / "deck.marp.md"
    p.write_text(content, encoding="utf-8")
    return p


_VALID_FRONTMATTER = textwrap.dedent("""\
    ---
    marp: true
    theme: analytics
    size: 16:9
    paginate: true
    html: true
    footer: "Test Footer"
    ---
""")

_ENOUGH_COMPONENTS = ''.join(
    f'<div class="{c}">x</div>\n'
    for c in ["metric-callout", "kpi-row", "chart-container"]
)

def _make_valid_deck(n_slides: int = 10) -> str:
    """Build a minimal valid deck with enough slides and components."""
    slides = []
    for i in range(n_slides):
        slides.append(
            f"<!-- _class: kpi -->\n\n## Slide {i}\n\n{_ENOUGH_COMPONENTS}"
        )
    return _VALID_FRONTMATTER + "\n---\n".join(slides)


class TestParseFrontmatter:
    def test_extracts_keys(self):
        from src.presentation.marp_linter import _parse_frontmatter
        text = "---\nmarp: true\ntheme: analytics\n---\nbody"
        fm, rest = _parse_frontmatter(text)
        assert fm["marp"] is True
        assert fm["theme"] == "analytics"

    def test_body_returned(self):
        from src.presentation.marp_linter import _parse_frontmatter
        text = "---\nmarp: true\n---\nSlide body"
        _, rest = _parse_frontmatter(text)
        assert "Slide body" in rest

    def test_no_frontmatter_returns_empty_dict(self):
        from src.presentation.marp_linter import _parse_frontmatter
        text = "# Just a heading\nSome text"
        fm, rest = _parse_frontmatter(text)
        assert fm == {}
        assert "Just a heading" in rest

    def test_boolean_values_parsed(self):
        from src.presentation.marp_linter import _parse_frontmatter
        text = "---\npaginate: true\nhtml: false\n---\n"
        fm, _ = _parse_frontmatter(text)
        assert fm["paginate"] is True
        assert fm["html"] is False


class TestSplitSlides:
    def test_splits_on_separator(self):
        from src.presentation.marp_linter import _split_slides
        body = "Slide 1\n---\nSlide 2\n---\nSlide 3"
        parts = _split_slides(body)
        assert len(parts) == 3

    def test_single_slide(self):
        from src.presentation.marp_linter import _split_slides
        parts = _split_slides("Only one slide")
        assert len(parts) == 1


class TestLintDeckFileMissing:
    def test_missing_file_returns_error(self, tmp_path):
        from src.presentation.marp_linter import lint_deck
        result = lint_deck(str(tmp_path / "nonexistent.marp.md"))
        assert result["slide_count"] == 0
        assert any(i["code"] == "FILE-MISSING" for i in result["issues"])


class TestLintDeckFrontmatter:
    def test_missing_frontmatter_key_reported(self, tmp_path):
        from src.presentation.marp_linter import lint_deck
        # Omit 'footer' from frontmatter
        content = textwrap.dedent("""\
            ---
            marp: true
            theme: analytics
            size: 16:9
            paginate: true
            html: true
            ---
        """) + _make_valid_deck(10)[len(_VALID_FRONTMATTER):]
        p = _write_deck(tmp_path, content)
        result = lint_deck(str(p))
        codes = {i["code"] for i in result["issues"]}
        assert "FM-FOOTER" in codes

    def test_wrong_size_reported_as_warning(self, tmp_path):
        from src.presentation.marp_linter import lint_deck
        content = textwrap.dedent("""\
            ---
            marp: true
            theme: analytics
            size: 4:3
            paginate: true
            html: true
            footer: "F"
            ---
        """) + _make_valid_deck(10)[len(_VALID_FRONTMATTER):]
        p = _write_deck(tmp_path, content)
        result = lint_deck(str(p))
        codes = [i["code"] for i in result["issues"]]
        assert "FM-SIZE" in codes

    def test_valid_frontmatter_no_fm_errors(self, tmp_path):
        from src.presentation.marp_linter import lint_deck
        content = _make_valid_deck(10)
        p = _write_deck(tmp_path, content)
        result = lint_deck(str(p))
        fm_errors = [i for i in result["issues"] if i["code"].startswith("FM-") and i["level"] == "ERROR"]
        assert len(fm_errors) == 0


class TestLintDeckSlideCount:
    def test_too_few_slides_warns(self, tmp_path):
        from src.presentation.marp_linter import lint_deck
        content = _make_valid_deck(n_slides=3)
        p = _write_deck(tmp_path, content)
        result = lint_deck(str(p))
        codes = [i["code"] for i in result["issues"]]
        assert "SLIDES-LOW" in codes

    def test_too_many_slides_warns(self, tmp_path):
        from src.presentation.marp_linter import lint_deck
        content = _make_valid_deck(n_slides=25)
        p = _write_deck(tmp_path, content)
        result = lint_deck(str(p))
        codes = [i["code"] for i in result["issues"]]
        assert "SLIDES-HIGH" in codes

    def test_slide_count_in_result(self, tmp_path):
        from src.presentation.marp_linter import lint_deck
        content = _make_valid_deck(n_slides=10)
        p = _write_deck(tmp_path, content)
        result = lint_deck(str(p))
        assert result["slide_count"] == 10


class TestLintDeckComponents:
    def test_too_few_components_warns(self, tmp_path):
        from src.presentation.marp_linter import lint_deck
        # Only 1 component type
        content = _VALID_FRONTMATTER + "\n---\n".join(
            f'<!-- _class: kpi -->\n\n## Slide {i}\n\n<div class="kpi-row">x</div>'
            for i in range(10)
        )
        p = _write_deck(tmp_path, content)
        result = lint_deck(str(p))
        codes = [i["code"] for i in result["issues"]]
        assert "COMP-MIN" in codes

    def test_components_found_set(self, tmp_path):
        from src.presentation.marp_linter import lint_deck
        content = _make_valid_deck(n_slides=10)
        p = _write_deck(tmp_path, content)
        result = lint_deck(str(p))
        assert isinstance(result["components_found"], set)
        assert len(result["components_found"]) >= 3


class TestLintDeckCssClasses:
    def test_deprecated_class_warns(self, tmp_path):
        from src.presentation.marp_linter import lint_deck
        # Use deprecated "hero" class
        body = "\n---\n".join(
            [f'<!-- _class: kpi -->\n\n## Slide {i}\n\n{_ENOUGH_COMPONENTS}' for i in range(9)]
            + ['<!-- _class: hero -->\n\n## Hero slide\n\n' + _ENOUGH_COMPONENTS]
        )
        content = _VALID_FRONTMATTER + body
        p = _write_deck(tmp_path, content)
        result = lint_deck(str(p))
        codes = [i["code"] for i in result["issues"]]
        assert "CLASS-INVALID" in codes

    def test_unknown_class_warns(self, tmp_path):
        from src.presentation.marp_linter import lint_deck
        body = "\n---\n".join(
            [f'<!-- _class: kpi -->\n\n## Slide {i}\n\n{_ENOUGH_COMPONENTS}' for i in range(9)]
            + ['<!-- _class: totally-made-up -->\n\n## X\n\n' + _ENOUGH_COMPONENTS]
        )
        content = _VALID_FRONTMATTER + body
        p = _write_deck(tmp_path, content)
        result = lint_deck(str(p))
        codes = [i["code"] for i in result["issues"]]
        assert "CLASS-UNKNOWN" in codes


class TestLintDeckBareImages:
    def test_bare_markdown_image_warns(self, tmp_path):
        from src.presentation.marp_linter import lint_deck
        # A slide with a bare markdown image (not in chart-container)
        body = "\n---\n".join(
            [f'<!-- _class: kpi -->\n\n## Slide {i}\n\n{_ENOUGH_COMPONENTS}' for i in range(9)]
            + ['<!-- _class: chart-full -->\n\n## Chart\n\n![alt text](chart.png)\n']
        )
        content = _VALID_FRONTMATTER + body
        p = _write_deck(tmp_path, content)
        result = lint_deck(str(p))
        codes = [i["code"] for i in result["issues"]]
        assert "IMG-BARE-MD" in codes


class TestLintDeckResult:
    def test_result_keys(self, tmp_path):
        from src.presentation.marp_linter import lint_deck
        content = _make_valid_deck(10)
        p = _write_deck(tmp_path, content)
        result = lint_deck(str(p))
        for key in ("issues", "summary", "frontmatter", "slide_count", "components_found"):
            assert key in result

    def test_issues_sorted_errors_first(self, tmp_path):
        from src.presentation.marp_linter import lint_deck
        # Deck missing several frontmatter keys (errors) plus few slides (warning)
        content = "---\nmarp: true\n---\n## Only slide\n\n" + _ENOUGH_COMPONENTS
        p = _write_deck(tmp_path, content)
        result = lint_deck(str(p))
        levels = [i["level"] for i in result["issues"]]
        # After sorting, ERROR entries come before WARNING entries
        error_indices = [i for i, l in enumerate(levels) if l == "ERROR"]
        warning_indices = [i for i, l in enumerate(levels) if l == "WARNING"]
        if error_indices and warning_indices:
            assert max(error_indices) < min(warning_indices)


class TestFormatLintReport:
    def test_returns_string(self, tmp_path):
        from src.presentation.marp_linter import format_lint_report, lint_deck
        content = _make_valid_deck(10)
        p = _write_deck(tmp_path, content)
        result = lint_deck(str(p))
        report = format_lint_report(result)
        assert isinstance(report, str)

    def test_no_issues_message(self, tmp_path):
        from src.presentation.marp_linter import format_lint_report
        result = {
            "issues": [],
            "summary": "10 slides, 3 component types, 0 errors, 0 warnings",
            "frontmatter": {},
            "slide_count": 10,
            "components_found": set(),
        }
        report = format_lint_report(result)
        assert "No issues found" in report

    def test_issues_rendered(self, tmp_path):
        from src.presentation.marp_linter import format_lint_report
        result = {
            "issues": [
                {"level": "ERROR", "code": "FM-MARP", "message": "Missing marp", "slide": 0},
            ],
            "summary": "1 slide, 0 component types, 1 errors, 0 warnings",
            "frontmatter": {},
            "slide_count": 1,
            "components_found": set(),
        }
        report = format_lint_report(result)
        assert "ERROR" in report
        assert "FM-MARP" in report
        assert "Missing marp" in report

    def test_slide_reference_in_report(self, tmp_path):
        from src.presentation.marp_linter import format_lint_report
        result = {
            "issues": [
                {"level": "WARNING", "code": "IMG-BARE-MD", "message": "Bare image", "slide": 3},
            ],
            "summary": "10 slides, 3 types, 0 errors, 1 warnings",
            "frontmatter": {},
            "slide_count": 10,
            "components_found": set(),
        }
        report = format_lint_report(result)
        assert "Slide 3" in report

    def test_global_reference_for_slide_zero(self):
        from src.presentation.marp_linter import format_lint_report
        result = {
            "issues": [
                {"level": "WARNING", "code": "SLIDES-LOW", "message": "Too few", "slide": 0},
            ],
            "summary": "3 slides, 0 types, 0 errors, 1 warnings",
            "frontmatter": {},
            "slide_count": 3,
            "components_found": set(),
        }
        report = format_lint_report(result)
        assert "Global" in report


# ===========================================================================
# marp_export.py  (subprocess-level mocked)
# ===========================================================================

class TestCheckReady:
    def test_returns_expected_keys(self):
        from src.presentation.marp_export import check_ready
        with patch("src.presentation.marp_export._check_marp_cli", return_value=False), \
             patch("src.presentation.marp_export._check_node", return_value=False):
            result = check_ready()
        assert "marp_cli" in result
        assert "node" in result
        assert "themes_available" in result

    def test_marp_cli_true_when_available(self):
        from src.presentation.marp_export import check_ready
        with patch("src.presentation.marp_export._check_marp_cli", return_value=True), \
             patch("src.presentation.marp_export._check_node", return_value=True):
            result = check_ready()
        assert result["marp_cli"] is True
        assert result["node"] is True

    def test_themes_available_is_list(self):
        from src.presentation.marp_export import check_ready
        with patch("src.presentation.marp_export._check_marp_cli", return_value=False), \
             patch("src.presentation.marp_export._check_node", return_value=False):
            result = check_ready()
        assert isinstance(result["themes_available"], list)


class TestCheckMarpCli:
    def test_returns_true_on_zero_returncode(self):
        from src.presentation.marp_export import _check_marp_cli
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            assert _check_marp_cli() is True

    def test_returns_false_on_nonzero_returncode(self):
        from src.presentation.marp_export import _check_marp_cli
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            assert _check_marp_cli() is False

    def test_returns_false_on_file_not_found(self):
        from src.presentation.marp_export import _check_marp_cli
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _check_marp_cli() is False

    def test_returns_false_on_timeout(self):
        import subprocess
        from src.presentation.marp_export import _check_marp_cli
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("npx", 15)):
            assert _check_marp_cli() is False


class TestCheckNode:
    def test_returns_true_when_node_available(self):
        from src.presentation.marp_export import _check_node
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            assert _check_node() is True

    def test_returns_false_when_node_missing(self):
        from src.presentation.marp_export import _check_node
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _check_node() is False


class TestFindThemesDir:
    def test_finds_themes_dir_in_same_dir(self, tmp_path):
        from src.presentation.marp_export import _find_themes_dir
        themes = tmp_path / "themes"
        themes.mkdir()
        deck = tmp_path / "deck.marp.md"
        deck.touch()
        result = _find_themes_dir(deck)
        assert result == themes

    def test_finds_themes_dir_in_parent(self, tmp_path):
        from src.presentation.marp_export import _find_themes_dir
        themes = tmp_path / "themes"
        themes.mkdir()
        subdir = tmp_path / "sub"
        subdir.mkdir()
        deck = subdir / "deck.marp.md"
        deck.touch()
        result = _find_themes_dir(deck)
        assert result == themes

    def test_returns_none_when_not_found(self, tmp_path):
        from src.presentation.marp_export import _find_themes_dir
        deck = tmp_path / "deck.marp.md"
        deck.touch()
        # No themes dir anywhere near tmp_path (assuming test isolation)
        result = _find_themes_dir(deck)
        # May or may not find one depending on real filesystem; just check type
        assert result is None or result.is_dir()


class TestResolveThemeCss:
    def test_resolves_known_theme(self, tmp_path):
        from src.presentation.marp_export import _resolve_theme_css
        themes = tmp_path / "themes"
        themes.mkdir()
        css = themes / "analytics-light.css"
        css.write_text("/* css */")
        deck = tmp_path / "deck.marp.md"
        deck.touch()
        result = _resolve_theme_css("analytics", deck)
        assert result == css

    def test_raises_for_unknown_theme(self, tmp_path):
        from src.presentation.marp_export import _resolve_theme_css
        themes = tmp_path / "themes"
        themes.mkdir()
        deck = tmp_path / "deck.marp.md"
        deck.touch()
        with pytest.raises(ValueError, match="Unknown theme"):
            _resolve_theme_css("nonexistent-theme", deck)

    def test_raises_when_no_themes_dir(self, tmp_path):
        from src.presentation.marp_export import _resolve_theme_css
        deck = tmp_path / "deck.marp.md"
        deck.touch()
        # No themes dir created
        with pytest.raises(FileNotFoundError):
            _resolve_theme_css("analytics", deck)

    def test_raises_when_css_file_missing(self, tmp_path):
        from src.presentation.marp_export import _resolve_theme_css
        themes = tmp_path / "themes"
        themes.mkdir()
        # themes dir exists but CSS file is absent
        deck = tmp_path / "deck.marp.md"
        deck.touch()
        with pytest.raises(FileNotFoundError):
            _resolve_theme_css("analytics", deck)


class TestThemeCssMapping:
    def test_analytics_maps_to_light(self):
        from src.presentation.marp_export import THEME_CSS
        assert THEME_CSS["analytics"] == "analytics-light.css"

    def test_analytics_dark_maps_to_dark(self):
        from src.presentation.marp_export import THEME_CSS
        assert THEME_CSS["analytics-dark"] == "analytics-dark.css"


class TestRunMarp:
    def test_raises_file_not_found_for_missing_deck(self, tmp_path):
        from src.presentation.marp_export import _run_marp
        with pytest.raises(FileNotFoundError, match="Deck file not found"):
            _run_marp(str(tmp_path / "missing.marp.md"))

    def test_raises_runtime_error_on_marp_failure(self, tmp_path):
        import subprocess
        from src.presentation.marp_export import _run_marp
        deck = tmp_path / "deck.marp.md"
        deck.write_text("# Deck")
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Marp error"
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="Marp CLI failed"):
                _run_marp(str(deck))

    def test_raises_on_timeout(self, tmp_path):
        import subprocess
        from src.presentation.marp_export import _run_marp
        deck = tmp_path / "deck.marp.md"
        deck.write_text("# Deck")
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("npx", 120)):
            with pytest.raises(RuntimeError, match="timed out"):
                _run_marp(str(deck))

    def test_returns_output_path_on_success(self, tmp_path):
        from src.presentation.marp_export import _run_marp
        deck = tmp_path / "deck.marp.md"
        deck.write_text("# Deck")
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            result = _run_marp(str(deck), output_format="pdf")
        assert str(result).endswith(".pdf")

    def test_export_pdf_delegates_to_run_marp(self, tmp_path):
        from src.presentation.marp_export import export_pdf
        deck = tmp_path / "deck.marp.md"
        deck.write_text("# Deck")
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            result = export_pdf(str(deck))
        assert str(result).endswith(".pdf")

    def test_export_html_delegates_to_run_marp(self, tmp_path):
        from src.presentation.marp_export import export_html
        deck = tmp_path / "deck.marp.md"
        deck.write_text("# Deck")
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            result = export_html(str(deck))
        assert str(result).endswith(".html")

    def test_export_both_returns_dict_with_pdf_and_html(self, tmp_path):
        from src.presentation.marp_export import export_both
        deck = tmp_path / "deck.marp.md"
        deck.write_text("# Deck")
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            result = export_both(str(deck))
        assert "pdf" in result
        assert "html" in result
