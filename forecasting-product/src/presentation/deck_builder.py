"""
DeckBuilder — high-level API for building Marp slide decks.

Collects slides, manages pacing (auto-inserts breathing slides), and
writes the final ``.marp.md`` file.

Integrates with:
    - ``CommentaryResult`` from ``src.ai.commentary``
    - Validation confidence badges from ``src.validation``
    - Chart images from ``src.visualization``

Usage::

    builder = DeckBuilder(title="Q1 Forecast Review", lob="Retail")
    builder.add_kpi_slide("Accuracy improved 12%", metrics)
    builder.add_chart_slide("WMAPE trend", "charts/wmape.png")
    builder.add_recommendation_slide("Next steps", recs)
    path = builder.save("outputs/q1_review.marp.md")
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.presentation.templates import (
    FRONTMATTER,
    title_slide,
    kpi_slide,
    chart_slide,
    finding_slide,
    recommendation_slide,
    section_slide,
    impact_slide,
    appendix_slide,
    data_source_slide,
)

# Max consecutive content slides before a breathing slide is inserted
_MAX_CONSECUTIVE_CONTENT = 4

# Slide classes considered "breathing" (don't count toward density)
_BREATHING_CLASSES = {"section-opener", "impact"}


class DeckBuilder:
    """Build a Marp slide deck from structured data.

    Parameters
    ----------
    title : str
        Deck title.
    subtitle : str
        Optional subtitle.
    lob : str
        Line of business (appears on title slide).
    footer : str
        Footer text for all slides.
    date_str : str
        Date string shown on title slide.  Defaults to current month/year.
    """

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        lob: str = "",
        footer: str = "",
        date_str: str = "",
    ):
        self.title = title
        self.subtitle = subtitle
        self.lob = lob
        self.footer = footer or f"{lob} | Forecasting Platform"
        self.date_str = date_str or date.today().strftime("%B %Y")
        self._slides: list[dict] = []
        self._data_sources: list[str] = []

    # ------------------------------------------------------------------
    # Slide adders
    # ------------------------------------------------------------------

    def add_section(self, heading: str, subheading: str = "") -> "DeckBuilder":
        """Add a section-opener / breathing slide."""
        self._slides.append({
            "type": "section-opener",
            "content": section_slide(heading, subheading),
        })
        return self

    def add_impact(self, question: str) -> "DeckBuilder":
        """Add an impact / breathing slide with a pivotal question."""
        self._slides.append({
            "type": "impact",
            "content": impact_slide(question),
        })
        return self

    def add_kpi_slide(
        self,
        headline: str,
        metrics: Sequence[Dict[str, Any]],
    ) -> "DeckBuilder":
        """Add a KPI dashboard slide.

        Args:
            headline: Action-oriented headline.
            metrics: List of dicts with ``label``, ``value``, ``delta``,
                ``direction`` ("up"/"down"/"flat"), ``detail``.
        """
        self._slides.append({
            "type": "kpi",
            "content": kpi_slide(headline, metrics),
        })
        return self

    def add_chart_slide(
        self,
        headline: str,
        chart_path: str,
        source: str = "",
        layout: str = "chart-full",
        narrative: str = "",
    ) -> "DeckBuilder":
        """Add a chart slide.

        Args:
            headline: Must differ from chart's baked-in title.
            chart_path: Relative path to chart image.
            source: Data source attribution.
            layout: "chart-full", "chart-left", or "chart-right".
            narrative: Narrative text (for split layouts).
        """
        self._slides.append({
            "type": layout,
            "content": chart_slide(headline, chart_path, source, layout, narrative),
        })
        return self

    def add_finding_slide(
        self,
        headline: str,
        so_what: str,
        findings: Sequence[Dict[str, str]],
    ) -> "DeckBuilder":
        """Add a takeaway slide with findings.

        Args:
            headline: Slide headline.
            so_what: Key takeaway (amber callout).
            findings: List of dicts with ``headline``, ``detail``, ``impact``.
        """
        self._slides.append({
            "type": "takeaway",
            "content": finding_slide(headline, so_what, findings),
        })
        return self

    def add_recommendation_slide(
        self,
        headline: str,
        recommendations: Sequence[Dict[str, str]],
    ) -> "DeckBuilder":
        """Add a recommendation slide (auto-sorted High→Medium→Low).

        Args:
            headline: Slide headline.
            recommendations: List of dicts with ``action``, ``rationale``,
                ``confidence`` ("high"/"medium"/"low").
        """
        self._slides.append({
            "type": "recommendation",
            "content": recommendation_slide(headline, recommendations),
        })
        return self

    def add_appendix(self, headline: str, content: str) -> "DeckBuilder":
        """Add an appendix slide with arbitrary markdown content."""
        self._slides.append({
            "type": "appendix",
            "content": appendix_slide(headline, content),
        })
        return self

    def add_raw_slide(self, markdown: str, slide_type: str = "custom") -> "DeckBuilder":
        """Add a raw markdown slide (escape hatch)."""
        self._slides.append({"type": slide_type, "content": markdown})
        return self

    def add_data_source(self, source: str) -> "DeckBuilder":
        """Register a data source for the attribution slide."""
        self._data_sources.append(source)
        return self

    def add_confidence_badge(
        self,
        headline: str,
        grade: str,
        score: int,
        description: str,
    ) -> "DeckBuilder":
        """Add a validation confidence badge as a KPI slide.

        Args:
            headline: E.g. "Data Quality Assessment".
            grade: Letter grade (A-F).
            score: Numeric score (0-100).
            description: Grade description.
        """
        metrics = [
            {
                "label": "Confidence Grade",
                "value": grade,
                "delta": f"{score}/100",
                "direction": "up" if score >= 80 else ("flat" if score >= 50 else "down"),
                "detail": description,
            }
        ]
        return self.add_kpi_slide(headline, metrics)

    # ------------------------------------------------------------------
    # Commentary integration
    # ------------------------------------------------------------------

    def add_commentary_slides(self, commentary_result) -> "DeckBuilder":
        """Add slides from a CommentaryResult (from src.ai.commentary).

        Auto-generates:
        - Executive summary as a finding slide
        - Key metrics as a KPI slide
        - Action items as a recommendation slide

        Args:
            commentary_result: CommentaryResult dataclass instance.
        """
        # Executive summary
        if commentary_result.executive_summary:
            self.add_finding_slide(
                headline="Executive Summary",
                so_what=commentary_result.executive_summary[:200],
                findings=[{"headline": e, "detail": ""} for e in commentary_result.exceptions[:3]],
            )

        # Key metrics
        if commentary_result.key_metrics:
            trend_map = {"improving": "up", "degrading": "down", "stable": "flat"}
            metrics = [
                {
                    "label": km.name,
                    "value": f"{km.value}{km.unit}",
                    "direction": trend_map.get(km.trend, "flat"),
                }
                for km in commentary_result.key_metrics[:4]
            ]
            self.add_kpi_slide("Key Performance Metrics", metrics)

        # Action items as recommendations
        if commentary_result.action_items:
            recs = [
                {"action": item, "confidence": "medium"}
                for item in commentary_result.action_items[:5]
            ]
            self.add_recommendation_slide("Action Items", recs)

        return self

    # ------------------------------------------------------------------
    # Build & save
    # ------------------------------------------------------------------

    def _auto_pace(self, slides: list[dict]) -> list[dict]:
        """Insert breathing slides after _MAX_CONSECUTIVE_CONTENT content slides."""
        paced: list[dict] = []
        content_streak = 0

        for slide in slides:
            if slide["type"] in _BREATHING_CLASSES:
                content_streak = 0
                paced.append(slide)
            else:
                content_streak += 1
                if content_streak > _MAX_CONSECUTIVE_CONTENT:
                    paced.append({
                        "type": "impact",
                        "content": impact_slide("Key Question"),
                    })
                    content_streak = 1
                paced.append(slide)

        return paced

    def build(self) -> str:
        """Build the full Marp markdown string.

        Returns:
            Complete Marp deck as a string.
        """
        parts = [FRONTMATTER.format(footer=self.footer)]

        # Title slide
        parts.append(title_slide(self.title, self.subtitle, self.date_str, self.lob))

        # Auto-pace content slides
        paced = self._auto_pace(self._slides)

        for slide in paced:
            parts.append("\n---\n")
            parts.append(slide["content"])

        # Data sources slide (if any)
        if self._data_sources:
            parts.append("\n---\n")
            parts.append(data_source_slide(self._data_sources))

        return "\n".join(parts)

    def save(self, path: str) -> Path:
        """Build and save the deck to a file.

        Args:
            path: Output file path (should end in ``.marp.md``).

        Returns:
            Path to the saved file.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        content = self.build()
        out.write_text(content, encoding="utf-8")
        return out

    @property
    def slide_count(self) -> int:
        """Number of content slides (excludes title)."""
        return len(self._slides)
