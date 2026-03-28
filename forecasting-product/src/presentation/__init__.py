"""
Marp Slide Deck Builder for the Forecasting Platform.

Generates Marp-compatible Markdown decks from analysis results—
S&OP meeting packs, backtest reviews, drift reports, etc.

Uses HTML components (kpi-row, chart-container, finding, so-what, rec-row)
for rich slide layouts. Integrates with the SWD-style chart palette.

Components:
    - deck_builder.py  — Build slide content from structured data
    - marp_export.py   — CLI wrapper for PDF/HTML export
    - marp_linter.py   — Validate deck quality before export
    - templates.py     — Slide templates and component helpers

Usage::

    from src.presentation import DeckBuilder, export_pdf, lint_deck

    builder = DeckBuilder(title="Q1 Forecast Review", lob="Retail")
    builder.add_kpi_slide(metrics)
    builder.add_chart_slide(chart_path, headline="WMAPE improved 12%")
    builder.add_recommendation_slide(recommendations)
    deck_path = builder.save("outputs/q1_review.marp.md")

    issues = lint_deck(deck_path)
    pdf_path = export_pdf(deck_path)
"""

from src.presentation.deck_builder import DeckBuilder
from src.presentation.marp_export import export_pdf, export_html, export_both, check_ready
from src.presentation.marp_linter import lint_deck, format_lint_report
from src.presentation.templates import (
    FRONTMATTER,
    title_slide,
    kpi_slide,
    chart_slide,
    finding_slide,
    recommendation_slide,
    section_slide,
    impact_slide,
)

__all__ = [
    "DeckBuilder",
    "export_pdf", "export_html", "export_both", "check_ready",
    "lint_deck", "format_lint_report",
    "FRONTMATTER",
    "title_slide", "kpi_slide", "chart_slide", "finding_slide",
    "recommendation_slide", "section_slide", "impact_slide",
]
