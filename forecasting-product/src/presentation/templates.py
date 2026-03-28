"""
Slide templates and HTML component helpers for Marp decks.

Each function returns a Markdown string for a single slide.  The deck
builder composes these into a full ``.marp.md`` file.

All components follow the AI Analyst Marp component library: accent-bar,
kpi-row, kpi-card, so-what, finding, chart-container, rec-row, etc.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------

FRONTMATTER = """\
---
marp: true
theme: analytics
size: 16:9
paginate: true
html: true
footer: "{footer}"
---
"""


def _speaker_notes(text: str) -> str:
    """Wrap text in Marp speaker-note HTML comment."""
    return f"\n<!--\n{text}\n-->\n"


# ---------------------------------------------------------------------------
# Slide builders
# ---------------------------------------------------------------------------

def title_slide(
    title: str,
    subtitle: str = "",
    date_str: str = "",
    lob: str = "",
) -> str:
    """Title slide with accent-bar.

    Args:
        title: Deck title (action-oriented).
        subtitle: Optional subtitle.
        date_str: Date string (e.g. "March 2026").
        lob: Line of business.
    """
    parts = [
        "<!-- _class: title -->\n",
        '<div class="accent-bar"></div>\n',
        f"\n# {title}\n",
    ]
    if subtitle:
        parts.append(f"### {subtitle}\n")
    meta = " | ".join(filter(None, [lob, date_str]))
    if meta:
        parts.append(f"\n{meta}\n")
    return "\n".join(parts)


def section_slide(heading: str, subheading: str = "") -> str:
    """Section opener / breathing slide.

    Args:
        heading: Section name (e.g. "Key Findings").
        subheading: Optional description.
    """
    parts = ["<!-- _class: section-opener -->\n", f"\n# {heading}\n"]
    if subheading:
        parts.append(f"\n{subheading}\n")
    return "\n".join(parts)


def impact_slide(question: str) -> str:
    """Breathing / impact slide with a pivotal question.

    Args:
        question: The question to pose (e.g. "What should we prioritize?").
    """
    return f"<!-- _class: impact -->\n\n# {question}\n"


def kpi_slide(
    headline: str,
    metrics: Sequence[Dict[str, Any]],
) -> str:
    """KPI dashboard slide with 2-4 metric cards.

    Args:
        headline: Slide headline (action-oriented).
        metrics: List of dicts, each with keys:
            - ``label``: metric name
            - ``value``: display value (str or number)
            - ``delta``: change string (e.g. "+2.3pp")
            - ``direction``: "up" | "down" | "flat" (CSS class)
            - ``detail``: optional detail line

    Example::

        kpi_slide("Forecast accuracy improved across all LOBs", [
            {"label": "WMAPE", "value": "18.2%", "delta": "-2.1pp", "direction": "down"},
            {"label": "Bias", "value": "+1.4%", "delta": "+0.8pp", "direction": "up"},
        ])
    """
    parts = [
        "<!-- _class: kpi -->\n",
        f"\n## {headline}\n",
        '\n<div class="kpi-row">\n',
    ]

    for m in metrics[:4]:
        direction = m.get("direction", "flat")
        parts.append(f'<div class="kpi-card">')
        parts.append(f'  <div class="kpi-label">{m["label"]}</div>')
        parts.append(f'  <div class="kpi-value">{m["value"]}</div>')
        if "delta" in m:
            parts.append(f'  <div class="delta {direction}">{m["delta"]}</div>')
        if "detail" in m:
            parts.append(f'  <div class="kpi-detail">{m["detail"]}</div>')
        parts.append("</div>\n")

    parts.append("</div>\n")
    return "\n".join(parts)


def chart_slide(
    headline: str,
    chart_path: str,
    source: str = "",
    layout: str = "chart-full",
    narrative: str = "",
) -> str:
    """Chart slide with optional narrative.

    Args:
        headline: Slide headline (must differ from chart's baked-in title).
        chart_path: Relative path to the chart image.
        source: Data source attribution.
        layout: ``"chart-full"``, ``"chart-left"``, or ``"chart-right"``.
        narrative: Optional narrative text (for chart-left/chart-right).
    """
    parts = [f"<!-- _class: {layout} -->\n", f"\n## {headline}\n"]

    if layout == "chart-right" and narrative:
        parts.append(f"\n{narrative}\n")

    parts.append('\n<div class="chart-container">')
    parts.append(f'  <img src="{chart_path}" />')
    if source:
        parts.append(f'  <div class="chart-source">{source}</div>')
    parts.append("</div>\n")

    if layout == "chart-left" and narrative:
        parts.append(f"\n{narrative}\n")

    return "\n".join(parts)


def finding_slide(
    headline: str,
    so_what: str,
    findings: Sequence[Dict[str, str]],
) -> str:
    """Takeaway slide with so-what callout and findings.

    Args:
        headline: Slide headline.
        so_what: Key takeaway (displayed in amber box).
        findings: List of dicts with ``headline``, ``detail``, ``impact``.
    """
    parts = [
        "<!-- _class: takeaway -->\n",
        f"\n## {headline}\n",
        f'\n<div class="so-what">{so_what}</div>\n',
    ]

    for f in findings:
        parts.append('<div class="finding">')
        parts.append(f'  <div class="finding-headline">{f["headline"]}</div>')
        if "detail" in f:
            parts.append(f'  <div class="finding-detail">{f["detail"]}</div>')
        if "impact" in f:
            parts.append(f'  <div class="finding-impact">{f["impact"]}</div>')
        parts.append("</div>\n")

    return "\n".join(parts)


def recommendation_slide(
    headline: str,
    recommendations: Sequence[Dict[str, str]],
) -> str:
    """Recommendation slide with numbered action items.

    Recommendations are sorted High → Medium → Low automatically.

    Args:
        headline: Slide headline.
        recommendations: List of dicts with:
            - ``action``: what to do
            - ``rationale``: why
            - ``confidence``: "high" | "medium" | "low"
    """
    # Sort: high first, then medium, then low
    priority = {"high": 0, "medium": 1, "low": 2}
    sorted_recs = sorted(
        recommendations,
        key=lambda r: priority.get(r.get("confidence", "medium"), 1),
    )

    parts = [
        "<!-- _class: recommendation -->\n",
        f"\n## {headline}\n",
    ]

    for i, rec in enumerate(sorted_recs, 1):
        conf = rec.get("confidence", "medium")
        parts.append(f'<div class="rec-row">')
        parts.append(f'  <div class="rec-number">{i}</div>')
        parts.append(f'  <div class="rec-action">{rec["action"]}</div>')
        if "rationale" in rec:
            parts.append(f'  <div class="rec-rationale">{rec["rationale"]}</div>')
        parts.append(f'  <div class="badge {conf}">{conf.upper()}</div>')
        parts.append("</div>\n")

    return "\n".join(parts)


def appendix_slide(
    headline: str,
    content: str,
) -> str:
    """Appendix / data detail slide.

    Args:
        headline: Slide headline.
        content: Markdown-formatted content (tables, lists, etc.)
    """
    return f"<!-- _class: appendix -->\n\n## {headline}\n\n{content}\n"


def data_source_slide(sources: Sequence[str]) -> str:
    """Data source attribution slide.

    Args:
        sources: List of source descriptions.
    """
    parts = ["<!-- _class: appendix -->\n", "\n## Data Sources\n"]
    for src in sources:
        parts.append(f'<div class="data-source">{src}</div>\n')
    return "\n".join(parts)
