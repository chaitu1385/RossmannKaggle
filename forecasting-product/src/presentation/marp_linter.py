"""
Marp deck linter — validates deck quality before export.

Checks:
    1. Frontmatter completeness (marp, theme, size, paginate, html, footer)
    2. HTML component usage (minimum 3 distinct component types)
    3. CSS class validity (per theme)
    4. Slide count bounds (8-22)
    5. Title collision (chart title ≠ slide headline)
    6. Pacing (max 4 consecutive content slides)
    7. Bare markdown images (must be in chart-container)

Usage::

    from src.presentation.marp_linter import lint_deck, format_lint_report

    result = lint_deck("outputs/deck.marp.md")
    if result["issues"]:
        print(format_lint_report(result))
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CLASSES_LIGHT = {
    "title", "section-opener", "insight", "impact", "two-col",
    "chart-left", "chart-right", "diagram", "chart-full", "kpi",
    "takeaway", "recommendation", "appendix",
}

VALID_CLASSES_DARK = {
    "dark-title", "dark-impact", "section-opener", "insight", "two-col",
    "chart-left", "chart-right", "diagram", "chart-full", "kpi",
    "takeaway", "recommendation", "appendix",
}

INVALID_CLASS_MIGRATION = {
    "breathing": "impact",
    "hero": "title",
    "break": "impact",
    "transition": "section-opener",
}

REQUIRED_FRONTMATTER = {"marp", "theme", "size", "paginate", "html", "footer"}

HTML_COMPONENTS = {
    "metric-callout", "kpi-row", "kpi-card", "so-what", "finding",
    "rec-row", "chart-container", "before-after", "box-grid",
    "flow", "vflow", "layers", "timeline", "checklist", "callout",
    "badge", "delta", "data-source", "accent-bar",
}

MIN_COMPONENT_TYPES = 3
MIN_SLIDES = 8
MAX_SLIDES = 22

BREATHING_CLASSES = {"section-opener", "impact", "dark-impact"}

_CLASS_RE = re.compile(r'<!--\s*_class:\s*(\S+)\s*-->')
_IMG_BARE_RE = re.compile(r'!\[.*?\]\(.*?\)')
_CHART_CONTAINER_RE = re.compile(r'class="chart-container"')


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter from Marp deck text.

    Returns:
        (frontmatter_dict, remaining_text)
    """
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
    if not match:
        return {}, text

    fm_text = match.group(1)
    rest = text[match.end():]

    fm = {}
    for line in fm_text.split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, val = line.partition(":")
            val = val.strip().strip('"').strip("'")
            if val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
            fm[key.strip()] = val

    return fm, rest


def _split_slides(text: str) -> list[str]:
    """Split deck body into individual slides."""
    return re.split(r'\n---\n', text)


# ---------------------------------------------------------------------------
# Lint checks
# ---------------------------------------------------------------------------

def lint_deck(deck_path: str) -> dict:
    """Run all lint checks on a Marp deck.

    Args:
        deck_path: Path to the .marp.md file.

    Returns:
        dict with keys:
            - ``issues``: list of issue dicts (level, code, message, slide)
            - ``summary``: human-readable summary
            - ``frontmatter``: parsed frontmatter
            - ``slide_count``: number of slides
            - ``components_found``: set of HTML component classes found
    """
    path = Path(deck_path)
    if not path.exists():
        return {
            "issues": [{"level": "ERROR", "code": "FILE-MISSING",
                        "message": f"File not found: {deck_path}", "slide": 0}],
            "summary": "File not found",
            "frontmatter": {},
            "slide_count": 0,
            "components_found": set(),
        }

    text = path.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(text)
    slides = _split_slides(body)
    issues: list[dict] = []

    # Determine theme
    theme = str(fm.get("theme", "analytics"))
    valid_classes = VALID_CLASSES_DARK if "dark" in theme else VALID_CLASSES_LIGHT

    # --- Check 1: Frontmatter ---
    for key in REQUIRED_FRONTMATTER:
        if key not in fm:
            issues.append({
                "level": "ERROR", "code": f"FM-{key.upper()}",
                "message": f"Missing required frontmatter key: '{key}'",
                "slide": 0,
            })

    if fm.get("size") and fm["size"] != "16:9":
        issues.append({
            "level": "WARNING", "code": "FM-SIZE",
            "message": f"Size is '{fm['size']}', expected '16:9'",
            "slide": 0,
        })

    # --- Check 2: HTML components ---
    components_found = set()
    for comp in HTML_COMPONENTS:
        if f'class="{comp}"' in text or f"class='{comp}'" in text:
            components_found.add(comp)

    if len(components_found) < MIN_COMPONENT_TYPES:
        issues.append({
            "level": "WARNING", "code": "COMP-MIN",
            "message": (f"Only {len(components_found)} HTML component types used "
                        f"(minimum {MIN_COMPONENT_TYPES}). Found: {sorted(components_found)}"),
            "slide": 0,
        })

    # Check for plain text slides (no components)
    for i, slide in enumerate(slides):
        has_component = any(f'class="{c}"' in slide for c in HTML_COMPONENTS)
        has_class = _CLASS_RE.search(slide) is not None
        if not has_component and not has_class and slide.strip():
            # Skip very short slides (likely separators)
            if len(slide.strip()) > 20:
                issues.append({
                    "level": "INFO", "code": "COMP-PLAIN",
                    "message": "Slide has no HTML components or class directive",
                    "slide": i + 1,
                })

    # --- Check 3: CSS classes ---
    for i, slide in enumerate(slides):
        match = _CLASS_RE.search(slide)
        if match:
            cls = match.group(1)
            if cls in INVALID_CLASS_MIGRATION:
                issues.append({
                    "level": "WARNING", "code": "CLASS-INVALID",
                    "message": (f"Class '{cls}' is deprecated. "
                                f"Use '{INVALID_CLASS_MIGRATION[cls]}' instead."),
                    "slide": i + 1,
                })
            elif cls not in valid_classes:
                issues.append({
                    "level": "WARNING", "code": "CLASS-UNKNOWN",
                    "message": f"Unknown slide class '{cls}'",
                    "slide": i + 1,
                })

    # --- Check 4: Slide count ---
    slide_count = len(slides)
    if slide_count < MIN_SLIDES:
        issues.append({
            "level": "WARNING", "code": "SLIDES-LOW",
            "message": f"Only {slide_count} slides (minimum {MIN_SLIDES})",
            "slide": 0,
        })
    elif slide_count > MAX_SLIDES:
        issues.append({
            "level": "WARNING", "code": "SLIDES-HIGH",
            "message": f"{slide_count} slides exceeds maximum {MAX_SLIDES}",
            "slide": 0,
        })

    # --- Check 5: Title collision ---
    for i, slide in enumerate(slides):
        match = _CLASS_RE.search(slide)
        if match and match.group(1) in ("chart-full", "chart-left", "chart-right"):
            # Extract headline (## text)
            headline_match = re.search(r'^##\s+(.+)$', slide, re.MULTILINE)
            # Extract img alt text (proxy for chart title)
            img_match = re.search(r'<img[^>]*alt="([^"]*)"', slide)
            if headline_match and img_match:
                hl = headline_match.group(1).strip().lower()
                alt = img_match.group(1).strip().lower()
                if hl and alt and hl == alt:
                    issues.append({
                        "level": "WARNING", "code": "R2-COLLISION",
                        "message": "Chart title matches slide headline — use a distinct headline",
                        "slide": i + 1,
                    })

    # --- Check 6: Pacing ---
    content_streak = 0
    for i, slide in enumerate(slides):
        match = _CLASS_RE.search(slide)
        cls = match.group(1) if match else None
        if cls in BREATHING_CLASSES:
            content_streak = 0
        else:
            content_streak += 1
            if content_streak > _MAX_PACING:
                issues.append({
                    "level": "INFO", "code": "R6-PACING",
                    "message": f"Over {_MAX_PACING} consecutive content slides without a break",
                    "slide": i + 1,
                })
                content_streak = 0  # Only warn once per streak

    # --- Check 7: Bare markdown images ---
    for i, slide in enumerate(slides):
        if _IMG_BARE_RE.search(slide) and not _CHART_CONTAINER_RE.search(slide):
            issues.append({
                "level": "WARNING", "code": "IMG-BARE-MD",
                "message": "Bare markdown image — wrap in chart-container div",
                "slide": i + 1,
            })

    # Sort issues by severity
    level_order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    issues.sort(key=lambda x: (level_order.get(x["level"], 9), x["slide"]))

    errors = sum(1 for i in issues if i["level"] == "ERROR")
    warnings = sum(1 for i in issues if i["level"] == "WARNING")

    summary = f"{slide_count} slides, {len(components_found)} component types, "
    summary += f"{errors} errors, {warnings} warnings"

    return {
        "issues": issues,
        "summary": summary,
        "frontmatter": fm,
        "slide_count": slide_count,
        "components_found": components_found,
    }


# Module-level constant used by pacing check
_MAX_PACING = 4


def format_lint_report(result: dict) -> str:
    """Format lint results as a human-readable report.

    Args:
        result: Output from ``lint_deck()``.

    Returns:
        Formatted string.
    """
    lines = [f"Deck Lint Report: {result['summary']}", "=" * 50]

    if not result["issues"]:
        lines.append("No issues found.")
        return "\n".join(lines)

    for issue in result["issues"]:
        slide = f"Slide {issue['slide']}" if issue["slide"] else "Global"
        lines.append(f"[{issue['level']}] {issue['code']} ({slide}): {issue['message']}")

    return "\n".join(lines)
