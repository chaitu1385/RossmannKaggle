"""
Chart palette — unified color system for the Forecasting Platform.

Single source of truth for all colors used in charts, dashboards, and reports.
Colors follow WCAG 2.1 AA contrast guidelines and are designed for
colorblind accessibility.

Usage:
    from src.visualization.chart_palette import (
        COLORS, SEVERITY_COLORS, MODEL_LAYER_COLORS, FVA_COLORS,
        palette_for_n, ensure_contrast, format_hex,
    )

    # Use in matplotlib
    ax.bar(x, y, color=COLORS["primary"])

    # Get N distinct colors for N series
    colors = palette_for_n(5)

    # Ensure text is readable on a given background
    fg = ensure_contrast("#AABBCC", background="#FFFFFF")
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Core color palette
# ---------------------------------------------------------------------------

COLORS = {
    # Brand / primary
    "primary":    "#4361EE",
    "secondary":  "#3A0CA3",
    "accent":     "#F72585",

    # Semantic
    "success":    "#06D6A0",
    "warning":    "#FFD166",
    "danger":     "#EF476F",
    "info":       "#4361EE",

    # Neutral scale
    "neutral":    "#8D99AE",
    "gray900":    "#1F2937",
    "gray700":    "#374151",
    "gray600":    "#6B7280",
    "gray400":    "#9CA3AF",
    "gray200":    "#E5E7EB",
    "gray100":    "#F3F4F6",

    # Background
    "bg":         "#F8F9FA",
    "bg_warm":    "#F7F6F2",
    "white":      "#FFFFFF",

    # SWD highlight (focus vs. context)
    "focus":      "#4361EE",
    "comparison": "#E5E7EB",
    "muted":      "#9CA3AF",
}

# ---------------------------------------------------------------------------
# Domain-specific palettes
# ---------------------------------------------------------------------------

SEVERITY_COLORS = {
    "critical":  "#EF476F",
    "warning":   "#FFD166",
    "info":      "#4361EE",
}

FVA_COLORS = {
    "ADDS_VALUE":      "#06D6A0",
    "NEUTRAL":         "#8D99AE",
    "DESTROYS_VALUE":  "#EF476F",
    "BASELINE":        "#4361EE",
}

MODEL_LAYER_COLORS = {
    "naive":          "#8D99AE",
    "statistical":    "#4361EE",
    "ml":             "#06D6A0",
    "neural":         "#F72585",
    "foundation":     "#3A0CA3",
    "intermittent":   "#FFD166",
    "ensemble":       "#7209B7",
    "override":       "#FF6B35",
}

DEMAND_CLASS_COLORS = {
    "smooth":        "#06D6A0",
    "intermittent":  "#FFD166",
    "erratic":       "#F72585",
    "lumpy":         "#EF476F",
}

# Categorical palette — 8 distinct, colorblind-safe colors for series
CATEGORICAL_PALETTE = [
    "#4361EE",  # primary blue
    "#F72585",  # magenta
    "#06D6A0",  # teal
    "#FFD166",  # amber
    "#3A0CA3",  # deep purple
    "#FF6B35",  # orange
    "#7209B7",  # violet
    "#8D99AE",  # neutral gray
]


# ---------------------------------------------------------------------------
# Palette generation
# ---------------------------------------------------------------------------

def palette_for_n(n: int) -> list[str]:
    """Return exactly *n* distinct colors.

    - n <= 8: uses the categorical palette (distinct, colorblind-safe).
    - n > 8: samples evenly from a blue-to-magenta gradient.

    Args:
        n: Number of colors needed.

    Returns:
        List of *n* hex color strings.
    """
    if n <= 0:
        return []
    if n <= len(CATEGORICAL_PALETTE):
        return CATEGORICAL_PALETTE[:n]

    # Generate evenly-spaced colors via HSL interpolation
    import colorsys

    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hls_to_rgb(hue, 0.45, 0.65)
        colors.append(
            f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"
        )
    return colors


# ---------------------------------------------------------------------------
# WCAG contrast utilities
# ---------------------------------------------------------------------------

def format_hex(color: str) -> str:
    """Normalize a hex color to uppercase 6-digit format.

    Handles 3-digit shorthand (``"#ABC"`` -> ``"#AABBCC"``) and strips
    surrounding whitespace.

    Args:
        color: Hex color string with leading ``#``.

    Returns:
        Uppercase 6-digit hex string.
    """
    color = color.strip()
    if not color.startswith("#"):
        color = "#" + color
    raw = color[1:]
    if len(raw) == 3:
        raw = raw[0] * 2 + raw[1] * 2 + raw[2] * 2
    return "#" + raw.upper()[:6]


def ensure_contrast(
    hex_color: str,
    background: str = "#F8F9FA",
    min_ratio: float = 4.5,
) -> str:
    """Adjust *hex_color* so it meets WCAG AA contrast against *background*.

    Uses the WCAG 2.1 relative-luminance formula with proper sRGB
    linearization. If the color already passes, it is returned unchanged.
    Otherwise it is progressively darkened (or lightened when the background
    is dark) until the threshold is met.

    Args:
        hex_color: Foreground color in hex.
        background: Background color in hex.
        min_ratio: Minimum WCAG contrast ratio. Default 4.5 (AA normal text).

    Returns:
        Hex color string meeting the contrast requirement.
    """
    fg = _hex_to_rgb(format_hex(hex_color))
    bg = _hex_to_rgb(format_hex(background))

    fg_lum = _relative_luminance(*fg)
    bg_lum = _relative_luminance(*bg)

    if _contrast_ratio(fg_lum, bg_lum) >= min_ratio:
        return format_hex(hex_color)

    bg_is_light = bg_lum > 0.5

    r, g, b = [float(c) for c in fg]
    for _ in range(200):
        if bg_is_light:
            r = max(0, r - 255 * 0.02)
            g = max(0, g - 255 * 0.02)
            b = max(0, b - 255 * 0.02)
        else:
            r = min(255, r + 255 * 0.02)
            g = min(255, g + 255 * 0.02)
            b = min(255, b + 255 * 0.02)

        new_lum = _relative_luminance(r, g, b)
        if _contrast_ratio(new_lum, bg_lum) >= min_ratio:
            break

    return _rgb_to_hex(int(round(r)), int(round(g)), int(round(b)))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return f"#{r:02X}{g:02X}{b:02X}"


def _linearize(channel_8bit: float) -> float:
    s = channel_8bit / 255.0
    if s <= 0.04045:
        return s / 12.92
    return ((s + 0.055) / 1.055) ** 2.4


def _relative_luminance(r: float, g: float, b: float) -> float:
    return (
        0.2126 * _linearize(r)
        + 0.7152 * _linearize(g)
        + 0.0722 * _linearize(b)
    )


def _contrast_ratio(lum1: float, lum2: float) -> float:
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)
