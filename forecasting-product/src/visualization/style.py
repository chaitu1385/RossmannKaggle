"""
SWD Style — Storytelling with Data theme setup for the Forecasting Platform.

Provides the core style application function and chart size constant.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.visualization.chart_palette import COLORS

# ---------------------------------------------------------------------------
# Chart dimensions
# ---------------------------------------------------------------------------

CHART_FIGSIZE = (10, 6)  # ~1500 x 900 @ 150 DPI — good for slides and reports

# ---------------------------------------------------------------------------
# Style file (deployed alongside this module)
# ---------------------------------------------------------------------------

_STYLE_FILE = Path(__file__).with_name("forecasting_chart_style.mplstyle")


def swd_style() -> dict:
    """Apply the SWD matplotlib style and return the color palette.

    Call this at the start of any chart-creation flow. Loads the .mplstyle
    file if present, otherwise applies critical settings directly.

    Returns:
        dict: Color palette mapping (e.g. ``colors["primary"] -> "#4361EE"``).
    """
    if _STYLE_FILE.exists():
        plt.style.use(str(_STYLE_FILE))
    else:
        plt.rcParams.update({
            "figure.figsize": (10, 6),
            "figure.dpi": 150,
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["bg"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "text.color": COLORS["gray900"],
            "axes.labelcolor": COLORS["gray600"],
        })
    return dict(COLORS)
