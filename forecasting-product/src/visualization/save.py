"""
SWD Save Utility — Chart export helper for the Forecasting Platform.

Provides ``save_chart()`` for saving matplotlib figures with consistent
DPI, tight layout, and background color settings.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.visualization.chart_palette import COLORS


def save_chart(fig, path, dpi: int = 150, close: bool = True) -> None:
    """Save a chart with tight layout and correct DPI.

    Args:
        fig: Matplotlib Figure.
        path: Output file path (str or Path).
        dpi: Resolution. Default: 150.
        close: If True (default), close the figure after saving.
    """
    fig.tight_layout()
    fig.savefig(
        path, dpi=dpi, bbox_inches="tight",
        facecolor=COLORS["bg"], edgecolor="none",
    )
    if close:
        plt.close(fig)
