"""
SWD Platform Analytics — Forecasting Platform-specific chart builders.

Provides ``fva_cascade_chart()`` for Forecast Value Added layer comparisons
and ``demand_class_chart()`` for demand classification distributions.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.visualization.chart_palette import COLORS, FVA_COLORS
from src.visualization.style import CHART_FIGSIZE, swd_style
from src.visualization.annotations import action_title


def fva_cascade_chart(
    layers: Sequence[str],
    wmape_values: Sequence[float],
    fva_labels: Sequence[str] | None = None,
    fig=None,
    ax=None,
) -> Tuple[Any, Any]:
    """FVA cascade bar chart showing error at each forecasting layer.

    Bars are color-coded by FVA classification (adds value / neutral /
    destroys value).

    Args:
        layers: Layer names (e.g. ``["Naive", "Statistical", "ML", "Override"]``).
        wmape_values: WMAPE at each layer.
        fva_labels: Optional classification for each layer
            (``"ADDS_VALUE"`` / ``"NEUTRAL"`` / ``"DESTROYS_VALUE"`` / ``"BASELINE"``).
        fig: Existing Figure.
        ax: Existing Axes.

    Returns:
        (fig, ax) tuple.
    """
    swd_style()

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)

    if fva_labels is None:
        fva_labels = ["BASELINE"] + [
            "ADDS_VALUE" if wmape_values[i] < wmape_values[i - 1]
            else "DESTROYS_VALUE" if wmape_values[i] > wmape_values[i - 1]
            else "NEUTRAL"
            for i in range(1, len(layers))
        ]

    bar_colors = [FVA_COLORS.get(lbl, COLORS["gray400"]) for lbl in fva_labels]

    bars = ax.bar(layers, wmape_values, color=bar_colors, width=0.6)

    for bar, val, lbl in zip(bars, wmape_values, fva_labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(wmape_values) * 0.02,
            f"{val:.1%}", ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            color=FVA_COLORS.get(lbl, COLORS["gray600"]),
        )

    ax.set_ylim(0, max(wmape_values) * 1.2)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=COLORS["gray100"], linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend for FVA colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=FVA_COLORS["ADDS_VALUE"], label="Adds value"),
        Patch(facecolor=FVA_COLORS["NEUTRAL"], label="Neutral"),
        Patch(facecolor=FVA_COLORS["DESTROYS_VALUE"], label="Destroys value"),
        Patch(facecolor=FVA_COLORS["BASELINE"], label="Baseline"),
    ]
    ax.legend(
        handles=legend_elements, fontsize=9, frameon=False,
        loc="upper right",
    )

    action_title(
        ax,
        "Forecast Value Added by layer",
        subtitle=f"WMAPE at each forecasting stage",
    )

    return fig, ax


def demand_class_chart(
    class_counts: dict,
    title: str = "Demand classification distribution",
    fig=None,
    ax=None,
) -> Tuple[Any, Any]:
    """Horizontal bar chart showing the distribution of demand classes.

    Args:
        class_counts: Dict mapping class name to count
            (e.g. ``{"smooth": 150, "intermittent": 42, ...}``).
        title: Chart title.
        fig: Existing Figure.
        ax: Existing Axes.

    Returns:
        (fig, ax) tuple.
    """
    from src.visualization.chart_palette import DEMAND_CLASS_COLORS

    swd_style()

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    total = sum(counts)

    bar_colors = [DEMAND_CLASS_COLORS.get(c, COLORS["gray400"]) for c in classes]

    bars = ax.barh(classes, counts, color=bar_colors, height=0.5)
    ax.set_xlim(0, max(counts) * 1.25)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)

    for bar, count in zip(bars, counts):
        pct = count / total if total > 0 else 0
        ax.text(
            count + max(counts) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({pct:.0%})", va="center", fontsize=9,
            color=COLORS["gray900"],
        )

    action_title(ax, title)

    return fig, ax
