"""
SWD Basic Charts — Core chart builders for the Forecasting Platform.

Provides highlight bar, highlight line, stacked bar, grouped bar,
and funnel/waterfall chart builders following SWD design principles.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.visualization.chart_palette import COLORS, CATEGORICAL_PALETTE
from src.visualization.annotations import action_title


def highlight_bar(
    ax,
    categories: Sequence,
    values: Sequence[float],
    highlight=None,
    highlight_color: str | None = None,
    base_color: str | None = None,
    horizontal: bool = True,
    sort: bool = True,
    fmt: str | None = None,
    label_offset: float = 0.02,
) -> None:
    """Bar chart with one bar highlighted, the rest gray.

    Args:
        ax: Matplotlib Axes.
        categories: Sequence of category labels.
        values: Sequence of numeric values.
        highlight: Category label to highlight (or list of labels).
        highlight_color: Hex color for highlighted bar(s). Default: primary.
        base_color: Hex color for non-highlighted bars. Default: gray200.
        horizontal: If True (default), draw horizontal bars.
        sort: If True (default), sort bars by value.
        fmt: Format string for value labels (e.g. ``"{:,.0f}"`` or ``"{:.1%}"``).
        label_offset: Fraction of max value used to offset labels from bars.
    """
    highlight_color = highlight_color or COLORS["primary"]
    base_color = base_color or COLORS["gray200"]

    cats = list(categories)
    vals = list(values)

    if sort:
        paired = sorted(zip(vals, cats), reverse=False)
        vals, cats = zip(*paired)
        vals, cats = list(vals), list(cats)

    if isinstance(highlight, str):
        highlight = [highlight]
    highlight_set = set(highlight) if highlight else set()

    bar_colors = [
        highlight_color if c in highlight_set else base_color for c in cats
    ]

    if horizontal:
        bars = ax.barh(cats, vals, color=bar_colors)
        ax.set_xlim(0, max(vals) * 1.15)
        ax.xaxis.set_visible(False)
        ax.spines["bottom"].set_visible(False)
        max_val = max(vals)
        for bar, v in zip(bars, vals):
            label = fmt.format(v) if fmt else f"{v:,.0f}"
            ax.text(
                v + max_val * label_offset,
                bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9, color=COLORS["gray900"],
            )
    else:
        bars = ax.bar(cats, vals, color=bar_colors)
        ax.set_ylim(0, max(vals) * 1.15)
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)
        max_val = max(vals)
        for bar, v in zip(bars, vals):
            label = fmt.format(v) if fmt else f"{v:,.0f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + max_val * label_offset,
                label, ha="center", fontsize=9, color=COLORS["gray900"],
            )

    ax.grid(False)


def highlight_line(
    ax,
    x,
    y_dict: dict,
    highlight=None,
    highlight_color: str | None = None,
    base_color: str | None = None,
    linewidth_highlight: float = 2.5,
    linewidth_base: float = 1.2,
) -> None:
    """Line chart with one line colored, the rest gray.

    Args:
        ax: Matplotlib Axes.
        x: Shared x-axis values (e.g. dates).
        y_dict: Dict mapping ``series_name -> y_values``.
        highlight: Series name to highlight (or list).
        highlight_color: Hex color for highlighted line(s). Default: primary.
        base_color: Hex color for non-highlighted lines. Default: gray200.
        linewidth_highlight: Line width for highlighted series.
        linewidth_base: Line width for background series.
    """
    highlight_color = highlight_color or COLORS["primary"]
    base_color = base_color or COLORS["gray200"]

    if isinstance(highlight, str):
        highlight = [highlight]
    highlight_set = set(highlight) if highlight else set()

    for name, y in y_dict.items():
        if name not in highlight_set:
            ax.plot(x, y, color=base_color, linewidth=linewidth_base, zorder=1)
            ax.text(
                x[-1], y[-1], f"  {name}",
                va="center", fontsize=8, color=COLORS["gray400"],
            )

    for name, y in y_dict.items():
        if name in highlight_set:
            ax.plot(x, y, color=highlight_color, linewidth=linewidth_highlight, zorder=2)
            ax.text(
                x[-1], y[-1], f"  {name}",
                va="center", fontsize=9, fontweight="bold", color=highlight_color,
            )

    ax.yaxis.grid(True, color=COLORS["gray200"], linewidth=0.5)
    ax.set_axisbelow(True)


def stacked_bar(ax, categories, layers: dict, colors_map=None,
                highlight_layer=None, show_totals: bool = True,
                fmt=None, normalize: bool = False, sort_by=None) -> None:
    """Stacked bar chart with one layer optionally highlighted.

    Args:
        ax: Matplotlib Axes.
        categories: Sequence of category labels.
        layers: Dict mapping ``layer_name -> sequence_of_values``.
        colors_map: Optional dict mapping layer_name -> hex color.
        highlight_layer: Layer name to highlight.
        show_totals: If True, show total above each stack.
        fmt: Format string for labels.
        normalize: If True, normalize to 100%.
        sort_by: Layer name whose values determine category sort order.
    """
    cats = list(categories)

    if sort_by is not None and sort_by in layers:
        sort_vals = list(layers[sort_by])
        sort_order = sorted(range(len(cats)), key=lambda i: sort_vals[i], reverse=True)
        cats = [cats[i] for i in sort_order]
        layers = {
            name: [list(vals)[i] for i in sort_order]
            for name, vals in layers.items()
        }

    bottom = np.zeros(len(cats))

    if normalize:
        totals = sum(np.array(v, dtype=float) for v in layers.values())
        fmt = fmt or "{:.0%}"
    else:
        totals = None
        fmt = fmt or "{:,.0f}"

    for name, values in layers.items():
        vals = np.array(values, dtype=float)
        if normalize:
            vals = vals / totals

        if colors_map and name in colors_map:
            color = colors_map[name]
        elif name == highlight_layer:
            color = COLORS["accent"]
        else:
            color = COLORS["gray200"]

        bars = ax.bar(cats, vals, bottom=bottom, color=color, width=0.7, label=name)

        if name == highlight_layer:
            for bar, v in zip(bars, vals):
                if v > 0:
                    label_text = f"{v:.0%}" if normalize else fmt.format(v)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        label_text, ha="center", va="center",
                        fontsize=9, fontweight="bold", color=COLORS["white"],
                    )

        bottom += vals

    if normalize:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    if show_totals:
        if normalize and totals is not None:
            for i, total in enumerate(totals):
                ax.text(
                    i, bottom[i] + max(bottom) * 0.02,
                    "{:,.0f}".format(total),
                    ha="center", fontsize=9, color=COLORS["gray600"],
                )
        else:
            for i, total in enumerate(bottom):
                ax.text(
                    i, total + max(bottom) * 0.02, fmt.format(total),
                    ha="center", fontsize=9, color=COLORS["gray600"],
                )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=COLORS["gray100"], linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        fontsize=9, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, -0.08), ncol=min(len(layers), 5),
    )


def funnel_waterfall(ax, steps, counts, highlight_step=None,
                     bar_color=None, highlight_color=None, fmt=None) -> None:
    """Render a funnel as a horizontal waterfall showing drop-off at each step.

    Args:
        ax: Matplotlib Axes.
        steps: Sequence of step labels.
        counts: Sequence of counts at each step (monotonically decreasing).
        highlight_step: Index of the step to highlight.
        bar_color: Non-highlighted bar color. Default: gray200.
        highlight_color: Highlighted step color. Default: accent.
        fmt: Format string for count labels.
    """
    bar_color = bar_color or COLORS["gray200"]
    highlight_color = highlight_color or COLORS["accent"]
    fmt = fmt or "{:,.0f}"

    n = len(steps)
    counts = list(counts)

    if highlight_step is None:
        drops = [counts[i] - counts[i + 1] for i in range(n - 1)]
        highlight_step = drops.index(max(drops)) + 1

    y_positions = list(range(n - 1, -1, -1))
    bar_colors = [
        highlight_color if i == highlight_step else bar_color
        for i in range(n)
    ]

    bars = ax.barh(y_positions, counts, color=bar_colors, height=0.6)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(steps)
    ax.tick_params(axis="y", length=0)

    max_val = max(counts)
    for bar, count in zip(bars, counts):
        ax.text(
            count + max_val * 0.02, bar.get_y() + bar.get_height() / 2,
            fmt.format(count), va="center", fontsize=9, color=COLORS["gray900"],
        )

    for i in range(n - 1):
        if counts[i] > 0:
            conv_rate = counts[i + 1] / counts[i]
            drop_rate = 1 - conv_rate
            y_mid = (y_positions[i] + y_positions[i + 1]) / 2
            x_pos = max(counts[i], counts[i + 1]) + max_val * 0.12

            label_color = highlight_color if (i + 1) == highlight_step else COLORS["gray600"]
            fontweight = "bold" if (i + 1) == highlight_step else "normal"

            ax.text(
                x_pos, y_mid, f"{conv_rate:.0%} pass\n{drop_rate:.0%} drop",
                va="center", ha="center", fontsize=8,
                color=label_color, fontweight=fontweight,
            )

    ax.set_xlim(0, max_val * 1.35)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def grouped_bar(df, x_col: str, y_col: str, group_col: str,
                highlight_group=None, title=None, ylabel=None,
                xlabel=None, figsize=(10, 6)) -> Tuple[Any, Any]:
    """Create a grouped bar chart comparing values across categories.

    Args:
        df: DataFrame with x, y, and group columns.
        x_col: Column for x-axis categories.
        y_col: Column for bar heights.
        group_col: Column for grouping.
        highlight_group: Optional group name to highlight.
        title: Chart title.
        ylabel: Y-axis label.
        xlabel: X-axis label.
        figsize: Figure size tuple.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(figsize=figsize)

    import polars as pl
    if not isinstance(df, pl.DataFrame):
        df = pl.DataFrame(df)  # Accept dict / pandas as convenience

    groups = df.get_column(group_col).unique().sort().to_list()
    categories = df.get_column(x_col).unique().sort().to_list()
    n_groups = len(groups)
    n_cats = len(categories)

    bar_width = 0.7 / n_groups
    gap = bar_width * 0.1
    x_indices = np.arange(n_cats)

    for i, group in enumerate(groups):
        group_data = df.filter(pl.col(group_col) == group)
        val_map = dict(zip(
            group_data.get_column(x_col).to_list(),
            group_data.get_column(y_col).to_list(),
        ))
        vals = [val_map.get(cat, 0) for cat in categories]

        offset = (i - (n_groups - 1) / 2) * (bar_width + gap)

        if highlight_group is not None:
            color = COLORS["primary"] if group == highlight_group else COLORS["gray200"]
        else:
            color = CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]

        bars = ax.bar(
            x_indices + offset, vals, width=bar_width,
            color=color, label=str(group),
        )

        for bar, v in zip(bars, vals):
            if v > 0:
                label_color = (
                    COLORS["gray900"]
                    if highlight_group is None or group == highlight_group
                    else COLORS["gray400"]
                )
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{v:,.0f}", ha="center", va="bottom",
                    fontsize=8, color=label_color,
                )

    ax.set_xticks(x_indices)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)

    if title:
        action_title(ax, title)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=COLORS["gray600"])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=COLORS["gray600"])

    ax.legend(fontsize=9, frameon=False, loc="upper right", ncol=min(n_groups, 4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=COLORS["gray100"], linewidth=0.5)
    ax.set_axisbelow(True)

    return fig, ax
