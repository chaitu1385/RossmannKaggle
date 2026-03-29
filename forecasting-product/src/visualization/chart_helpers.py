"""
SWD Chart Helpers — Storytelling with Data chart builders for the Forecasting Platform.

Reusable functions for creating publication-quality matplotlib charts following
SWD principles: minimal chrome, direct labels, focus-vs-context contrast,
action titles, and proper color encoding.

Includes general-purpose builders (bars, lines, heatmaps) plus
forecasting-specific builders (forecast fan charts, FVA cascades,
leaderboard bars, drift timelines, control charts).

Usage:
    from src.visualization.chart_helpers import (
        swd_style, highlight_bar, highlight_line, action_title,
        format_date_axis, annotate_point, save_chart, CHART_FIGSIZE,
        stacked_bar, add_trendline, add_event_span, fill_between_lines,
        big_number_layout, funnel_waterfall, grouped_bar,
        forecast_plot, control_chart_plot,
        fva_cascade_chart, leaderboard_chart, drift_timeline,
    )

    colors = swd_style()
    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    highlight_bar(ax, categories, values, highlight="LightGBM")
    action_title(ax, "LightGBM achieves lowest WMAPE across all series")
    save_chart(fig, "leaderboard.png")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.visualization.chart_palette import (
    COLORS, FVA_COLORS, MODEL_LAYER_COLORS, CATEGORICAL_PALETTE,
)

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Core chart builders — ported from AI Analyst with forecasting adaptations
# ---------------------------------------------------------------------------

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
):
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
):
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


def action_title(ax, title: str, subtitle: str | None = None):
    """Add a bold action title and optional subtle subtitle.

    Args:
        ax: Matplotlib Axes.
        title: The takeaway statement (e.g. "LightGBM wins 62% of series").
        subtitle: Context line (e.g. "Backtest results, WMAPE by model").
    """
    if subtitle:
        ax.text(
            0, 1.12, title, transform=ax.transAxes,
            fontsize=17, fontweight="bold", color=COLORS["gray900"],
            va="bottom", ha="left",
        )
        ax.text(
            0, 1.06, subtitle, transform=ax.transAxes,
            fontsize=12, color=COLORS["gray600"], va="bottom", ha="left",
        )
        ax.set_title("")
    else:
        ax.set_title(
            title, fontsize=17, fontweight="bold",
            color=COLORS["gray900"], loc="left", pad=16,
        )


def format_date_axis(ax, fmt: str = "%b", axis: str = "x"):
    """Format a date axis with readable labels.

    Args:
        ax: Matplotlib Axes.
        fmt: strftime format string. Default: ``"%b"`` (abbreviated month).
        axis: Which axis to format — ``"x"`` or ``"y"``.
    """
    import pandas as pd

    target = ax.xaxis if axis == "x" else ax.yaxis

    if isinstance(target.get_major_formatter(), mdates.DateFormatter):
        target.set_major_formatter(mdates.DateFormatter(fmt))
        return

    try:
        target.set_major_formatter(mdates.DateFormatter(fmt))
        ax.figure.canvas.draw()
        labels = [t.get_text() for t in target.get_ticklabels() if t.get_text().strip()]
        if labels:
            return
    except Exception:
        logger.debug("Failed to apply date formatter to axis", exc_info=True)

    try:
        tick_labels = [t.get_text() for t in target.get_ticklabels()]
        if tick_labels and any(tick_labels):
            parsed = pd.to_datetime(tick_labels, errors="coerce")
            new_labels = [
                d.strftime(fmt) if pd.notna(d) else lbl
                for d, lbl in zip(parsed, tick_labels)
            ]
            if axis == "x":
                ax.set_xticklabels(new_labels)
            else:
                ax.set_yticklabels(new_labels)
    except Exception:
        logger.debug("Failed to reformat tick labels", exc_info=True)


def annotate_point(ax, x, y, text: str, arrow_color: str | None = None, offset=(20, 20)):
    """Add a clean annotation with an arrow to a specific data point.

    Args:
        ax: Matplotlib Axes.
        x: X-coordinate of the data point.
        y: Y-coordinate of the data point.
        text: Annotation text.
        arrow_color: Arrow/text color. Default: gray600.
        offset: (dx, dy) offset in points.
    """
    arrow_color = arrow_color or COLORS["gray600"]
    ax.annotate(
        text, xy=(x, y), xytext=offset, textcoords="offset points",
        fontsize=9, color=arrow_color,
        arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.0),
    )


def save_chart(fig, path, dpi: int = 150, close: bool = True):
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


# ---------------------------------------------------------------------------
# Advanced chart builders
# ---------------------------------------------------------------------------

def stacked_bar(ax, categories, layers: dict, colors_map=None,
                highlight_layer=None, show_totals: bool = True,
                fmt=None, normalize: bool = False, sort_by=None):
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


def add_trendline(ax, x, y, exclude_indices=None, degree: int = 1,
                  color: str | None = None, label: str = "expected\ntrend"):
    """Fit and draw a trend line, optionally excluding outlier indices.

    Args:
        ax: Matplotlib Axes.
        x: Sequence of x-values (numeric).
        y: Sequence of y-values.
        exclude_indices: List of integer indices to exclude from the fit.
        degree: Polynomial degree. Default: 1 (linear).
        color: Line color. Default: gray400.
        label: End-of-line label. Set to None to suppress.

    Returns:
        np.ndarray: Fitted trend-line y-values.
    """
    color = color or COLORS["gray400"]
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if exclude_indices:
        mask = np.ones(len(x_arr), dtype=bool)
        for idx in exclude_indices:
            mask[idx] = False
        z = np.polyfit(x_arr[mask], y_arr[mask], degree)
    else:
        z = np.polyfit(x_arr, y_arr, degree)

    trend_vals = np.polyval(z, x_arr)
    ax.plot(x_arr, trend_vals, color=color, linewidth=1, linestyle="--", zorder=0)

    if label:
        ax.text(
            x_arr[-1] + (x_arr[-1] - x_arr[0]) * 0.03, trend_vals[-1],
            label, fontsize=8, color=color, va="center",
        )

    return trend_vals


def add_event_span(ax, start, end, label: str | None = None,
                   color: str | None = None, alpha: float = 0.08):
    """Highlight a time window with a shaded span and boundary lines.

    Args:
        ax: Matplotlib Axes.
        start: Left boundary.
        end: Right boundary.
        label: Optional label positioned above the span center.
        color: Span color. Default: accent.
        alpha: Background fill opacity.
    """
    color = color or COLORS["accent"]
    ax.axvspan(start, end, alpha=alpha, color=color, zorder=0)
    ax.axvline(start, color=color, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(end, color=color, linewidth=0.8, linestyle="--", alpha=0.5)

    if label:
        mid = start + (end - start) / 2 if hasattr(start, "__add__") else (start + end) / 2
        y_top = ax.get_ylim()[1]
        ax.text(
            mid, y_top * 0.97, label, ha="center", va="top",
            fontsize=9, color=color, fontstyle="italic",
        )


def fill_between_lines(ax, x, y1, y2, label1=None, label2=None,
                       color1=None, color2=None, fill_color=None,
                       fill_alpha: float = 0.15):
    """Draw two lines with shaded area between them.

    Args:
        ax: Matplotlib Axes.
        x: Shared x-axis values.
        y1: Y-values for the first (upper) line.
        y2: Y-values for the second (lower) line.
        label1: End-of-line label for y1.
        label2: End-of-line label for y2.
        color1: Color for line 1. Default: primary.
        color2: Color for line 2. Default: gray400.
        fill_color: Color of the shaded region. Default: primary.
        fill_alpha: Opacity of the fill.
    """
    color1 = color1 or COLORS["primary"]
    color2 = color2 or COLORS["gray400"]
    fill_color = fill_color or COLORS["primary"]

    ax.plot(x, y1, color=color1, linewidth=2)
    ax.plot(x, y2, color=color2, linewidth=1.5, linestyle="--")
    ax.fill_between(x, y1, y2, color=fill_color, alpha=fill_alpha)

    if label1:
        ax.text(x[-1], y1[-1], f"  {label1}", va="center", fontsize=9,
                fontweight="bold", color=color1)
    if label2:
        ax.text(x[-1], y2[-1], f"  {label2}", va="center", fontsize=9,
                color=color2)

    ax.yaxis.grid(True, color=COLORS["gray200"], linewidth=0.5)
    ax.set_axisbelow(True)


def big_number_layout(ax, metrics: list, findings=None,
                      recommendation=None, title=None, subtitle=None):
    """Render a big-number summary card — no data axes, just KPIs and text.

    Args:
        ax: Matplotlib Axes (will be turned off).
        metrics: List of tuples: ``(big_number_str, label_str, color)``.
        findings: Optional list of bullet-point strings.
        recommendation: Optional recommendation string.
        title: Optional title.
        subtitle: Optional subtitle.
    """
    ax.axis("off")

    y_cursor = 0.95
    if title:
        ax.text(
            0.5, y_cursor, title, fontsize=18, fontweight="bold",
            color=COLORS["gray900"], ha="center", va="top",
            transform=ax.transAxes,
        )
        y_cursor -= 0.06
    if subtitle:
        ax.text(
            0.5, y_cursor, subtitle, fontsize=11, color=COLORS["gray600"],
            ha="center", va="top", transform=ax.transAxes,
        )
        y_cursor -= 0.05

    ax.plot(
        [0.1, 0.9], [y_cursor, y_cursor], color=COLORS["gray200"],
        linewidth=1, transform=ax.transAxes, clip_on=False,
    )
    y_cursor -= 0.05

    n = len(metrics)
    num_fontsize = 28 if n >= 4 else 36
    label_fontsize = 10 if n >= 4 else 11
    label_offset = 0.14 if n >= 4 else 0.16
    row_height = 0.30 if n >= 4 else 0.32
    spacing = 0.8 / max(n, 1)

    for i, (big_num, label, color) in enumerate(metrics):
        x_pos = 0.1 + spacing / 2 + i * spacing
        ax.text(
            x_pos, y_cursor - 0.02, big_num, fontsize=num_fontsize,
            fontweight="bold", color=color, ha="center", va="center",
            transform=ax.transAxes,
        )
        ax.text(
            x_pos, y_cursor - label_offset, label, fontsize=label_fontsize,
            color=COLORS["gray600"], ha="center", va="center",
            transform=ax.transAxes, linespacing=1.4,
        )
    y_cursor -= row_height

    ax.plot(
        [0.1, 0.9], [y_cursor, y_cursor], color=COLORS["gray200"],
        linewidth=1, transform=ax.transAxes, clip_on=False,
    )
    y_cursor -= 0.04

    if findings:
        ax.text(
            0.12, y_cursor, "Key Findings", fontsize=13,
            fontweight="bold", color=COLORS["gray900"], va="top",
            transform=ax.transAxes,
        )
        y_cursor -= 0.06
        for finding in findings:
            ax.text(
                0.14, y_cursor, f"\u2022  {finding}", fontsize=10,
                color=COLORS["gray600"], va="top", transform=ax.transAxes,
            )
            y_cursor -= 0.055

    if recommendation:
        y_cursor -= 0.02
        ax.text(
            0.12, y_cursor, "Recommendation", fontsize=13,
            fontweight="bold", color=COLORS["primary"], va="top",
            transform=ax.transAxes,
        )
        y_cursor -= 0.06
        ax.text(
            0.14, y_cursor, recommendation, fontsize=10,
            color=COLORS["primary"], va="top", transform=ax.transAxes,
        )


def funnel_waterfall(ax, steps, counts, highlight_step=None,
                     bar_color=None, highlight_color=None, fmt=None):
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
                xlabel=None, figsize=(10, 6)):
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

    groups = df[group_col].unique()
    categories = df[x_col].unique()
    n_groups = len(groups)
    n_cats = len(categories)

    bar_width = 0.7 / n_groups
    gap = bar_width * 0.1
    x_indices = np.arange(n_cats)

    for i, group in enumerate(groups):
        group_data = df[df[group_col] == group]
        val_map = dict(zip(group_data[x_col], group_data[y_col]))
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


# ---------------------------------------------------------------------------
# Forecasting-specific chart builders
# ---------------------------------------------------------------------------

def forecast_plot(
    historical,
    forecast,
    title: str | None = None,
    confidence_band=None,
    fig=None,
    ax=None,
):
    """Time-series chart with historical actuals and dashed forecast line.

    Historical data is rendered as a solid line, forecast as dashed, with an
    optional shaded confidence band. A vertical boundary line marks where
    actuals end and the forecast begins.

    Args:
        historical: pd.Series with DatetimeIndex — actual observed values.
        forecast: pd.Series with DatetimeIndex — forecasted values.
        title: Chart title. Default: ``"Forecast: {series_name}"``.
        confidence_band: Optional tuple ``(lower, upper)`` of pd.Series.
        fig: Existing Figure. If None, creates new.
        ax: Existing Axes. If None, creates new.

    Returns:
        (fig, ax) tuple.
    """
    swd_style()

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)

    ax.plot(
        historical.index, historical.values, color=COLORS["primary"],
        linewidth=2, label="Actual",
    )

    ax.plot(
        forecast.index, forecast.values, color=COLORS["primary"],
        linewidth=2, linestyle="--", alpha=0.7, label="Forecast",
    )

    if confidence_band is not None:
        lower, upper = confidence_band
        ax.fill_between(
            forecast.index, lower.values, upper.values,
            color=COLORS["primary"], alpha=0.15, label="Confidence band",
        )

    boundary = historical.index[-1]
    y_min, y_max = ax.get_ylim()
    ax.axvline(boundary, color=COLORS["muted"], linewidth=1, linestyle="--", zorder=0)
    ax.text(
        boundary, y_max, "  Forecast \u00BB", va="top", ha="left",
        fontsize=9, color=COLORS["muted"],
    )

    ax.yaxis.grid(True, color=COLORS["gray200"], linewidth=0.5)
    ax.set_axisbelow(True)

    series_name = getattr(historical, "name", None) or "series"
    chart_title = title or f"Forecast: {series_name}"
    action_title(ax, chart_title)

    return fig, ax


def control_chart_plot(
    series,
    center_line,
    ucl,
    lcl,
    violations=None,
    title: str | None = None,
    fig=None,
    ax=None,
):
    """Shewhart control chart with center line, limits, and violations.

    Plots a metric over time with SPC overlays: center line, upper/lower
    control limits, in-control band, and optional violation markers.

    Args:
        series: pd.Series with DatetimeIndex — metric values.
        center_line: float or pd.Series — center line value(s).
        ucl: float or pd.Series — upper control limit.
        lcl: float or pd.Series — lower control limit.
        violations: Optional list of dicts with keys: ``index``, ``value``,
            ``rule``, ``description``.
        title: Chart title. Default: ``"Control Chart: {series_name}"``.
        fig: Existing Figure.
        ax: Existing Axes.

    Returns:
        (fig, ax) tuple.
    """
    swd_style()

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)

    ax.plot(
        series.index, series.values, color=COLORS["primary"],
        linewidth=1.8, label="Value", zorder=2,
    )

    if isinstance(center_line, (int, float)):
        ax.axhline(
            center_line, color=COLORS["muted"], linewidth=1,
            linestyle="--", label="Center", zorder=1,
        )
    else:
        ax.plot(
            center_line.index, center_line.values, color=COLORS["muted"],
            linewidth=1, linestyle="--", label="Center", zorder=1,
        )

    if isinstance(ucl, (int, float)):
        ax.axhline(
            ucl, color=COLORS["danger"], alpha=0.5, linewidth=1,
            linestyle=":", label="UCL/LCL", zorder=1,
        )
        ucl_vals = ucl
    else:
        ax.plot(
            ucl.index, ucl.values, color=COLORS["danger"], alpha=0.5,
            linewidth=1, linestyle=":", label="UCL/LCL", zorder=1,
        )
        ucl_vals = ucl.values

    if isinstance(lcl, (int, float)):
        ax.axhline(
            lcl, color=COLORS["danger"], alpha=0.5, linewidth=1,
            linestyle=":", zorder=1,
        )
        lcl_vals = lcl
    else:
        ax.plot(
            lcl.index, lcl.values, color=COLORS["danger"], alpha=0.5,
            linewidth=1, linestyle=":", zorder=1,
        )
        lcl_vals = lcl.values

    if isinstance(ucl_vals, (int, float)) and isinstance(lcl_vals, (int, float)):
        ax.axhspan(lcl_vals, ucl_vals, color=COLORS["muted"], alpha=0.08, zorder=0)
    else:
        _ucl = ucl_vals if not isinstance(ucl_vals, (int, float)) else np.full(len(series), ucl_vals)
        _lcl = lcl_vals if not isinstance(lcl_vals, (int, float)) else np.full(len(series), lcl_vals)
        ax.fill_between(
            series.index, _lcl, _ucl, color=COLORS["muted"],
            alpha=0.08, zorder=0,
        )

    if violations:
        v_x = [v["index"] for v in violations]
        v_y = [v["value"] for v in violations]
        ax.scatter(
            v_x, v_y, color=COLORS["danger"], marker="o", s=60,
            zorder=5, label="Violations",
        )

    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.yaxis.grid(True, color=COLORS["gray200"], linewidth=0.5)
    ax.set_axisbelow(True)

    series_name = getattr(series, "name", None) or "metric"
    chart_title = title or f"Control Chart: {series_name}"
    action_title(ax, chart_title)

    return fig, ax


# ---------------------------------------------------------------------------
# Forecasting Platform-specific chart builders
# ---------------------------------------------------------------------------

def leaderboard_chart(
    model_names: Sequence[str],
    metric_values: Sequence[float],
    metric_name: str = "WMAPE",
    champion: str | None = None,
    fig=None,
    ax=None,
):
    """Horizontal bar chart ranking models by a metric (lower is better).

    Champion model is highlighted. Others are gray context.

    Args:
        model_names: Model display names.
        metric_values: Metric values (e.g. WMAPE scores).
        metric_name: Name of the metric for the subtitle.
        champion: Model name to highlight as champion.
        fig: Existing Figure.
        ax: Existing Axes.

    Returns:
        (fig, ax) tuple.
    """
    swd_style()

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)

    champion = champion or model_names[np.argmin(metric_values)]

    highlight_bar(
        ax, model_names, metric_values,
        highlight=champion,
        highlight_color=COLORS["success"],
        base_color=COLORS["gray200"],
        horizontal=True,
        sort=True,
        fmt="{:.1%}" if max(metric_values) <= 2 else "{:,.2f}",
    )

    action_title(
        ax,
        f"{champion} achieves lowest {metric_name}",
        subtitle=f"Model comparison — {metric_name} (lower is better)",
    )

    return fig, ax


def fva_cascade_chart(
    layers: Sequence[str],
    wmape_values: Sequence[float],
    fva_labels: Sequence[str] | None = None,
    fig=None,
    ax=None,
):
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


def drift_timeline(
    dates,
    metric_values: Sequence[float],
    threshold: float | None = None,
    alert_dates=None,
    metric_name: str = "WMAPE",
    fig=None,
    ax=None,
):
    """Time-series plot of a metric with optional drift threshold and alerts.

    Args:
        dates: Sequence of datetime values (x-axis).
        metric_values: Metric values over time.
        threshold: Optional horizontal threshold line.
        alert_dates: Optional list of dates where drift was detected.
        metric_name: Name of the metric.
        fig: Existing Figure.
        ax: Existing Axes.

    Returns:
        (fig, ax) tuple.
    """
    swd_style()

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)

    ax.plot(
        dates, metric_values, color=COLORS["primary"],
        linewidth=2, marker="o", markersize=4, label=metric_name,
    )

    if threshold is not None:
        ax.axhline(
            threshold, color=COLORS["danger"], linewidth=1.5,
            linestyle="--", alpha=0.7, label=f"Threshold ({threshold})",
        )

    if alert_dates:
        alert_vals = []
        for ad in alert_dates:
            for d, v in zip(dates, metric_values):
                if d == ad:
                    alert_vals.append(v)
                    break
            else:
                alert_vals.append(None)

        valid_x = [d for d, v in zip(alert_dates, alert_vals) if v is not None]
        valid_y = [v for v in alert_vals if v is not None]
        if valid_x:
            ax.scatter(
                valid_x, valid_y, color=COLORS["danger"],
                marker="X", s=100, zorder=5, label="Drift alert",
            )

    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.yaxis.grid(True, color=COLORS["gray200"], linewidth=0.5)
    ax.set_axisbelow(True)

    action_title(
        ax,
        f"{metric_name} stability over time",
        subtitle="Monitoring forecast accuracy drift",
    )

    return fig, ax


def demand_class_chart(
    class_counts: dict,
    title: str = "Demand classification distribution",
    fig=None,
    ax=None,
):
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
