"""
SWD Composed Charts — Multi-series and overlay chart builders for the Forecasting Platform.

Provides add_trendline, add_event_span, fill_between_lines, and funnel_waterfall
for composed/overlay visualisations following Storytelling with Data principles.
"""

from __future__ import annotations

import numpy as np

from src.visualization.chart_palette import COLORS


def add_trendline(ax, x, y, exclude_indices=None, degree: int = 1,
                  color: str | None = None, label: str = "expected\ntrend") -> np.ndarray:
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
                   color: str | None = None, alpha: float = 0.08) -> None:
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
                       fill_alpha: float = 0.15) -> None:
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
