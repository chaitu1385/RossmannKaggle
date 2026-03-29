"""
SWD Annotations — Action titles, callout annotations, and axis formatting.

Provides helpers for adding narrative text elements and clean annotations
to matplotlib charts following Storytelling with Data principles.
"""

from __future__ import annotations

import logging

import matplotlib.dates as mdates

from src.visualization.chart_palette import COLORS

logger = logging.getLogger(__name__)


def action_title(ax, title: str, subtitle: str | None = None) -> None:
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


def format_date_axis(ax, fmt: str = "%b", axis: str = "x") -> None:
    """Format a date axis with readable labels.

    Args:
        ax: Matplotlib Axes.
        fmt: strftime format string. Default: ``"%b"`` (abbreviated month).
        axis: Which axis to format — ``"x"`` or ``"y"``.
    """
    from datetime import datetime as _dt

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
            new_labels = []
            for lbl in tick_labels:
                try:
                    parsed = _dt.fromisoformat(lbl)
                    new_labels.append(parsed.strftime(fmt))
                except (ValueError, TypeError):
                    new_labels.append(lbl)
            if axis == "x":
                ax.set_xticklabels(new_labels)
            else:
                ax.set_yticklabels(new_labels)
    except Exception:
        logger.debug("Failed to reformat tick labels", exc_info=True)


def annotate_point(ax, x, y, text: str, arrow_color: str | None = None, offset=(20, 20)) -> None:
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
