"""
SWD Summary Layouts — Big-number dashboard card builder for the Forecasting Platform.

Provides ``big_number_layout()`` for KPI summary cards: large metric numbers,
finding bullets, and recommendation text, with no data axes.
"""

from __future__ import annotations

from src.visualization.chart_palette import COLORS


def big_number_layout(ax, metrics: list, findings=None,
                      recommendation=None, title=None, subtitle=None) -> None:
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
