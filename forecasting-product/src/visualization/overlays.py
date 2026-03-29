"""
SWD Overlays — Control chart and band overlay builders for the Forecasting Platform.

Provides ``control_chart_plot()`` for Shewhart SPC charts with center line,
control limits, in-control band, and optional violation markers.
"""

from __future__ import annotations

from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.chart_palette import COLORS
from src.visualization.style import CHART_FIGSIZE, swd_style
from src.visualization.annotations import action_title


def control_chart_plot(
    series,
    center_line,
    ucl,
    lcl,
    violations=None,
    title: str | None = None,
    fig=None,
    ax=None,
) -> Tuple[Any, Any]:
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
