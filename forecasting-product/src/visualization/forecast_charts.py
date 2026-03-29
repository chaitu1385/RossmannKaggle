"""
SWD Forecast Charts — Time-series forecast visualisations for the Forecasting Platform.

Provides ``forecast_plot()``, ``leaderboard_chart()``, and ``drift_timeline()``
for presenting model output, champion comparisons, and accuracy monitoring.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import matplotlib.pyplot as plt

from src.visualization.chart_palette import COLORS
from src.visualization.style import CHART_FIGSIZE, swd_style
from src.visualization.annotations import action_title
from src.visualization.basic_charts import highlight_bar


def forecast_plot(
    historical,
    forecast,
    title: str | None = None,
    confidence_band=None,
    fig=None,
    ax=None,
) -> Tuple[Any, Any]:
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


def leaderboard_chart(
    model_names: Sequence[str],
    metric_values: Sequence[float],
    metric_name: str = "WMAPE",
    champion: str | None = None,
    fig=None,
    ax=None,
) -> Tuple[Any, Any]:
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
    import numpy as np

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


def drift_timeline(
    dates,
    metric_values: Sequence[float],
    threshold: float | None = None,
    alert_dates=None,
    metric_name: str = "WMAPE",
    fig=None,
    ax=None,
) -> Tuple[Any, Any]:
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
