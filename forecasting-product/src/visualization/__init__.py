"""
Visualization module — SWD-style chart helpers for the Forecasting Platform.

Provides consistent, publication-quality chart builders following
Storytelling with Data (SWD) principles. Both matplotlib (static export)
and Plotly (interactive) helpers are available.

Usage:
    from src.visualization import (
        swd_style, CHART_FIGSIZE,
        highlight_bar, highlight_line,
        action_title, format_date_axis, annotate_point,
        stacked_bar, funnel_waterfall, grouped_bar,
        add_trendline, add_event_span, fill_between_lines,
        control_chart_plot,
        big_number_layout,
        forecast_plot, leaderboard_chart, drift_timeline,
        fva_cascade_chart, demand_class_chart,
        save_chart,
    )
    from src.visualization.chart_palette import (
        COLORS, SEVERITY_COLORS, MODEL_LAYER_COLORS, FVA_COLORS,
        palette_for_n, ensure_contrast,
    )
    from src.visualization.plotly_theme import apply_swd_plotly_theme
"""

from src.visualization.annotations import action_title, annotate_point, format_date_axis
from src.visualization.basic_charts import (
    funnel_waterfall,
    grouped_bar,
    highlight_bar,
    highlight_line,
    stacked_bar,
)
from src.visualization.chart_palette import COLORS
from src.visualization.composed_charts import (
    add_event_span,
    add_trendline,
    fill_between_lines,
)
from src.visualization.forecast_charts import (
    drift_timeline,
    forecast_plot,
    leaderboard_chart,
)
from src.visualization.overlays import control_chart_plot
from src.visualization.platform_analytics import (
    demand_class_chart,
    fva_cascade_chart,
)
from src.visualization.save import save_chart
from src.visualization.style import CHART_FIGSIZE, swd_style
from src.visualization.summary_layouts import big_number_layout

__all__ = [
    # palette
    "COLORS",
    # style
    "swd_style",
    "CHART_FIGSIZE",
    # annotations
    "action_title",
    "format_date_axis",
    "annotate_point",
    # basic charts
    "highlight_bar",
    "highlight_line",
    "stacked_bar",
    "funnel_waterfall",
    "grouped_bar",
    # composed charts
    "add_trendline",
    "add_event_span",
    "fill_between_lines",
    # overlays
    "control_chart_plot",
    # summary layouts
    "big_number_layout",
    # forecast charts
    "forecast_plot",
    "leaderboard_chart",
    "drift_timeline",
    # platform analytics
    "fva_cascade_chart",
    "demand_class_chart",
    # save
    "save_chart",
]
