"""
SWD Chart Helpers — Backward-compatibility re-export shim.

All chart builders have been moved to dedicated submodules. This module
re-exports every public name so that existing imports continue to work
without modification.

    from src.visualization.chart_helpers import (
        swd_style, highlight_bar, highlight_line, action_title,
        format_date_axis, annotate_point, save_chart, CHART_FIGSIZE,
        stacked_bar, add_trendline, add_event_span, fill_between_lines,
        big_number_layout, funnel_waterfall, grouped_bar,
        forecast_plot, control_chart_plot,
        fva_cascade_chart, leaderboard_chart, drift_timeline,
        demand_class_chart,
    )

New code should import directly from the relevant submodule instead.
"""

from src.visualization.style import (
    swd_style,
    CHART_FIGSIZE,
)
from src.visualization.annotations import (
    action_title,
    format_date_axis,
    annotate_point,
)
from src.visualization.basic_charts import (
    highlight_bar,
    highlight_line,
    stacked_bar,
    funnel_waterfall,
    grouped_bar,
)
from src.visualization.composed_charts import (
    add_trendline,
    add_event_span,
    fill_between_lines,
)
from src.visualization.overlays import (
    control_chart_plot,
)
from src.visualization.summary_layouts import (
    big_number_layout,
)
from src.visualization.forecast_charts import (
    forecast_plot,
    leaderboard_chart,
    drift_timeline,
)
from src.visualization.platform_analytics import (
    fva_cascade_chart,
    demand_class_chart,
)
from src.visualization.save import (
    save_chart,
)

__all__ = [
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
