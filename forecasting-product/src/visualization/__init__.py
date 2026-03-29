"""
Visualization module — SWD-style chart helpers for the Forecasting Platform.

Provides consistent, publication-quality chart builders following
Storytelling with Data (SWD) principles. Both matplotlib (static export)
and Plotly (interactive) helpers are available.

Usage:
    from src.visualization.chart_helpers import (
        swd_style, highlight_bar, highlight_line, action_title,
        forecast_plot, control_chart_plot, save_chart, CHART_FIGSIZE,
    )
    from src.visualization.chart_palette import (
        COLORS, SEVERITY_COLORS, MODEL_LAYER_COLORS, FVA_COLORS,
        palette_for_n, ensure_contrast,
    )
    from src.visualization.plotly_theme import apply_swd_plotly_theme
"""

from src.visualization.chart_palette import COLORS
from src.visualization.chart_helpers import swd_style, save_chart

__all__ = [
    "COLORS",
    "swd_style",
    "save_chart",
]
