"""
Plotly theme — SWD-style configuration for all Plotly charts.

Provides a consistent visual language across the frontend and reports,
matching the matplotlib SWD style used for static export/reports.

Usage:
    from src.visualization.plotly_theme import (
        apply_swd_plotly_theme, swd_plotly_layout, SWD_PLOTLY_TEMPLATE,
    )

    # Apply globally at app startup
    apply_swd_plotly_theme()

    # Or per-figure
    fig.update_layout(**swd_plotly_layout())

    # Or use the template name
    fig = px.bar(df, template=SWD_PLOTLY_TEMPLATE)
"""

from __future__ import annotations

from src.visualization.chart_palette import (
    COLORS, CATEGORICAL_PALETTE,
)

SWD_PLOTLY_TEMPLATE = "swd_forecasting"


def swd_plotly_layout(**overrides) -> dict:
    """Return a Plotly layout dict matching SWD style guidelines.

    Pass additional keyword arguments to override any setting.

    Returns:
        dict: Plotly layout-compatible dict.
    """
    layout = {
        "font": {
            "family": "Helvetica Neue, Helvetica, Arial, sans-serif",
            "size": 12,
            "color": COLORS["gray900"],
        },
        "title": {
            "font": {"size": 17, "color": COLORS["gray900"]},
            "x": 0,
            "xanchor": "left",
        },
        "paper_bgcolor": COLORS["bg"],
        "plot_bgcolor": COLORS["bg"],
        "colorway": CATEGORICAL_PALETTE,
        "xaxis": {
            "showgrid": False,
            "zeroline": False,
            "showline": False,
            "tickfont": {"size": 10, "color": COLORS["gray600"]},
            "title_font": {"size": 11, "color": COLORS["gray600"]},
        },
        "yaxis": {
            "showgrid": True,
            "gridcolor": COLORS["gray200"],
            "gridwidth": 0.5,
            "zeroline": False,
            "showline": False,
            "tickfont": {"size": 10, "color": COLORS["gray600"]},
            "title_font": {"size": 11, "color": COLORS["gray600"]},
        },
        "legend": {
            "font": {"size": 10},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
        },
        "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
        "hoverlabel": {
            "bgcolor": COLORS["white"],
            "font_size": 11,
            "font_color": COLORS["gray900"],
            "bordercolor": COLORS["gray200"],
        },
    }
    layout.update(overrides)
    return layout


def apply_swd_plotly_theme():
    """Register the SWD template as a named Plotly template and set as default.

    Call this once at application startup to set the default Plotly template.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    template = go.layout.Template()
    template.layout = go.Layout(**swd_plotly_layout())
    template.layout.colorway = CATEGORICAL_PALETTE

    pio.templates[SWD_PLOTLY_TEMPLATE] = template
    pio.templates.default = SWD_PLOTLY_TEMPLATE
