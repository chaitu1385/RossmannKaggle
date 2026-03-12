"""
Pydantic response models for the forecasting platform REST API.
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when the service is healthy")
    version: str = Field(description="API version string")


class ForecastPoint(BaseModel):
    """A single (series, week, forecast) data point."""
    series_id: str
    week: date
    forecast: float
    model: Optional[str] = None
    lob: Optional[str] = None


class ForecastResponse(BaseModel):
    lob: str
    series_count: int
    forecast_origin: Optional[str] = None
    points: List[ForecastPoint]


class LeaderboardEntry(BaseModel):
    """One model's aggregated performance metrics."""
    model: str
    wmape: float
    normalized_bias: float
    rank: int
    n_series: int


class LeaderboardResponse(BaseModel):
    lob: str
    run_type: str
    entries: List[LeaderboardEntry]


class DriftAlertItem(BaseModel):
    """One drift alert for a series/metric pair."""
    series_id: str
    metric: str
    severity: str     # "warning" | "critical"
    current_value: float
    baseline_value: float
    message: str


class DriftResponse(BaseModel):
    lob: str
    n_critical: int
    n_warning: int
    alerts: List[DriftAlertItem]
