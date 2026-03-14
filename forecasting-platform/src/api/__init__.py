"""
REST API / serving layer for the forecasting platform.

Modules
-------
schemas     Pydantic response models for all API endpoints.
app         FastAPI application with routes.
"""

from .app import create_app  # noqa: F401
from .schemas import (  # noqa: F401
    DriftAlertItem,
    DriftResponse,
    ForecastPoint,
    ForecastResponse,
    HealthResponse,
    LeaderboardEntry,
    LeaderboardResponse,
)
