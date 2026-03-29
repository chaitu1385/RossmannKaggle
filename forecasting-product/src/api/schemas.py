"""
Pydantic response models for the forecasting product REST API.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' or 'degraded'")
    version: str = Field(description="API version string")
    checks: Optional[Dict[str, bool]] = Field(None, description="Dependency health checks")


class ForecastPoint(BaseModel):
    """A single (series, week, forecast) data point."""
    series_id: str
    week: date
    forecast: float
    model: Optional[str] = None
    lob: Optional[str] = None
    forecast_p10: Optional[float] = None
    forecast_p50: Optional[float] = None
    forecast_p90: Optional[float] = None


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


class AnalysisResponse(BaseModel):
    """Response from the /analyze endpoint."""
    lob_name: str
    # Schema detection
    time_column: str
    target_column: str
    id_columns: List[str]
    n_series: int
    n_rows: int
    date_range_start: str
    date_range_end: str
    frequency: str
    # Forecastability
    overall_forecastability: float
    forecastability_distribution: Dict[str, int]
    demand_classes: Dict[str, int]
    # Hierarchy
    detected_hierarchies: List[Dict[str, Any]]
    # Config recommendation
    recommended_config_yaml: str
    config_reasoning: List[str]
    hypotheses: List[str]
    # LLM insights (null if not enabled)
    llm_narrative: Optional[str] = None
    llm_risk_factors: Optional[List[str]] = None



# --------------------------------------------------------------------------- #
#  AI feature request/response models
# --------------------------------------------------------------------------- #

class NLQueryRequest(BaseModel):
    """Request body for POST /ai/explain."""
    series_id: str
    question: str
    lob: str


class NLQueryResponse(BaseModel):
    """Response from POST /ai/explain."""
    answer: str
    supporting_data: Dict[str, Any] = {}
    confidence: str = "low"       # "high" | "medium" | "low"
    sources_used: List[str] = []


class TriageRequest(BaseModel):
    """Request body for POST /ai/triage."""
    lob: str
    run_type: str = "backtest"
    severity_filter: Optional[str] = None   # "warning" | "critical" | None
    max_alerts: int = 50


class TriagedAlertItem(BaseModel):
    """A single triaged drift alert."""
    series_id: str
    metric: str
    severity: str
    business_impact_score: float = 0.0
    suggested_action: str = ""
    reasoning: str = ""
    original_message: str = ""


class TriageResponse(BaseModel):
    """Response from POST /ai/triage."""
    lob: str
    executive_summary: str = ""
    total_alerts: int = 0
    critical_count: int = 0
    warning_count: int = 0
    ranked_alerts: List[TriagedAlertItem] = []


class ConfigTuneRequest(BaseModel):
    """Request body for POST /ai/recommend-config."""
    lob: str
    run_type: str = "backtest"


class ConfigRecommendationItem(BaseModel):
    """A single configuration change recommendation."""
    field_path: str
    current_value: Any = None
    recommended_value: Any = None
    reasoning: str = ""
    expected_impact: str = ""
    risk: str = "low"             # "low" | "medium" | "high"


class ConfigTuneResponse(BaseModel):
    """Response from POST /ai/recommend-config."""
    lob: str
    recommendations: List[ConfigRecommendationItem] = []
    overall_assessment: str = ""
    risk_summary: str = ""


class CommentaryRequest(BaseModel):
    """Request body for POST /ai/commentary."""
    lob: str
    run_type: str = "backtest"
    period_start: Optional[date] = None
    period_end: Optional[date] = None


class KeyMetricItem(BaseModel):
    """A key metric with trend direction."""
    name: str
    value: float
    unit: str = ""
    trend: str = "stable"         # "improving" | "stable" | "degrading"


class CommentaryResponse(BaseModel):
    """Response from POST /ai/commentary."""
    lob: str
    executive_summary: str = ""
    key_metrics: List[KeyMetricItem] = []
    exceptions: List[str] = []
    action_items: List[str] = []
