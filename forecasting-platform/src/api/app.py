"""
Forecasting Platform REST API
(Design document §13 / Phase 2)

Exposes forecast results, model leaderboards, and drift alerts over HTTP.
Built with FastAPI — auto-generates OpenAPI (Swagger) docs at /docs.

Endpoints
---------
GET  /health                         Liveness probe.
GET  /forecast/{lob}                 Latest forecasts for a LOB.
GET  /forecast/{lob}/{series_id}     Latest forecast for a specific series.
GET  /metrics/leaderboard/{lob}      Model leaderboard from metric store.
GET  /metrics/drift/{lob}            Drift alerts for a LOB.

Usage
-----
>>> from src.api.app import create_app
>>> app = create_app(data_dir="data/", metrics_dir="data/metrics/")

Or via scripts/serve.py:
    python scripts/serve.py --port 8000 --data-dir data/

Environment variables
---------------------
API_DATA_DIR      Path to local data directory (default: data/).
API_METRICS_DIR   Path to metrics store (default: data/metrics/).
API_VERSION       Version string embedded in /health (default: "1.0.0").
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query

from .schemas import (
    DriftAlertItem,
    DriftResponse,
    ForecastPoint,
    ForecastResponse,
    HealthResponse,
    LeaderboardEntry,
    LeaderboardResponse,
)

logger = logging.getLogger(__name__)

_API_VERSION = os.environ.get("API_VERSION", "1.0.0")


def create_app(
    data_dir: str = "data/",
    metrics_dir: str = "data/metrics/",
    title: str = "Forecasting Platform API",
    auth_enabled: bool = False,
    jwt_secret: str = "",
    audit_log_path: str = "data/audit_log/",
) -> FastAPI:
    """
    Factory function — returns a configured FastAPI application.

    Parameters
    ----------
    data_dir:
        Root directory for forecast Parquet files.
        Structure: ``{data_dir}/forecasts/{lob}/forecast_*.parquet``.
    metrics_dir:
        Root directory for the MetricStore Parquet partitions.
    title:
        Application title (appears in /docs).
    """
    app = FastAPI(
        title=title,
        description=(
            "REST API for the forecasting platform. "
            "Serves forecast results, model leaderboards, and drift alerts."
        ),
        version=_API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Shared state (attached to app for testability) ─────────────────────
    app.state.data_dir    = Path(data_dir)
    app.state.metrics_dir = Path(metrics_dir)
    app.state.auth_enabled = auth_enabled
    app.state.jwt_secret = jwt_secret

    from ..audit.logger import AuditLogger
    app.state.audit_logger = AuditLogger(audit_log_path)

    from ..auth.models import Permission, User
    from ..auth.rbac import get_current_user, require_permission

    # ── Routes ─────────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health():
        """Liveness probe — returns 200 OK when the service is running."""
        return HealthResponse(status="ok", version=_API_VERSION)

    @app.post("/auth/token", tags=["auth"])
    def create_auth_token(
        username: str = Query(..., description="Username"),
        role: str = Query("viewer", description="Role"),
    ):
        """Issue a JWT token (development endpoint)."""
        if not auth_enabled:
            return {"detail": "Auth is disabled. All endpoints are open."}
        from ..auth.token import create_token
        token = create_token(
            user_id=username,
            email=f"{username}@example.com",
            role=role,
            secret_key=jwt_secret,
        )
        return {"access_token": token, "token_type": "bearer"}

    @app.get("/audit", tags=["audit"])
    def get_audit_log(
        user: User = Depends(require_permission(Permission.VIEW_AUDIT_LOG)),
        action: Optional[str] = Query(None),
        resource_type: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
    ):
        """Query the audit log. Requires VIEW_AUDIT_LOG permission."""
        results = app.state.audit_logger.query(
            action=action,
            resource_type=resource_type,
            limit=limit,
        )
        return {
            "count": len(results),
            "events": results.to_dicts(),
        }

    @app.get("/forecast/{lob}", response_model=ForecastResponse, tags=["forecasts"])
    def get_forecast(
        lob: str,
        series_id: Optional[str] = Query(None, description="Filter to a single series"),
        horizon: Optional[int] = Query(None, description="Limit output to first N weeks"),
        user: User = Depends(get_current_user),
    ):
        """
        Return the latest forecast Parquet file for a LOB.

        Reads the most recently written ``forecast_*.parquet`` under
        ``{data_dir}/forecasts/{lob}/``.
        """
        forecast_dir = app.state.data_dir / "forecasts" / lob
        if not forecast_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No forecast data found for LOB '{lob}'. "
                       f"Expected directory: {forecast_dir}",
            )

        parquet_files = sorted(forecast_dir.glob("forecast_*.parquet"))
        if not parquet_files:
            raise HTTPException(
                status_code=404,
                detail=f"No forecast Parquet files in {forecast_dir}.",
            )

        import polars as pl
        df = pl.read_parquet(parquet_files[-1])   # most recent

        if series_id:
            df = df.filter(pl.col("series_id") == series_id)
            if df.is_empty():
                raise HTTPException(
                    status_code=404,
                    detail=f"series_id '{series_id}' not found in LOB '{lob}'.",
                )

        if horizon:
            df = df.sort("week").head(horizon * df["series_id"].n_unique())

        points = [
            ForecastPoint(
                series_id=row["series_id"],
                week=row["week"],
                forecast=float(row["forecast"]),
                model=row.get("model"),
                lob=lob,
            )
            for row in df.iter_rows(named=True)
        ]

        return ForecastResponse(
            lob=lob,
            series_count=df["series_id"].n_unique(),
            forecast_origin=parquet_files[-1].stem.replace(f"forecast_{lob}_", ""),
            points=points,
        )

    @app.get(
        "/forecast/{lob}/{series_id}",
        response_model=ForecastResponse,
        tags=["forecasts"],
    )
    def get_forecast_series(lob: str, series_id: str, user: User = Depends(get_current_user)):
        """Return the latest forecast for a single series within a LOB."""
        return get_forecast(lob=lob, series_id=series_id, horizon=None, user=user)

    @app.get(
        "/metrics/leaderboard/{lob}",
        response_model=LeaderboardResponse,
        tags=["metrics"],
    )
    def get_leaderboard(
        lob: str,
        run_type: str = Query("backtest", description="'backtest' or 'live'"),
        user: User = Depends(get_current_user),
    ):
        """
        Return the model leaderboard for a LOB from the metric store.

        Models are ranked by WMAPE (ascending).
        """
        from ..metrics.store import MetricStore

        store = MetricStore(str(app.state.metrics_dir))
        try:
            df = store.leaderboard(lob=lob, run_type=run_type)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read leaderboard: {exc}",
            )

        if df.is_empty():
            raise HTTPException(
                status_code=404,
                detail=f"No leaderboard data found for LOB '{lob}' (run_type={run_type}).",
            )

        entries = [
            LeaderboardEntry(
                model=row["model_id"],
                wmape=float(row["wmape"]),
                normalized_bias=float(row.get("normalized_bias", 0.0) or 0.0),
                rank=int(row.get("rank", i + 1)),
                n_series=int(row.get("n_series", 0) or 0),
            )
            for i, row in enumerate(df.iter_rows(named=True))
        ]

        return LeaderboardResponse(lob=lob, run_type=run_type, entries=entries)

    @app.get(
        "/metrics/drift/{lob}",
        response_model=DriftResponse,
        tags=["metrics"],
    )
    def get_drift(
        lob: str,
        run_type: str = Query("backtest", description="'backtest' or 'live'"),
        baseline_weeks: int = Query(26, ge=4),
        recent_weeks:   int = Query(8,  ge=2),
        user: User = Depends(get_current_user),
    ):
        """
        Return drift alerts for all series in a LOB.

        Reads the metric store, runs ``ForecastDriftDetector``, and returns
        any accuracy / bias / volume alerts.
        """
        from ..metrics.drift import DriftConfig, ForecastDriftDetector
        from ..metrics.store import MetricStore

        store = MetricStore(str(app.state.metrics_dir))
        try:
            df = store.read(lob=lob, run_type=run_type)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read metric store: {exc}",
            )

        if df is None or df.is_empty():
            raise HTTPException(
                status_code=404,
                detail=f"No metric data found for LOB '{lob}' (run_type={run_type}).",
            )

        cfg = DriftConfig(baseline_weeks=baseline_weeks, recent_weeks=recent_weeks)
        detector = ForecastDriftDetector(cfg)
        alerts = detector.detect(df)

        alert_items = [
            DriftAlertItem(
                series_id=a.series_id,
                metric=a.metric,
                severity=a.severity.value,
                current_value=a.current_value,
                baseline_value=a.baseline_value,
                message=a.message,
            )
            for a in alerts
        ]

        return DriftResponse(
            lob=lob,
            n_critical=sum(1 for a in alerts if a.severity.value == "critical"),
            n_warning=sum(1 for a in alerts if a.severity.value == "warning"),
            alerts=alert_items,
        )

    return app
