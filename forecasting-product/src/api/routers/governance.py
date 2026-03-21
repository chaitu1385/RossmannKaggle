"""Governance endpoints — model cards, forecast lineage, BI export."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..deps import get_app_state
from ...auth.models import Permission, User
from ...auth.rbac import get_current_user, require_permission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/governance", tags=["governance"])


@router.get("/model-cards")
def list_model_cards(
    app_state=Depends(get_app_state),
    user: User = Depends(get_current_user),
):
    """List all registered model cards."""
    from ...analytics.governance import ModelCardRegistry

    base_path = str(app_state.data_dir / "model_cards")
    registry = ModelCardRegistry(base_path=base_path)

    try:
        cards_df = registry.all_cards()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read model cards: {exc}")

    if cards_df is None or cards_df.is_empty():
        return {"count": 0, "model_cards": []}

    return {
        "count": cards_df.height,
        "model_cards": cards_df.to_dicts(),
    }


@router.get("/model-cards/{model_name}")
def get_model_card(
    model_name: str,
    app_state=Depends(get_app_state),
    user: User = Depends(get_current_user),
):
    """Get a specific model card by name."""
    from ...analytics.governance import ModelCardRegistry

    base_path = str(app_state.data_dir / "model_cards")
    registry = ModelCardRegistry(base_path=base_path)

    card = registry.get(model_name)
    if card is None:
        raise HTTPException(status_code=404, detail=f"Model card '{model_name}' not found.")

    return card.to_dict()


@router.get("/lineage")
def get_lineage(
    lob: Optional[str] = Query(None),
    model_id: Optional[str] = Query(None),
    app_state=Depends(get_app_state),
    user: User = Depends(get_current_user),
):
    """Get forecast lineage history."""
    from ...analytics.governance import ForecastLineage

    base_path = str(app_state.data_dir / "lineage")
    lineage = ForecastLineage(base_path=base_path)

    try:
        history = lineage.history(lob=lob, model_id=model_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read lineage: {exc}")

    if history is None or history.is_empty():
        return {"count": 0, "lineage": []}

    return {
        "count": history.height,
        "lineage": history.to_dicts(),
    }


@router.post("/export/{report_type}")
def bi_export(
    report_type: str,
    lob: str = Query(..., description="LOB name"),
    run_type: str = Query("backtest"),
    model_id: Optional[str] = Query(None),
    app_state=Depends(get_app_state),
    user: User = Depends(require_permission(Permission.VIEW_METRICS)),
):
    """Export BI report (forecast-actual, leaderboard, or bias-report)."""
    from ...analytics.bi_export import BIExporter
    from ...metrics.store import MetricStore

    valid_types = ("forecast-actual", "leaderboard", "bias-report")
    if report_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid report type '{report_type}'. Must be one of: {valid_types}",
        )

    base_path = str(app_state.data_dir / "bi_exports")
    exporter = BIExporter(base_path=base_path)

    if report_type == "leaderboard":
        store = MetricStore(str(app_state.metrics_dir))
        try:
            path = exporter.export_leaderboard(
                metric_store=store, lob=lob, run_type=run_type,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Export failed: {exc}")
        return {"report_type": report_type, "export_path": str(path), "status": "exported"}

    elif report_type == "bias-report":
        store = MetricStore(str(app_state.metrics_dir))
        try:
            path = exporter.export_bias_report(
                metric_store=store, lob=lob, model_id=model_id, run_type=run_type,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Export failed: {exc}")
        return {"report_type": report_type, "export_path": str(path), "status": "exported"}

    elif report_type == "forecast-actual":
        import polars as pl

        # Load forecast and actuals
        forecast_dir = app_state.data_dir / "forecasts" / lob
        if not forecast_dir.exists():
            raise HTTPException(status_code=404, detail=f"No forecast data for LOB '{lob}'.")

        pfiles = sorted(forecast_dir.glob("forecast_*.parquet"))
        if not pfiles:
            raise HTTPException(status_code=404, detail="No forecast files found.")

        forecasts = pl.read_parquet(pfiles[-1])

        # Load actuals
        actuals = pl.DataFrame()
        for subdir in ("history", "actuals"):
            d = app_state.data_dir / subdir / lob
            if d.exists():
                afiles = sorted(d.glob("*.parquet"))
                if afiles:
                    actuals = pl.read_parquet(afiles[-1])
                    break

        if actuals.is_empty():
            raise HTTPException(status_code=404, detail=f"No actuals data for LOB '{lob}'.")

        try:
            path = exporter.export_forecast_vs_actual(
                forecasts=forecasts, actuals=actuals, lob=lob,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Export failed: {exc}")

        return {"report_type": report_type, "export_path": str(path), "status": "exported"}
