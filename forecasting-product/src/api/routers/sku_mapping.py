"""SKU mapping endpoints — Phase 1 and Phase 2 mapping pipelines."""

from __future__ import annotations

import io
import logging
from typing import Optional

import polars as pl
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from ..deps import get_app_state
from ...auth.models import Permission, User
from ...auth.rbac import require_permission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sku-mapping", tags=["sku-mapping"])


@router.post("/phase1")
async def run_phase1(
    product_master: UploadFile = File(..., description="Product master CSV"),
    launch_window_days: int = Query(180),
    min_base_similarity: float = Query(0.70),
    min_confidence: str = Query("Low", description="'Low', 'Medium', or 'High'"),
    user: User = Depends(require_permission(Permission.RUN_PIPELINE)),
):
    """Run Phase 1 SKU mapping (attribute + naming matching)."""
    from ...sku_mapping.pipeline import build_phase1_pipeline

    content = await product_master.read()
    try:
        df = pl.read_csv(io.BytesIO(content), try_parse_dates=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read product master: {exc}")

    pipeline = build_phase1_pipeline(
        launch_window_days=launch_window_days,
        min_base_similarity=min_base_similarity,
        min_confidence=min_confidence,
    )

    try:
        mappings = pipeline.run(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Phase 1 mapping failed: {exc}")

    return {
        "phase": 1,
        "total_mappings": mappings.height,
        "mappings": mappings.to_dicts(),
    }


@router.post("/phase2")
async def run_phase2(
    product_master: UploadFile = File(..., description="Product master CSV"),
    sales_history: Optional[UploadFile] = File(None, description="Sales history CSV for curve fitting"),
    launch_window_days: int = Query(180),
    min_base_similarity: float = Query(0.70),
    min_confidence: str = Query("Low"),
    window_weeks: int = Query(13),
    user: User = Depends(require_permission(Permission.RUN_PIPELINE)),
):
    """Run Phase 2 SKU mapping (attribute + naming + curve fitting)."""
    from ...sku_mapping.pipeline import build_phase2_pipeline

    content = await product_master.read()
    try:
        pm_df = pl.read_csv(io.BytesIO(content), try_parse_dates=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read product master: {exc}")

    sales_df = None
    if sales_history:
        sales_content = await sales_history.read()
        try:
            sales_df = pl.read_csv(io.BytesIO(sales_content), try_parse_dates=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read sales history: {exc}")

    pipeline = build_phase2_pipeline(
        sales_df=sales_df,
        launch_window_days=launch_window_days,
        min_base_similarity=min_base_similarity,
        min_confidence=min_confidence,
        window_weeks=window_weeks,
    )

    try:
        mappings = pipeline.run(pm_df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Phase 2 mapping failed: {exc}")

    return {
        "phase": 2,
        "total_mappings": mappings.height,
        "mappings": mappings.to_dicts(),
    }
