"""Planner override CRUD endpoints."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..deps import get_app_state
from ...auth.models import Permission, User
from ...auth.rbac import require_permission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/overrides", tags=["overrides"])


class CreateOverrideRequest(BaseModel):
    old_sku: str
    new_sku: str
    proportion: float = Field(ge=0.0, le=1.0)
    scenario: str = "manual"
    ramp_shape: str = "linear"
    effective_date: Optional[str] = None
    notes: Optional[str] = None


class UpdateOverrideRequest(BaseModel):
    proportion: Optional[float] = Field(None, ge=0.0, le=1.0)
    scenario: Optional[str] = None
    ramp_shape: Optional[str] = None
    effective_date: Optional[str] = None
    notes: Optional[str] = None


@router.get("")
def list_overrides(
    old_sku: Optional[str] = Query(None),
    new_sku: Optional[str] = Query(None),
    app_state=Depends(get_app_state),
    user: User = Depends(require_permission(Permission.VIEW_FORECASTS)),
):
    """List all planner overrides, optionally filtered by SKU."""
    from ...overrides.store import get_override_store

    store_path = str(app_state.data_dir / "overrides")
    store = get_override_store(path=store_path)

    try:
        if old_sku or new_sku:
            df = store.get_overrides(old_sku=old_sku, new_sku=new_sku)
        else:
            df = store.get_all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read overrides: {exc}")
    finally:
        store.close()

    return {
        "count": df.height,
        "overrides": df.to_dicts(),
    }


@router.post("")
def create_override(
    request: CreateOverrideRequest,
    app_state=Depends(get_app_state),
    user: User = Depends(require_permission(Permission.CREATE_OVERRIDE)),
):
    """Create a new planner override."""
    from ...overrides.store import get_override_store

    store_path = str(app_state.data_dir / "overrides")
    store = get_override_store(path=store_path)

    try:
        override_id = store.add_override(
            old_sku=request.old_sku,
            new_sku=request.new_sku,
            proportion=request.proportion,
            scenario=request.scenario,
            ramp_shape=request.ramp_shape,
            effective_date=request.effective_date,
            created_by=user.user_id if hasattr(user, "user_id") else "api",
            notes=request.notes,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create override: {exc}")
    finally:
        store.close()

    return {"override_id": override_id, "status": "created"}


@router.put("/{override_id}")
def update_override(
    override_id: str,
    request: UpdateOverrideRequest,
    app_state=Depends(get_app_state),
    user: User = Depends(require_permission(Permission.CREATE_OVERRIDE)),
):
    """Update an existing planner override."""
    from ...overrides.store import get_override_store

    store_path = str(app_state.data_dir / "overrides")
    store = get_override_store(path=store_path)

    try:
        # Read existing, delete, re-create with updated fields
        existing = store.get_overrides()
        match = existing.filter(
            existing.columns[0] == override_id
        ) if not existing.is_empty() else existing

        if match.is_empty():
            raise HTTPException(status_code=404, detail=f"Override '{override_id}' not found.")

        row = match.row(0, named=True)
        store.delete_override(override_id)

        new_id = store.add_override(
            old_sku=row.get("old_sku", ""),
            new_sku=row.get("new_sku", ""),
            proportion=request.proportion if request.proportion is not None else row.get("proportion", 1.0),
            scenario=request.scenario or row.get("scenario", "manual"),
            ramp_shape=request.ramp_shape or row.get("ramp_shape", "linear"),
            effective_date=request.effective_date or row.get("effective_date"),
            created_by=user.user_id if hasattr(user, "user_id") else "api",
            notes=request.notes if request.notes is not None else row.get("notes"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to update override: {exc}")
    finally:
        store.close()

    return {"override_id": new_id, "status": "updated"}


@router.delete("/{override_id}")
def delete_override(
    override_id: str,
    app_state=Depends(get_app_state),
    user: User = Depends(require_permission(Permission.DELETE_OVERRIDE)),
):
    """Delete a planner override."""
    from ...overrides.store import get_override_store

    store_path = str(app_state.data_dir / "overrides")
    store = get_override_store(path=store_path)

    try:
        deleted = store.delete_override(override_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete override: {exc}")
    finally:
        store.close()

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Override '{override_id}' not found.")

    return {"override_id": override_id, "status": "deleted"}
