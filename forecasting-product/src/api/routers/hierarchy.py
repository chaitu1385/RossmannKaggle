"""Hierarchy management endpoints — build tree, aggregate, reconcile."""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional

import polars as pl
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from ...auth.models import Permission, User
from ...auth.rbac import get_current_user, require_permission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hierarchy", tags=["hierarchy"])


@router.post("/build")
async def build_hierarchy(
    file: UploadFile = File(...),
    levels: str = Query(..., description="Comma-separated hierarchy levels, root to leaf"),
    id_column: str = Query("series_id", description="Leaf-level ID column"),
    name: str = Query("product", description="Hierarchy name"),
    user: User = Depends(get_current_user),
):
    """Build a hierarchy tree and return structure stats."""
    from ...config.schema import HierarchyConfig
    from ...hierarchy.tree import HierarchyTree

    content = await file.read()
    filename = file.filename or ""
    try:
        if filename.endswith(".parquet"):
            df = pl.read_parquet(io.BytesIO(content))
        else:
            df = pl.read_csv(io.BytesIO(content), try_parse_dates=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    level_list = [l.strip() for l in levels.split(",")]
    missing = [l for l in level_list if l not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Hierarchy levels not found in data: {missing}. Available columns: {df.columns}",
        )

    config = HierarchyConfig(name=name, levels=level_list, id_column=id_column)
    tree = HierarchyTree(config, df)

    # Build stats per level
    level_stats = []
    for level in level_list:
        nodes = tree.get_nodes(level)
        level_stats.append({
            "level": level,
            "node_count": len(nodes),
        })

    leaves = tree.get_leaves()
    s_matrix = tree.summing_matrix()
    s_sample = s_matrix.head(20).to_dicts() if not s_matrix.is_empty() else []

    # Build flat tree_nodes list for sunburst visualization
    tree_nodes = []
    for level in level_list:
        for node in tree.get_nodes(level):
            tree_nodes.append({
                "key": node.key,
                "level": level,
                "parent": node.parent.key if node.parent else "",
                "is_leaf": node.is_leaf,
            })

    return {
        "name": name,
        "levels": level_list,
        "level_stats": level_stats,
        "total_nodes": sum(s["node_count"] for s in level_stats),
        "leaf_count": len(leaves),
        "s_matrix_shape": [s_matrix.height, s_matrix.width],
        "s_matrix_sample": s_sample,
        "tree_nodes": tree_nodes,
    }


@router.post("/aggregate")
async def aggregate_hierarchy(
    file: UploadFile = File(...),
    levels: str = Query(..., description="Comma-separated hierarchy levels"),
    id_column: str = Query("series_id"),
    target_level: str = Query(..., description="Level to aggregate to"),
    value_columns: str = Query("quantity", description="Comma-separated columns to aggregate"),
    time_column: str = Query("week"),
    agg: str = Query("sum", description="Aggregation function: 'sum', 'mean'"),
    top_n: int = Query(10, description="Return top N nodes by total value"),
    user: User = Depends(get_current_user),
):
    """Aggregate leaf-level data to a target hierarchy level."""
    from ...config.schema import HierarchyConfig
    from ...hierarchy.aggregator import HierarchyAggregator
    from ...hierarchy.tree import HierarchyTree

    content = await file.read()
    filename = file.filename or ""
    try:
        if filename.endswith(".parquet"):
            df = pl.read_parquet(io.BytesIO(content))
        else:
            df = pl.read_csv(io.BytesIO(content), try_parse_dates=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    level_list = [l.strip() for l in levels.split(",")]
    val_cols = [c.strip() for c in value_columns.split(",")]

    config = HierarchyConfig(name="hierarchy", levels=level_list, id_column=id_column)
    tree = HierarchyTree(config, df)
    aggregator = HierarchyAggregator(tree)

    aggregated = aggregator.aggregate_to(
        df, target_level=target_level,
        value_columns=val_cols, time_column=time_column, agg=agg,
    )

    # Top N nodes by total value
    if val_cols and val_cols[0] in aggregated.columns:
        top_nodes = (
            aggregated
            .group_by(target_level)
            .agg(pl.col(val_cols[0]).sum().alias("total"))
            .sort("total", descending=True)
            .head(top_n)
        )
        top_node_ids = top_nodes[target_level].to_list()
        top_data = aggregated.filter(pl.col(target_level).is_in(top_node_ids))
    else:
        top_data = aggregated.head(top_n * 52)

    return {
        "target_level": target_level,
        "total_rows": aggregated.height,
        "unique_nodes": aggregated[target_level].n_unique() if target_level in aggregated.columns else 0,
        "top_n_data": top_data.to_dicts(),
    }


@router.post("/reconcile")
async def reconcile_hierarchy(
    file: UploadFile = File(...),
    levels: str = Query(..., description="Comma-separated hierarchy levels"),
    id_column: str = Query("series_id"),
    method: str = Query("bottom_up", description="Reconciliation method: bottom_up, top_down, ols, wls, mint"),
    value_columns: str = Query("forecast", description="Comma-separated value columns"),
    time_column: str = Query("week"),
    user: User = Depends(require_permission(Permission.RUN_PIPELINE)),
):
    """Run hierarchical reconciliation and return before/after comparison."""
    from ...config.schema import HierarchyConfig, ReconciliationConfig
    from ...hierarchy.reconciler import Reconciler
    from ...hierarchy.tree import HierarchyTree

    content = await file.read()
    filename = file.filename or ""
    try:
        if filename.endswith(".parquet"):
            df = pl.read_parquet(io.BytesIO(content))
        else:
            df = pl.read_csv(io.BytesIO(content), try_parse_dates=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    level_list = [l.strip() for l in levels.split(",")]
    val_cols = [c.strip() for c in value_columns.split(",")]

    hier_config = HierarchyConfig(name="hierarchy", levels=level_list, id_column=id_column)
    tree = HierarchyTree(hier_config, df)

    recon_config = ReconciliationConfig(method=method)
    reconciler = Reconciler(trees={"hierarchy": tree}, config=recon_config)

    try:
        reconciled = reconciler.reconcile(
            forecasts={"hierarchy": df},
            value_columns=val_cols,
            time_column=time_column,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reconciliation failed: {exc}")

    # Compute before/after totals for comparison
    before_total = 0.0
    after_total = 0.0
    if val_cols and val_cols[0] in df.columns:
        before_total = float(df[val_cols[0]].sum())
    if val_cols and val_cols[0] in reconciled.columns:
        after_total = float(reconciled[val_cols[0]].sum())

    return {
        "method": method,
        "before_total": round(before_total, 2),
        "after_total": round(after_total, 2),
        "rows": reconciled.height,
        "reconciled_preview": reconciled.head(200).to_dicts(),
    }
