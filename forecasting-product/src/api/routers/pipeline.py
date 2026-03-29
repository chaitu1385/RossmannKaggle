"""Pipeline operation endpoints — run backtest/forecast, manifests, costs."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Optional

import polars as pl
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile

from ...auth.models import Permission, User
from ...auth.rbac import require_permission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post("/backtest")
async def run_backtest(
    request: Request,
    file: UploadFile = File(..., description="Actuals CSV/Parquet"),
    config_file: Optional[UploadFile] = File(None, description="YAML config file"),
    lob: str = Query("default", description="LOB name"),
    user: User = Depends(require_permission(Permission.RUN_BACKTEST)),
):
    """Run the backtest pipeline on uploaded data."""
    from ...pipeline.backtest import BacktestPipeline

    content = await file.read()
    filename = file.filename or ""
    try:
        if filename.endswith(".parquet"):
            df = pl.read_parquet(io.BytesIO(content))
        else:
            df = pl.read_csv(io.BytesIO(content), try_parse_dates=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    # Load config
    config = None
    if config_file:
        import yaml
        from ...config.schema import PlatformConfig
        config_content = await config_file.read()
        try:
            config_dict = yaml.safe_load(config_content.decode("utf-8"))
            config = PlatformConfig(**config_dict) if config_dict else PlatformConfig(lob=lob)
        except Exception:
            logging.getLogger(__name__).warning("Invalid config file, using defaults for lob=%s", lob, exc_info=True)
            config = PlatformConfig(lob=lob)
    else:
        from ...config.schema import PlatformConfig
        config = PlatformConfig(lob=lob)

    try:
        pipeline = BacktestPipeline(config=config, data_dir=str(request.app.state.data_dir))
        results = pipeline.run(actuals=df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}")

    # Summarize results
    summary = {
        "lob": lob,
        "status": "completed",
    }
    if hasattr(results, "leaderboard") and results.leaderboard is not None:
        summary["leaderboard"] = results.leaderboard.to_dicts()
    if hasattr(results, "champion_model_id"):
        summary["champion_model"] = results.champion_model_id
    if hasattr(results, "wmape"):
        summary["best_wmape"] = float(results.wmape) if results.wmape else None

    return summary


@router.post("/forecast")
async def run_forecast(
    request: Request,
    file: UploadFile = File(..., description="Actuals CSV/Parquet"),
    config_file: Optional[UploadFile] = File(None, description="YAML config file"),
    lob: str = Query("default", description="LOB name"),
    model_id: Optional[str] = Query(None, description="Specific model to use"),
    horizon: int = Query(12, description="Forecast horizon in periods"),
    user: User = Depends(require_permission(Permission.RUN_PIPELINE)),
):
    """Run the forecast pipeline on uploaded data."""
    from ...pipeline.forecast import ForecastPipeline

    content = await file.read()
    filename = file.filename or ""
    try:
        if filename.endswith(".parquet"):
            df = pl.read_parquet(io.BytesIO(content))
        else:
            df = pl.read_csv(io.BytesIO(content), try_parse_dates=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    config = None
    if config_file:
        import yaml
        from ...config.schema import PlatformConfig
        config_content = await config_file.read()
        try:
            config_dict = yaml.safe_load(config_content.decode("utf-8"))
            config = PlatformConfig(**config_dict) if config_dict else PlatformConfig(lob=lob)
        except Exception:
            logging.getLogger(__name__).warning("Invalid config file, using defaults for lob=%s", lob, exc_info=True)
            config = PlatformConfig(lob=lob)
    else:
        from ...config.schema import PlatformConfig
        config = PlatformConfig(lob=lob)

    try:
        pipeline = ForecastPipeline(config=config, data_dir=str(request.app.state.data_dir))
        result = pipeline.run(
            actuals=df,
            model_id=model_id,
            horizon=horizon,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast pipeline failed: {exc}")

    summary = {
        "lob": lob,
        "status": "completed",
        "model_id": model_id,
        "horizon": horizon,
    }
    if hasattr(result, "forecast") and result.forecast is not None:
        summary["forecast_rows"] = result.forecast.height
        summary["series_count"] = result.forecast["series_id"].n_unique() if "series_id" in result.forecast.columns else 0
        summary["forecast_preview"] = result.forecast.head(100).to_dicts()

    return summary


@router.get("/manifests")
def list_manifests(
    request: Request,
    lob: Optional[str] = Query(None, description="Filter by LOB"),
    limit: int = Query(20, ge=1, le=100),
    user: User = Depends(require_permission(Permission.VIEW_METRICS)),
):
    """List recent pipeline run manifests."""
    from ...pipeline.manifest import PipelineManifest, read_manifest

    manifests_dir = request.app.state.data_dir / "forecasts"
    if not manifests_dir.exists():
        return {"count": 0, "manifests": []}

    manifest_files = []
    search_dirs = [manifests_dir / lob] if lob else list(manifests_dir.iterdir())

    for d in search_dirs:
        if d.is_dir():
            manifest_files.extend(sorted(d.glob("manifest_*.json"), reverse=True))

    manifest_files = sorted(manifest_files, key=lambda f: f.stat().st_mtime, reverse=True)[:limit]

    results = []
    for mf in manifest_files:
        try:
            manifest = read_manifest(str(mf))
            results.append({
                "run_id": manifest.run_id,
                "timestamp": manifest.timestamp,
                "lob": manifest.lob,
                "series_count": manifest.input_series_count,
                "champion_model": manifest.champion_model_id,
                "backtest_wmape": manifest.backtest_wmape,
                "forecast_horizon": manifest.forecast_horizon,
                "forecast_rows": manifest.forecast_row_count,
                "validation_passed": manifest.validation_passed,
                "validation_warnings": manifest.validation_warnings,
                "cleansing_applied": manifest.cleansing_applied,
                "outliers_clipped": manifest.outliers_clipped,
            })
        except Exception as exc:
            logger.warning(f"Failed to read manifest {mf}: {exc}")

    return {"count": len(results), "manifests": results}


@router.get("/costs")
def get_costs(
    request: Request,
    lob: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: User = Depends(require_permission(Permission.VIEW_METRICS)),
):
    """Get cost tracking data from pipeline manifests."""
    from ...pipeline.manifest import read_manifest

    manifests_dir = request.app.state.data_dir / "forecasts"
    if not manifests_dir.exists():
        return {"count": 0, "costs": []}

    manifest_files = []
    search_dirs = [manifests_dir / lob] if lob else list(manifests_dir.iterdir())

    for d in search_dirs:
        if d.is_dir():
            manifest_files.extend(sorted(d.glob("manifest_*.json"), reverse=True))

    manifest_files = sorted(manifest_files, key=lambda f: f.stat().st_mtime, reverse=True)[:limit]

    costs = []
    for mf in manifest_files:
        try:
            manifest = read_manifest(str(mf))
            series_count = manifest.input_series_count or 1
            # Try to read timing from manifest (may not always be present)
            total_seconds = getattr(manifest, "total_seconds", 0.0)
            costs.append({
                "run_id": manifest.run_id,
                "timestamp": manifest.timestamp,
                "lob": manifest.lob,
                "series_count": series_count,
                "champion_model": manifest.champion_model_id,
                "total_seconds": total_seconds,
                "seconds_per_series": round(total_seconds / series_count, 3) if total_seconds else None,
            })
        except Exception as exc:
            logger.warning(f"Failed to read manifest {mf}: {exc}")

    return {"count": len(costs), "costs": costs}


@router.post("/analyze-multi-file")
async def analyze_multi_file(
    files: list[UploadFile] = File(..., description="Multiple CSV/Parquet files"),
    lob_name: str = Query("analyzed"),
    user: User = Depends(require_permission(Permission.RUN_PIPELINE)),
):
    """Upload multiple files, auto-classify roles, and merge."""
    from ...data.file_classifier import FileClassifier
    from ...data.file_merger import MultiFileMerger

    file_dfs = {}
    for f in files:
        content = await f.read()
        filename = f.filename or f"file_{len(file_dfs)}"
        try:
            if filename.endswith(".parquet"):
                file_dfs[filename] = pl.read_parquet(io.BytesIO(content))
            else:
                file_dfs[filename] = pl.read_csv(io.BytesIO(content), try_parse_dates=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read '{filename}': {exc}")

    classifier = FileClassifier()
    classification = classifier.classify_files(file_dfs)

    profiles = []
    for p in classification.profiles:
        profiles.append({
            "filename": p.filename,
            "role": p.role,
            "confidence": round(p.confidence, 3),
            "time_column": p.time_column,
            "id_columns": p.id_columns,
            "n_rows": p.n_rows,
            "n_columns": p.n_columns,
            "reasoning": p.reasoning,
        })

    result = {
        "profiles": profiles,
        "primary_file": classification.primary_file.filename if classification.primary_file else None,
        "dimension_files": [f.filename for f in classification.dimension_files],
        "regressor_files": [f.filename for f in classification.regressor_files],
        "warnings": classification.warnings,
    }

    # Attempt merge if there's a primary file and secondary files
    if classification.primary_file and (classification.dimension_files or classification.regressor_files):
        try:
            merger = MultiFileMerger()
            preview = merger.preview_merge(classification)
            result["merge_preview"] = {
                "total_rows": preview.total_rows,
                "total_columns": preview.total_columns,
                "matched_rows": preview.matched_rows,
                "unmatched_primary_keys": preview.unmatched_primary_keys,
                "null_fill_columns": preview.null_fill_columns,
                "warnings": preview.warnings,
                "sample_rows": preview.sample_rows.head(20).to_dicts() if preview.sample_rows is not None else [],
            }
        except Exception as exc:
            result["merge_error"] = str(exc)

    return result
