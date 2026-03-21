"""
Pipeline manifest — provenance sidecar for every forecast run.

Captures the full lineage of a forecast: input data fingerprint, cleansing
actions, validation results, regressor screening, config hash, champion model,
and output metadata.  Written as a JSON file alongside each forecast Parquet.

Usage
-----
>>> manifest = build_manifest(
...     run_id="abc123",
...     config=config,
...     actuals=actuals_df,
...     series_builder=builder,
...     champion_model_id="lgbm_direct",
...     forecast=forecast_df,
...     forecast_file="forecast_retail_2024-06-01.parquet",
... )
>>> path = write_manifest(manifest, "data/forecasts/forecast_retail_2024-06-01.parquet")
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import polars as pl

from ..config.schema import PlatformConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineManifest:
    """Full provenance record for a single forecast run."""

    # Run identity
    run_id: str = ""
    timestamp: str = ""
    lob: str = ""

    # Input data fingerprint
    input_data_hash: str = ""
    input_row_count: int = 0
    input_series_count: int = 0
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None

    # Cleansing summary
    cleansing_applied: bool = False
    outliers_clipped: int = 0
    stockout_periods_imputed: int = 0
    rows_modified: int = 0

    # Validation summary
    validation_applied: bool = False
    validation_passed: bool = True
    validation_warnings: int = 0
    validation_errors: int = 0

    # Regressor screen summary
    regressor_screen_applied: bool = False
    regressors_dropped: List[str] = field(default_factory=list)
    regressor_warnings: List[str] = field(default_factory=list)

    # Config
    config_hash: str = ""

    # Model
    champion_model_id: str = ""
    backtest_wmape: Optional[float] = None

    # Output
    forecast_horizon: int = 0
    forecast_row_count: int = 0
    forecast_file: str = ""


def build_manifest(
    run_id: str,
    config: PlatformConfig,
    actuals: pl.DataFrame,
    series_builder: Any,
    champion_model_id: str,
    forecast: pl.DataFrame,
    forecast_file: str,
    backtest_wmape: Optional[float] = None,
) -> PipelineManifest:
    """
    Collect provenance from existing pipeline components into a manifest.

    Parameters
    ----------
    run_id:
        Unique identifier for this pipeline run.
    config:
        The PlatformConfig that drove the run.
    actuals:
        Raw input actuals DataFrame (before any processing).
    series_builder:
        The SeriesBuilder instance (carries validation/cleansing reports).
    champion_model_id:
        Name of the champion model used for forecasting.
    forecast:
        The output forecast DataFrame.
    forecast_file:
        Filename of the forecast Parquet file.
    backtest_wmape:
        Optional backtest WMAPE of the champion model.
    """
    manifest = PipelineManifest(
        run_id=run_id,
        timestamp=datetime.utcnow().isoformat(timespec="seconds"),
        lob=config.lob,
        champion_model_id=champion_model_id,
        backtest_wmape=backtest_wmape,
        forecast_horizon=config.forecast.horizon_weeks,
        forecast_row_count=len(forecast),
        forecast_file=forecast_file,
    )

    # Input data fingerprint
    manifest.input_row_count = len(actuals)
    time_col = config.forecast.time_column
    id_col = config.forecast.series_id_column
    if id_col in actuals.columns:
        manifest.input_series_count = actuals[id_col].n_unique()
    if time_col in actuals.columns:
        mn = actuals[time_col].min()
        mx = actuals[time_col].max()
        if mn is not None:
            manifest.date_range_start = mn.isoformat() if isinstance(mn, (date, datetime)) else str(mn)
        if mx is not None:
            manifest.date_range_end = mx.isoformat() if isinstance(mx, (date, datetime)) else str(mx)

    manifest.input_data_hash = _hash_dataframe(actuals)

    # Config hash
    manifest.config_hash = _hash_config(config)

    # Cleansing report (if available)
    cleansing = getattr(series_builder, "_last_cleansing_report", None)
    if cleansing is not None:
        manifest.cleansing_applied = True
        manifest.outliers_clipped = getattr(cleansing, "total_outliers", 0)
        manifest.stockout_periods_imputed = getattr(cleansing, "total_stockout_periods", 0)
        manifest.rows_modified = getattr(cleansing, "rows_modified", 0)

    # Validation report (if available)
    validation = getattr(series_builder, "_last_validation_report", None)
    if validation is not None:
        manifest.validation_applied = True
        manifest.validation_passed = getattr(validation, "passed", True)
        manifest.validation_warnings = len(getattr(validation, "warnings", []))
        manifest.validation_errors = len(getattr(validation, "errors", []))

    # Regressor screen report (if available)
    screen = getattr(series_builder, "_last_regressor_screen_report", None)
    if screen is not None:
        manifest.regressor_screen_applied = True
        manifest.regressors_dropped = list(getattr(screen, "dropped_columns", []))
        manifest.regressor_warnings = list(getattr(screen, "warnings", []))

    return manifest


def write_manifest(manifest: PipelineManifest, forecast_path: str) -> str:
    """
    Write a manifest as a JSON sidecar alongside the forecast file.

    Parameters
    ----------
    manifest:
        The PipelineManifest to write.
    forecast_path:
        Full path to the forecast Parquet file.

    Returns
    -------
    Path to the written manifest JSON file.
    """
    p = Path(forecast_path)
    manifest_path = p.parent / f"{p.stem}_manifest.json"

    data = asdict(manifest)
    manifest_path.write_text(json.dumps(data, indent=2, default=str))

    logger.info("Manifest written to %s", manifest_path)
    return str(manifest_path)


def read_manifest(manifest_path: str) -> PipelineManifest:
    """
    Load a PipelineManifest from a JSON file.

    Parameters
    ----------
    manifest_path:
        Path to the manifest JSON file.
    """
    data = json.loads(Path(manifest_path).read_text())
    return PipelineManifest(**{
        k: v for k, v in data.items()
        if k in PipelineManifest.__dataclass_fields__
    })


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hash_dataframe(df: pl.DataFrame) -> str:
    """Compute a stable hash of a Polars DataFrame."""
    try:
        sorted_df = df.sort(df.columns)
        raw = sorted_df.write_ipc(None).getvalue()
        return hashlib.md5(raw).hexdigest()[:12]
    except Exception:
        return hashlib.md5(str(df.shape).encode()).hexdigest()[:12]


def _hash_config(config: PlatformConfig) -> str:
    """Compute an MD5 hash of the config (matching ModelCard pattern)."""
    try:
        payload = json.dumps(
            asdict(config), sort_keys=True, default=str
        ).encode()
        return hashlib.md5(payload).hexdigest()[:8]
    except Exception:
        return ""
