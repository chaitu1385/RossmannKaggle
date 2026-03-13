"""
YAML configuration loader with LOB inheritance.

Supports a base config that LOB-specific configs can selectively override.
Nested dicts are deep-merged; lists are replaced entirely.
"""

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .schema import (
    BacktestConfig,
    ExternalRegressorConfig,
    ForecastConfig,
    HierarchyConfig,
    OutputConfig,
    PlatformConfig,
    ReconciliationConfig,
    TransitionConfig,
)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into *base*. Lists are replaced, not appended."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_hierarchy(raw: Dict[str, Any]) -> HierarchyConfig:
    return HierarchyConfig(
        name=raw["name"],
        levels=raw.get("levels", []),
        id_column=raw.get("id_column", ""),
        fixed=raw.get("fixed", False),
        reconciliation_level=raw.get("reconciliation_level"),
    )


def _dict_to_config(d: Dict[str, Any]) -> PlatformConfig:
    """Convert a raw dict (from YAML) to a typed PlatformConfig."""
    hierarchies = [
        _parse_hierarchy(h) for h in d.get("hierarchies", [])
    ]

    recon_raw = d.get("reconciliation", {})
    reconciliation = ReconciliationConfig(
        method=recon_raw.get("method", "bottom_up"),
        product_level=recon_raw.get("product_level"),
        geography_level=recon_raw.get("geography_level"),
    )

    fc_raw = d.get("forecast", {})
    forecast = ForecastConfig(
        horizon_weeks=fc_raw.get("horizon_weeks", 39),
        frequency=fc_raw.get("frequency", "W"),
        target_column=fc_raw.get("target_column", "quantity"),
        time_column=fc_raw.get("time_column", "week"),
        series_id_column=fc_raw.get("series_id_column", "series_id"),
        forecasters=fc_raw.get("forecasters", ["naive_seasonal"]),
        external_regressors=ExternalRegressorConfig(
            enabled=fc_raw.get("external_regressors", {}).get("enabled", False),
            feature_columns=fc_raw.get("external_regressors", {}).get("feature_columns", []),
            future_features_path=fc_raw.get("external_regressors", {}).get("future_features_path"),
        ),
    )

    bt_raw = d.get("backtest", {})
    backtest = BacktestConfig(
        n_folds=bt_raw.get("n_folds", 3),
        val_weeks=bt_raw.get("val_weeks", 13),
        gap_weeks=bt_raw.get("gap_weeks", 0),
        champion_granularity=bt_raw.get("champion_granularity", "lob"),
        primary_metric=bt_raw.get("primary_metric", "wmape"),
        secondary_metric=bt_raw.get("secondary_metric", "normalized_bias"),
    )

    tr_raw = d.get("transition", {})
    transition = TransitionConfig(
        transition_window_weeks=tr_raw.get("transition_window_weeks", 13),
        ramp_shape=tr_raw.get("ramp_shape", "linear"),
        enable_overrides=tr_raw.get("enable_overrides", True),
        override_store_path=tr_raw.get(
            "override_store_path", "data/overrides.duckdb"
        ),
    )

    out_raw = d.get("output", {})
    output = OutputConfig(
        grain=out_raw.get("grain", {}),
        forecast_path=out_raw.get("forecast_path", "data/forecasts/"),
        metrics_path=out_raw.get("metrics_path", "data/metrics/"),
        bi_export_path=out_raw.get("bi_export_path", "data/bi_exports/"),
        format=out_raw.get("format", "parquet"),
    )

    return PlatformConfig(
        lob=d.get("lob", "default"),
        description=d.get("description", ""),
        hierarchies=hierarchies,
        reconciliation=reconciliation,
        forecast=forecast,
        backtest=backtest,
        transition=transition,
        output=output,
        metrics=d.get("metrics", ["wmape", "normalized_bias"]),
    )


def load_config(path: str) -> PlatformConfig:
    """Load a single YAML config file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _dict_to_config(raw)


def load_config_with_overrides(
    base_path: str,
    override_path: Optional[str] = None,
) -> PlatformConfig:
    """
    Load a base config and optionally deep-merge an LOB override on top.

    Usage::

        # Base only
        cfg = load_config_with_overrides("configs/platform_config.yaml")

        # Base + Surface overrides
        cfg = load_config_with_overrides(
            "configs/platform_config.yaml",
            "configs/lob/surface.yaml",
        )
    """
    with open(base_path) as f:
        base = yaml.safe_load(f) or {}

    if override_path and Path(override_path).exists():
        with open(override_path) as f:
            override = yaml.safe_load(f) or {}
        merged = _deep_merge(base, override)
    else:
        merged = base

    return _dict_to_config(merged)
