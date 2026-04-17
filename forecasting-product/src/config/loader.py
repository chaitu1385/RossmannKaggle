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
    AlertConfig,
    BacktestConfig,
    ExternalRegressorConfig,
    ForecastConfig,
    HierarchyConfig,
    ObservabilityConfig,
    OutputConfig,
    ParallelismConfig,
    PlatformConfig,
    PostValidationConfig,
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
        horizon_weeks=fc_raw.get("horizon_periods",
                                 fc_raw.get("horizon_weeks", 39)),
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
        val_weeks=bt_raw.get("val_periods",
                             bt_raw.get("val_weeks", 13)),
        gap_weeks=bt_raw.get("gap_periods",
                             bt_raw.get("gap_weeks", 0)),
        champion_granularity=bt_raw.get("champion_granularity", "lob"),
        primary_metric=bt_raw.get("primary_metric", "wmape"),
        secondary_metric=bt_raw.get("secondary_metric", "normalized_bias"),
    )

    tr_raw = d.get("transition", {})
    transition = TransitionConfig(
        transition_window_weeks=tr_raw.get(
            "transition_window_periods",
            tr_raw.get("transition_window_weeks", 13),
        ),
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

    # ── Parallelism config ───────────────────────────────────────────────
    par_raw = d.get("parallelism", {})
    parallelism = ParallelismConfig(
        backend=par_raw.get("backend", "local"),
        n_workers=par_raw.get("n_workers", -1),
        n_jobs_statsforecast=par_raw.get("n_jobs_statsforecast", -1),
        num_threads_mlforecast=par_raw.get("num_threads_mlforecast", -1),
        batch_size=par_raw.get("batch_size", 0),
        gpu=par_raw.get("gpu", False),
    )

    # ── Observability config ──────────────────────────────────────────────
    obs_raw = d.get("observability", {})
    alert_raw = obs_raw.get("alerts", {})
    observability = ObservabilityConfig(
        log_format=obs_raw.get("log_format", "text"),
        log_level=obs_raw.get("log_level", "INFO"),
        metrics_backend=obs_raw.get("metrics_backend", "log"),
        statsd_host=obs_raw.get("statsd_host", "localhost"),
        statsd_port=obs_raw.get("statsd_port", 8125),
        metrics_prefix=obs_raw.get("metrics_prefix", "forecast_platform"),
        cost_per_second=obs_raw.get("cost_per_second", 0.0),
        alerts=AlertConfig(
            channels=alert_raw.get("channels", ["log"]),
            webhook_url=alert_raw.get("webhook_url", ""),
            min_severity=alert_raw.get("min_severity", "warning"),
            webhook_timeout=alert_raw.get("webhook_timeout", 10),
        ),
    )

    # ── Post-validation config ─────────────────────────────────────────
    pv_raw = d.get("post_validation", {})
    post_validation = PostValidationConfig(
        enabled=pv_raw.get("enabled", True),
        structural_checks=pv_raw.get("structural_checks", True),
        logical_checks=pv_raw.get("logical_checks", True),
        business_rules_checks=pv_raw.get("business_rules_checks", True),
        simpsons_paradox_checks=pv_raw.get("simpsons_paradox_checks", True),
        max_yoy_change_pct=pv_raw.get("max_yoy_change_pct", 500.0),
        max_period_change_pct=pv_raw.get("max_period_change_pct", 500.0),
        custom_range_rules=pv_raw.get("custom_range_rules", []),
        simpsons_segment_columns=pv_raw.get("simpsons_segment_columns", []),
        halt_on_blocker=pv_raw.get("halt_on_blocker", False),
        min_grade=pv_raw.get("min_grade", "D"),
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
        parallelism=parallelism,
        observability=observability,
        post_validation=post_validation,
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
