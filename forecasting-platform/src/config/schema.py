"""
Platform configuration schema.

All platform behaviour is driven by these dataclasses, loaded from YAML.
The schema is intentionally generic — hierarchy level names, metric choices,
model selections, and reconciliation strategies are all configuration, never
hard-coded to a specific LOB.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HierarchyLevelConfig:
    """One level in a hierarchy dimension (e.g. 'country' inside geography)."""
    name: str
    parent: Optional[str] = None


@dataclass
class ReconciliationConfig:
    """How forecasts at different hierarchy levels are made consistent."""
    method: str = "bottom_up"       # bottom_up | top_down | middle_out | mint | ols | wls
    product_level: Optional[str] = None   # middle-out home grain for product
    geography_level: Optional[str] = None  # middle-out home grain for geography


@dataclass
class HierarchyConfig:
    """Definition of a single hierarchy dimension."""
    name: str                            # "product", "geography", "channel"
    levels: List[str] = field(default_factory=list)  # ordered root → leaf
    id_column: str = ""                  # column name in data for leaf-level key
    fixed: bool = False                  # if True, pipeline runs separately per value
    reconciliation_level: Optional[str] = None  # middle-out home grain


@dataclass
class ForecastConfig:
    """Forecast horizon and model selection."""
    horizon_weeks: int = 39              # 9 months
    frequency: str = "W"                 # weekly
    target_column: str = "quantity"
    time_column: str = "week"
    series_id_column: str = "series_id"  # unique key per time series
    forecasters: List[str] = field(
        default_factory=lambda: ["naive_seasonal"]
    )
    quantiles: List[float] = field(
        default_factory=list
    )                                    # e.g. [0.1, 0.5, 0.9]; empty = point forecast only
    intermittent_forecasters: List[str] = field(
        default_factory=list
    )                                    # e.g. ["croston_sba", "tsb"]; empty = no sparse routing
    sparse_detection: bool = True        # auto-detect sparse series when intermittent_forecasters set
    sparse_adi_threshold: float = 1.32   # ADI ≥ threshold → sparse (SBC recommendation)
    sparse_cv2_threshold: float = 0.49   # CV² threshold for SBC classification


@dataclass
class BacktestConfig:
    """Walk-forward cross-validation settings."""
    n_folds: int = 3
    val_weeks: int = 13                  # each fold validates on 13 weeks
    gap_weeks: int = 0                   # gap between train end and val start
    champion_granularity: str = "lob"    # "lob" | "product_group" | "series"
    primary_metric: str = "wmape"
    secondary_metric: str = "normalized_bias"
    selection_strategy: str = "champion" # "champion" | "weighted_ensemble"


@dataclass
class TransitionConfig:
    """Product transition stitching settings."""
    transition_window_weeks: int = 13    # 3-month overlap window
    ramp_shape: str = "linear"           # "linear" | "scurve" | "step"
    enable_overrides: bool = True        # read planner overrides from store
    override_store_path: str = "data/overrides.duckdb"


@dataclass
class OutputConfig:
    """What the pipeline produces and where it writes."""
    grain: Dict[str, str] = field(default_factory=dict)
    # e.g. {"product": "product_unit", "geography": "country", "channel": "channel"}
    forecast_path: str = "data/forecasts/"
    metrics_path: str = "data/metrics/"
    bi_export_path: str = "data/bi_exports/"
    format: str = "parquet"              # "parquet" | "csv"


@dataclass
class PlatformConfig:
    """
    Top-level configuration for the forecasting platform.

    This is the single source of truth that every module reads from.
    A different YAML file → a completely different forecasting setup
    (Surface vs. Xbox vs. Walmart).
    """
    lob: str = "default"
    description: str = ""

    hierarchies: List[HierarchyConfig] = field(default_factory=list)
    reconciliation: ReconciliationConfig = field(
        default_factory=ReconciliationConfig
    )
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    metrics: List[str] = field(
        default_factory=lambda: ["wmape", "normalized_bias"]
    )

    def get_hierarchy(self, name: str) -> HierarchyConfig:
        """Look up a hierarchy dimension by name."""
        for h in self.hierarchies:
            if h.name == name:
                return h
        raise KeyError(f"Hierarchy {name!r} not found in config")

    def get_fixed_hierarchies(self) -> List[HierarchyConfig]:
        """Return hierarchies marked as fixed (pipeline runs per value)."""
        return [h for h in self.hierarchies if h.fixed]

    def get_reconcilable_hierarchies(self) -> List[HierarchyConfig]:
        """Return hierarchies eligible for reconciliation."""
        return [h for h in self.hierarchies if not h.fixed]
