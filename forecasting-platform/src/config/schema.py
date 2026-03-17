"""
Platform configuration schema.

All platform behaviour is driven by these dataclasses, loaded from YAML.
The schema is intentionally generic — hierarchy level names, metric choices,
model selections, and reconciliation strategies are all configuration, never
hard-coded to a specific LOB.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ExternalRegressorConfig:
    """Configuration for external regressors (promotions, holidays, price, etc.)."""
    enabled: bool = False
    feature_columns: List[str] = field(default_factory=list)
    future_features_path: Optional[str] = None
    # Maps column name → "known_ahead" | "contemporaneous".
    # "known_ahead" features (holidays, planned promos) can be forward-filled.
    # "contemporaneous" features (actual promo ratio, foot traffic) MUST have
    # explicit future values or they are dropped at prediction time.
    # Columns not listed default to "known_ahead" for backward compatibility.
    feature_types: Dict[str, str] = field(default_factory=dict)


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
class CalibrationConfig:
    """Prediction interval calibration settings."""
    enabled: bool = False                          # opt-in
    conformal_correction: bool = True              # apply conformal adjustment to production forecasts
    coverage_targets: Dict[str, float] = field(    # interval label → nominal coverage
        default_factory=lambda: {"80": 0.80}       # P10–P90 should cover 80%
    )
    # coverage_targets maps a label to a nominal rate. The quantile pair is inferred:
    # "80" → (0.10, 0.90), "90" → (0.05, 0.95), "50" → (0.25, 0.75)


@dataclass
class ConstraintConfig:
    """Capacity and business-rule constraints applied to forecasts."""
    enabled: bool = False
    min_demand: float = 0.0                          # floor (non-negativity by default)
    max_capacity: Optional[float] = None             # global per-series-per-week cap
    capacity_column: Optional[str] = None            # column name for per-series capacity
    aggregate_max: Optional[float] = None            # sum across all series per period
    proportional_redistribution: bool = True         # when aggregate exceeded: proportional vs clip-largest


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
    external_regressors: ExternalRegressorConfig = field(
        default_factory=ExternalRegressorConfig
    )
    calibration: CalibrationConfig = field(
        default_factory=CalibrationConfig
    )
    constraints: ConstraintConfig = field(
        default_factory=ConstraintConfig
    )


@dataclass
class HorizonBucket:
    """A named range of forecast steps for multi-horizon model selection."""
    name: str                    # "short", "medium", "long"
    start_step: int              # inclusive, 1-based
    end_step: int                # inclusive


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
    horizon_buckets: List[HorizonBucket] = field(default_factory=list)  # empty = single champion


@dataclass
class TransitionConfig:
    """Product transition stitching settings."""
    transition_window_weeks: int = 13    # 3-month overlap window
    ramp_shape: str = "linear"           # "linear" | "scurve" | "step"
    enable_overrides: bool = True        # read planner overrides from store
    override_store_path: str = "data/overrides.duckdb"


@dataclass
class CleansingConfig:
    """Demand cleansing settings — outlier detection, stockout imputation, period exclusion."""
    enabled: bool = False                          # opt-in; existing pipelines unaffected

    # Outlier detection
    outlier_method: str = "iqr"                    # "iqr" | "zscore"
    iqr_multiplier: float = 1.5                    # IQR fence multiplier (1.5 = standard)
    zscore_threshold: float = 3.0                  # z-score cutoff
    outlier_action: str = "clip"                   # "clip" | "interpolate" | "flag_only"

    # Stockout imputation
    stockout_detection: bool = True                # detect consecutive-zero runs → recovery
    min_zero_run: int = 2                          # min consecutive zeros to flag as stockout
    stockout_imputation: str = "seasonal"          # "seasonal" | "interpolate" | "none"

    # Period exclusion (COVID, warehouse fire, etc.)
    exclude_periods: List[Dict[str, str]] = field(default_factory=list)
    # Each entry: {"start": "2020-03-15", "end": "2020-06-30", "action": "interpolate"|"drop"|"flag"}

    # Output
    add_flag_columns: bool = True                  # add _outlier_flag, _stockout_flag, _excluded_flag


@dataclass
class StructuralBreakConfig:
    """Structural break detection settings."""
    enabled: bool = False                    # opt-in
    method: str = "cusum"                    # "pelt" | "cusum"
    min_segment_length: int = 13             # minimum weeks between breaks
    penalty: float = 3.0                     # PELT penalty (higher = fewer breaks)
    max_breakpoints: int = 5                 # cap on detected breaks per series
    truncate_to_last_break: bool = False     # if True, discard pre-break history
    cost_model: str = "rbf"                  # PELT cost function: "l2" | "rbf" | "normal"


@dataclass
class ValidationConfig:
    """Schema enforcement and data validation settings."""
    enabled: bool = False                    # opt-in
    require_columns: List[str] = field(default_factory=list)  # extra required beyond time/target/id
    check_duplicates: bool = True            # flag duplicate (series_id, time_col) pairs
    check_frequency: bool = True             # validate consistent weekly intervals
    check_non_negative: bool = True          # demand values >= 0
    min_value: Optional[float] = None        # custom floor (overrides non_negative if set)
    max_value: Optional[float] = None        # custom ceiling
    max_missing_pct: float = 100.0           # fail if any series exceeds this % missing weeks
    min_series_count: int = 1                # minimum number of series required
    strict: bool = False                     # if True, warnings become errors


@dataclass
class DataQualityReportConfig:
    """Pre-training data quality report settings."""
    enabled: bool = False                    # opt-in
    include_series_detail: bool = True       # per-series breakdown
    sparse_classification: bool = True       # run SBC demand classification


@dataclass
class DataQualityConfig:
    """Data quality and preprocessing settings."""
    fill_gaps: bool = True               # fill missing weeks with fill_value
    fill_value: float = 0.0              # value to use for gap-filling
    min_series_length_weeks: int = 52    # drop series shorter than this
    drop_zero_series: bool = False       # drop series with all-zero target
    validate_frequency: bool = False     # if True, raise on non-weekly gaps
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    cleansing: CleansingConfig = field(default_factory=CleansingConfig)
    structural_breaks: StructuralBreakConfig = field(default_factory=StructuralBreakConfig)
    report: DataQualityReportConfig = field(default_factory=DataQualityReportConfig)


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
class AnalysisConfig:
    """Data analysis module settings."""
    enabled: bool = True
    season_length: int = 52
    forecastability_signals: List[str] = field(
        default_factory=lambda: ["cv", "apen", "spectral_entropy", "snr",
                                  "trend_strength", "seasonal_strength"]
    )
    llm_enabled: bool = False            # opt-in for Anthropic API calls
    llm_model: str = "claude-sonnet-4-20250514"


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
    data_quality: DataQualityConfig = field(
        default_factory=DataQualityConfig
    )
    output: OutputConfig = field(default_factory=OutputConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
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
