"""
Platform configuration schema.

All platform behaviour is driven by these dataclasses, loaded from YAML.
The schema is intentionally generic — hierarchy level names, metric choices,
model selections, and reconciliation strategies are all configuration, never
hard-coded to a specific LOB.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Frequency profiles — single source of truth for frequency-dependent defaults
# ---------------------------------------------------------------------------
# Each supported frequency maps to a profile dict containing:
#   season_length     — primary seasonal cycle length in periods
#   secondary_season  — optional secondary cycle (e.g. yearly for daily data)
#   default_lags      — lag set for ML models
#   min_series_length  — minimum history length (periods) to train on
#   default_val_periods — default backtest validation window (periods)
#   default_horizon    — default forecast horizon (periods)
#   statsforecast_freq — freq string accepted by statsforecast / neuralforecast
#   timedelta_kwargs   — kwargs for datetime.timedelta representing one period
# ---------------------------------------------------------------------------
FREQUENCY_PROFILES: Dict[str, Dict[str, Any]] = {
    "D": {
        "season_length": 7,
        "secondary_season": 365,
        "default_lags": [1, 2, 3, 7, 14, 21, 28, 56, 91, 182, 364],
        "min_series_length": 90,
        "default_val_periods": 28,
        "default_horizon": 90,
        "statsforecast_freq": "D",
        "timedelta_kwargs": {"days": 1},
    },
    "W": {
        "season_length": 52,
        "secondary_season": None,
        "default_lags": [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 16, 20, 26, 52],
        "min_series_length": 52,
        "default_val_periods": 13,
        "default_horizon": 39,
        "statsforecast_freq": "W",
        "timedelta_kwargs": {"weeks": 1},
    },
    "M": {
        "season_length": 12,
        "secondary_season": None,
        "default_lags": [1, 2, 3, 6, 12],
        "min_series_length": 24,
        "default_val_periods": 3,
        "default_horizon": 12,
        "statsforecast_freq": "MS",
        "timedelta_kwargs": {"days": 30},
    },
    "Q": {
        "season_length": 4,
        "secondary_season": None,
        "default_lags": [1, 2, 4, 8],
        "min_series_length": 8,
        "default_val_periods": 2,
        "default_horizon": 8,
        "statsforecast_freq": "QS",
        "timedelta_kwargs": {"days": 91},
    },
}


def get_frequency_profile(freq: str) -> Dict[str, Any]:
    """Look up the frequency profile; raise on unknown frequency.

    Parameters
    ----------
    freq : str
        One of ``"D"``, ``"W"``, ``"M"``, ``"Q"``.

    Returns
    -------
    dict
        Profile dict with season_length, default_lags, statsforecast_freq, etc.
    """
    if freq not in FREQUENCY_PROFILES:
        raise ValueError(
            f"Unsupported frequency {freq!r}. "
            f"Choose from: {sorted(FREQUENCY_PROFILES)}"
        )
    return FREQUENCY_PROFILES[freq]


def freq_timedelta(freq: str, periods: int = 1) -> timedelta:
    """Return a ``timedelta`` for *periods* steps at the given frequency.

    Parameters
    ----------
    freq : str
        One of ``"D"``, ``"W"``, ``"M"``, ``"Q"``.
    periods : int
        Number of periods (default 1).
    """
    kwargs = get_frequency_profile(freq)["timedelta_kwargs"]
    return timedelta(**{k: v * periods for k, v in kwargs.items()})


@dataclass
class RegressorScreenConfig:
    """Pre-training regressor quality screening."""
    enabled: bool = False
    variance_threshold: float = 1e-6       # drop features with variance below this
    correlation_threshold: float = 0.95    # warn if pairwise |correlation| exceeds this
    mi_enabled: bool = False               # mutual information check (requires sklearn)
    mi_threshold: float = 0.01             # warn if MI with target below this
    auto_drop: bool = True                 # automatically drop flagged columns


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
    screen: RegressorScreenConfig = field(default_factory=RegressorScreenConfig)


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
    """Forecast horizon and model selection.

    The ``frequency`` field (``"D"``, ``"W"``, ``"M"``, ``"Q"``) drives
    downstream defaults — season length, ML lags, statsforecast freq string,
    backtest window sizing, and date arithmetic.  See :data:`FREQUENCY_PROFILES`.

    ``horizon_weeks`` is kept for backward-compatible YAML loading; it
    represents *periods* regardless of frequency.  Use the
    :pyattr:`horizon_periods` alias for clarity.
    """
    horizon_weeks: int = 39              # periods (name kept for YAML compat)
    frequency: str = "W"                 # "D" | "W" | "M" | "Q"
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

    # -- Computed helpers (not stored in YAML) --

    @property
    def horizon_periods(self) -> int:
        """Alias for ``horizon_weeks`` (periods, frequency-agnostic)."""
        return self.horizon_weeks

    @property
    def season_length(self) -> int:
        """Primary seasonal cycle length derived from ``frequency``."""
        return get_frequency_profile(self.frequency)["season_length"]

    @property
    def statsforecast_freq(self) -> str:
        """Frequency string accepted by statsforecast / neuralforecast."""
        return get_frequency_profile(self.frequency)["statsforecast_freq"]

    @property
    def default_lags(self) -> List[int]:
        """Default ML lag set for the configured frequency."""
        return list(get_frequency_profile(self.frequency)["default_lags"])


@dataclass
class HorizonBucket:
    """A named range of forecast steps for multi-horizon model selection."""
    name: str                    # "short", "medium", "long"
    start_step: int              # inclusive, 1-based
    end_step: int                # inclusive


@dataclass
class BacktestConfig:
    """Walk-forward cross-validation settings.

    ``val_weeks`` and ``gap_weeks`` represent *periods* (not calendar weeks)
    when the platform runs at non-weekly frequencies.  Names are kept for
    backward-compatible YAML loading; use :pyattr:`val_periods` /
    :pyattr:`gap_periods` aliases for clarity.
    """
    n_folds: int = 3
    val_weeks: int = 13                  # periods per fold (name kept for compat)
    gap_weeks: int = 0                   # gap periods between train end and val start
    champion_granularity: str = "lob"    # "lob" | "product_group" | "series"
    primary_metric: str = "wmape"
    secondary_metric: str = "normalized_bias"
    selection_strategy: str = "champion" # "champion" | "weighted_ensemble"
    horizon_buckets: List[HorizonBucket] = field(default_factory=list)  # empty = single champion

    @property
    def val_periods(self) -> int:
        """Alias for ``val_weeks`` (periods, frequency-agnostic)."""
        return self.val_weeks

    @property
    def gap_periods(self) -> int:
        """Alias for ``gap_weeks`` (periods, frequency-agnostic)."""
        return self.gap_weeks


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
    """Data quality and preprocessing settings.

    ``min_series_length_weeks`` represents *periods* (not calendar weeks)
    when the platform runs at non-weekly frequencies.
    """
    fill_gaps: bool = True               # fill missing periods with fill_value
    fill_value: float = 0.0              # value to use for gap-filling
    min_series_length_weeks: int = 52    # periods (name kept for YAML compat)
    drop_zero_series: bool = False       # drop series with all-zero target
    validate_frequency: bool = False     # if True, raise on inconsistent intervals

    @property
    def min_series_length(self) -> int:
        """Alias for ``min_series_length_weeks`` (periods, frequency-agnostic)."""
        return self.min_series_length_weeks
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
class AIConfig:
    """AI feature settings (Claude integration for AI-native endpoints)."""
    enabled: bool = False                     # master switch for all AI endpoints
    api_key_env_var: str = "ANTHROPIC_API_KEY"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2000


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
    ai: AIConfig = field(default_factory=AIConfig)
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
