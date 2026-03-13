# Forecasting Platform — Technical Specification

**Status:** Phase 4 complete
**Last updated:** 2026-03-13

---

## 1. Problem Statement

Weekly sales forecasting at scale for a retail / supply-chain S&OP process:

- Produce coherent weekly forecasts at multiple hierarchy levels (product group → category → SKU; region → country).
- Support new and discontinued SKUs via analogue mapping and proportional splitting.
- Run model comparison, champion selection, and automated exception flagging every planning cycle.
- Provide explainability to planners (narrative + decomposition) and data scientists (SHAP).
- Maintain a full governance trail (model cards, lineage, drift monitoring).
- Deploy to Microsoft Fabric / Delta Lake at scale; expose via REST API.
- Support distributed execution on PySpark for large LOBs.

---

## 2. System Layers

```
┌─────────────────────────────────────────────────────────────┐
│  REST API  (FastAPI)                  src/api/              │
│  GET /health  /forecast/{lob}  /metrics/leaderboard/{lob}  │
│  GET /forecast/{lob}/{series_id}  /metrics/drift/{lob}     │
├─────────────────────────────────────────────────────────────┤
│  Analytics Layer                      src/analytics/        │
│  ForecastAnalytics (notebook API)                           │
│  BIExporter (Power BI / Parquet)                            │
│  ForecastComparator · ExceptionEngine                       │
│  ForecastExplainer (STL + SHAP + narrative)                 │
│  DriftDetector · ModelCard · ModelCardRegistry              │
│  ForecastLineage                                            │
├─────────────────────────────────────────────────────────────┤
│  Pipeline Layer                       src/pipeline/         │
│  BacktestPipeline · ForecastPipeline                        │
├─────────────────────────────────────────────────────────────┤
│  Backtest Engine                      src/backtesting/      │
│  BacktestEngine · WalkForwardCV · ChampionSelector          │
├─────────────────────────────────────────────────────────────┤
│  Override Store                       src/overrides/        │
│  OverrideStore (DuckDB)                                     │
├─────────────────────────────────────────────────────────────┤
│  Hierarchy Layer                      src/hierarchy/        │
│  HierarchyTree · HierarchyNode · HierarchyAggregator        │
│  Reconciler (bottom_up · top_down · middle_out)             │
│            (ols · wls · mint)                               │
├─────────────────────────────────────────────────────────────┤
│  Model Library                        src/forecasting/      │
│  SeasonalNaive · AutoARIMA · AutoETS                        │
│  LGBMDirect · XGBoostDirect                                 │
│  Chronos · TimeGPT (zero-shot foundation)                   │
│  Croston · CrostonSBA · TSB (intermittent)                  │
│  WeightedEnsemble · ForecasterRegistry                      │
├─────────────────────────────────────────────────────────────┤
│  Metrics                              src/metrics/          │
│  MetricStore · ForecastDriftDetector · metric functions     │
├─────────────────────────────────────────────────────────────┤
│  SKU Mapping                          src/sku_mapping/      │
│  AttributeMatching · NamingConvention · CurveFitting        │
│  TemporalComovement · CandidateFusion                       │
│  BayesianProportionEstimator                                │
├─────────────────────────────────────────────────────────────┤
│  Series Management                    src/series/           │
│  SeriesBuilder · SparseDetector · TransitionEngine          │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                           src/data/             │
│  DataLoader · DataPreprocessor · FeatureEngineer (pandas)   │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure                       src/fabric/ src/spark/│
│  DeploymentOrchestrator · FabricLakehouse · DeltaWriter     │
│  SparkForecastPipeline · SparkSeriesBuilder                 │
│  SparkFeatureEngineer · SparkDataLoader                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Data Model

### 3.1 Panel DataFrame (core interchange format)

All production modules consume and produce Polars DataFrames in panel format:

| Column | Type | Description |
|--------|------|-------------|
| `series_id` | `Utf8` | Unique series identifier (e.g. `SKU_001_STORE_A`) |
| `week` | `Date` | ISO week start date (Monday) |
| `quantity` | `Float64` | Actual sales (history rows) |
| `forecast` | `Float64` | Point forecast / P50 |
| `p10` | `Float64` | 10th percentile (optional) |
| `p90` | `Float64` | 90th percentile (optional) |
| `lob` | `Utf8` | Line of business / segment |
| `model_id` | `Utf8` | Model that produced the forecast |

### 3.2 Hierarchy DataFrame

| Column | Type | Description |
|--------|------|-------------|
| `node_key` | `Utf8` | Node identifier |
| `level` | `Utf8` | Level name (e.g. `sku`, `category`, `total`) |
| `parent_key` | `Utf8` | Parent node key (null for root) |
| `parent_level` | `Utf8` | Parent level name |

### 3.3 MetricStore Schema

Parquet files partitioned by `run_type` (`backtest` | `live`) under `data/metrics/`:

| Column | Type |
|--------|------|
| `run_type` | `Utf8` |
| `model_id` | `Utf8` |
| `lob` | `Utf8` |
| `series_id` | `Utf8` |
| `target_week` | `Date` |
| `wmape` | `Float64` |
| `normalized_bias` | `Float64` |
| `rmspe` | `Float64` |
| `fold` | `Int32` |

### 3.4 Override Store Schema (DuckDB)

| Column | Type | Description |
|--------|------|-------------|
| `override_id` | `Utf8` | UUID |
| `series_id` | `Utf8` | Target series |
| `lob` | `Utf8` | Line of business |
| `week` | `Date` | Forecast week |
| `override_value` | `Float64` | Planner value |
| `override_type` | `Utf8` | `absolute` or `percent_change` |
| `reason` | `Utf8` | Free-text justification |
| `created_by` | `Utf8` | Planner identifier |
| `created_at` | `Datetime` | Timestamp |

---

## 4. Model Library

### 4.1 Common Interface (`src/forecasting/base.py`)

```python
class BaseForecaster(ABC):
    def fit(self, data: pl.DataFrame, config: ForecastConfig) -> "BaseForecaster": ...
    def predict(self, horizon: int, ...) -> pl.DataFrame: ...             # P50 point
    def predict_quantiles(self, horizon: int, quantiles: list[float]) -> pl.DataFrame: ...
    def get_params(self) -> Dict[str, Any]: ...
```

### 4.2 Statistical Models (`statistical.py`)

Both use `statsforecast` internally:

| Class | Algorithm | Default `season_length` |
|-------|-----------|------------------------|
| `AutoARIMAForecaster` | Auto-ARIMA with AIC selection | 52 |
| `AutoETSForecaster` | ETS with automatic error/trend/seasonal selection | 52 |

### 4.3 ML Models (`ml.py`)

Direct multi-step forecasting: one model per horizon step, trained on lagged features.

```python
class LGBMDirectForecaster(_DirectMLBase)
class XGBoostDirectForecaster(_DirectMLBase)
```

Both share `_DirectMLBase` which provides:
- `fit()` using `mlforecast` or manual lag construction
- `predict()` for point forecasts
- `predict_quantiles()` using quantile regression or residual bootstrap
- Configurable `lag_weeks`, `rolling_windows`, `horizon`

### 4.4 Foundation Models (`foundation.py`)

Zero-shot — no training required:

```python
class ChronosForecaster(BaseForecaster)    # Amazon Chronos
class TimeGPTForecaster(BaseForecaster)    # Nixtla TimeGPT (REST API)
```

`ChronosForecaster` loads a HuggingFace pipeline; `TimeGPTForecaster` calls the Nixtla REST API via `_get_client()`.

### 4.5 Intermittent Demand Models (`intermittent.py`)

| Class | Method | Key params |
|-------|--------|-----------|
| `CrostonForecaster` | Croston's original | `alpha=0.1` |
| `CrostonSBAForecaster` | Syntetos-Boylan-Adelson bias correction | inherits from Croston |
| `TSBForecaster` | Teunter-Syntetos-Babai | `alpha_z=0.1`, `alpha_p=0.1` |

Internal helpers: `_croston_fit()`, `_tsb_fit()` (numpy-based, no external dependency).

### 4.6 Ensemble (`ensemble.py`)

```python
class WeightedEnsembleForecaster(BaseForecaster):
    def __init__(self, models: list, weights: list[float], quantiles: list[float] = [0.1, 0.5, 0.9])
```

Produces probabilistic forecasts by weighted mixture; quantiles via bootstrap of base model residuals.

### 4.7 ForecasterRegistry (`registry.py`)

```python
registry = ForecasterRegistry()
registry.register("lgbm", LGBMDirectForecaster)
model = registry.build("lgbm", horizon=13)
model = registry.build_from_config(config)  # reads config.models list
```

---

## 5. Series Management

### 5.1 SparseDetector (`series/sparse_detector.py`)

Classifies each series using **Average Demand Interval (ADI)** and **Coefficient of Variation squared (CV²)**:

| Class | CV² | ADI | Model routing |
|-------|-----|-----|--------------|
| Smooth | ≤ 0.49 | ≤ 1.32 | `AutoETS` or `LGBMDirect` |
| Intermittent | ≤ 0.49 | > 1.32 | `CrostonSBA` |
| Erratic | > 0.49 | ≤ 1.32 | `TSB` |
| Lumpy | > 0.49 | > 1.32 | `TSB` or `WeightedEnsemble` |

`classify(df)` → per-series classification DataFrame
`split(df)` → two DataFrames: smooth-class and intermittent-class

### 5.2 TransitionEngine (`series/transition.py`)

Determines which transition scenario applies to each (old SKU, new SKU) pair and returns stitched time series:

```python
class TransitionScenario(Enum):
    A_LAUNCHED      # new SKU already live → stitch history
    B_IN_HORIZON    # launches within transition_window_weeks → ramp
    C_BEYOND_HORIZON # launches after horizon end → forecast old only
    MANUAL          # planner override

class TransitionPlan:  # dataclass
    old_sku, new_sku, scenario, proportion, ramp_start, ramp_end, ramp_shape, notes

class TransitionEngine:
    def compute_plans(mapping_table, product_master, forecast_origin) -> List[TransitionPlan]
    def stitch_series(actuals, plans) -> pl.DataFrame
```

Ramp shapes: `linear`, `scurve`, `step`.

---

## 6. Hierarchical Reconciliation

### 6.1 Summing Matrix S

`S` has shape `(n_all × n_leaves)` where `S[i, j] = 1` if leaf `j` is in the subtree of node `i`.
Leaf rows of `S` form an identity sub-matrix (`S[leaf_i, i] = 1`).

Built by `HierarchyTree.summing_matrix()` as a Polars DataFrame; converted to numpy for reconciliation arithmetic.

### 6.2 Linear Reconciliation (OLS / WLS / MinT)

```
G      = (S′W⁻¹S)⁻¹ S′W⁻¹     # projection onto coherent subspace
P̃_leaf = G · P̂_all             # reconciled leaf forecasts  (n_leaves × T)
P̃_all  = S · P̃_leaf            # all-level coherent forecasts (n_all × T)
```

Regularisation: `(S′W⁻¹S + λI)` with `λ = 1e-6` for numerical stability.
Non-negativity: `clip(0)` on leaf forecasts after projection.

### 6.3 W Matrix Variants

| Method | W | `residuals` needed? |
|--------|---|---------------------|
| `ols` | Identity `I_{n_all}` | No |
| `wls` (structural) | `diag(n_leaf_descendants_per_node)` | No |
| `wls` (residual) | `diag(per-series residual variance)` | Yes |
| `mint` | Ledoit–Wolf diagonal shrinkage | Recommended; falls back to WLS-structural |

### 6.4 MinT Ledoit–Wolf Shrinkage

```
λ* = min(1, n / T)                         # shrinkage intensity; n=series, T=time obs
Σ_shrunk = (1 - λ*) · Σ_sample + λ* · diag(Σ_sample)
W = diag(Σ_shrunk)                         # diagonal only, for stability
```

When `T < n` (fewer time observations than series), `λ* = 1` → pure diagonal → equivalent to WLS-residual.

### 6.5 Reconciler API

```python
reconciler = Reconciler(tree)
result = reconciler.reconcile(
    forecasts,                      # pl.DataFrame with [node_key, level, week, forecast, ...]
    method="mint",                  # bottom_up | top_down | middle_out | ols | wls | mint
    residuals=residuals_df,         # optional; enables residual-variance WLS and MinT
    value_columns=["forecast"],     # which columns to reconcile
    time_column="week",
)
```

---

## 7. Backtesting

### 7.1 Walk-Forward Protocol (`backtesting/cross_validator.py`)

`WalkForwardCV` generates expanding-window folds:

```
Fold 1: train=[0..T₁],   test=[T₁..T₁+H]
Fold 2: train=[0..T₂],   test=[T₂..T₂+H]
...
```

Config: `n_folds`, `horizon_weeks`, `min_train_weeks`, `step_weeks`.

### 7.2 BacktestEngine

```python
engine = BacktestEngine(config)
results = engine.run(models=[lgbm, naive], data=panel_df)
# results: per-(model, fold, series, week) DataFrame written to MetricStore
```

`_run_one(model, train_df, test_df)` fits the model and returns metric rows for one fold.

### 7.3 Champion Selection

```python
selector = ChampionSelector(config)
champion_id = selector.select(metric_store)
# Ranks by mean WMAPE across all folds and series (ascending).
# Tie-break: normalized_bias closest to 0.

weights = selector.compute_ensemble_weights(metric_store)
# Returns softmax-normalised inverse-WMAPE weights for ensemble construction.
```

---

## 8. Metrics

### 8.1 Metric Functions (`metrics/definitions.py`)

```python
wmape(actual, forecast)           # Weighted MAE / sum(actual)
normalized_bias(actual, forecast) # mean(forecast - actual) / mean(actual)
mape(actual, forecast)            # mean |error / actual|
mae(actual, forecast)             # mean |error|
rmse(actual, forecast)            # sqrt(mean(error²))
compute_all_metrics(actual, forecast) -> dict
```

### 8.2 MetricStore (`metrics/store.py`)

Append-only Parquet files partitioned by `{base_path}/{run_type}/{lob}/metrics_{timestamp}.parquet`.

```python
store = MetricStore("data/metrics/")
store.write(records_df, run_type="backtest", lob="retail")
store.read(run_type="backtest", lob="retail", model_id="lgbm") -> pl.DataFrame
store.leaderboard(lob, run_type, primary_metric="wmape") -> pl.DataFrame
store.accuracy_over_time(model_id, lob) -> pl.DataFrame
```

### 8.3 ForecastDriftDetector (`metrics/drift.py`)

Detects three drift types by comparing a `baseline_window` to a `recent_window`:

| Method | Detects |
|--------|---------|
| `detect_accuracy_drift(df)` | WMAPE of recent weeks > threshold vs baseline |
| `detect_bias_drift(df)` | Signed bias shifted significantly |
| `detect_volume_anomaly(df)` | Actual volume outside expected range |
| `detect(df)` | Runs all three; returns `List[DriftAlert]` |
| `summary(df)` | Aggregated drift DataFrame per series |

`DriftSeverity`: `warning` or `critical`.
`DriftConfig(baseline_weeks=26, recent_weeks=8)`.

---

## 9. Analytics Layer

### 9.1 ForecastAnalytics (`analytics/notebook_api.py`)

Notebook-ready queries over `MetricStore`. All methods return `pl.DataFrame`.

```python
fa = ForecastAnalytics("data/metrics/")
fa.model_leaderboard(lob, run_type, primary_metric, secondary_metric, grain_level)
fa.model_comparison_by_fold(lob)
fa.accuracy_over_time(model, channel)
fa.accuracy_by_grain(lob)
fa.bias_distribution(lob)
fa.transition_impact(lob)
fa.backtest_vs_live(lob)
```

### 9.2 BIExporter (`analytics/bi_export.py`)

Writes Hive-partitioned Parquet for direct Power BI consumption:

```
bi_exports/
├── forecast_vs_actual/lob=retail/<run_date>.parquet
├── model_leaderboard/lob=retail/<run_date>.parquet
└── bias_report/lob=retail/<run_date>.parquet
```

```python
exporter = BIExporter("data/bi_exports/")
exporter.export_forecast_vs_actual(forecasts, actuals, lob)   -> Path
exporter.export_leaderboard(lob)                               -> Path
exporter.export_bias_report(lob)                               -> Path
```

### 9.3 ForecastComparator (`analytics/comparator.py`)

```python
comparator = ForecastComparator()
result = comparator.compare(
    model_forecast,                          # [series_id, week, forecast, p10, p90]
    external_forecasts={"field": field_df},  # any number of named external sources
    prior_model_forecast=last_cycle_df,
    id_col="series_id", time_col="week", value_col="forecast",
)
```

Output columns added to `model_forecast`:
- `{name}_forecast`, `{name}_gap`, `{name}_gap_pct` — per external source
- `prior_model_forecast`, `cycle_change`, `cycle_change_pct`
- `uncertainty_ratio` = (p90 − p10) / p50 (null when p10/p90 not present)

```python
summary = comparator.summary(result)   # one row per series_id; mean of gap columns
```

### 9.4 ExceptionEngine (`analytics/exceptions.py`)

```python
engine = ExceptionEngine(
    cycle_change_pct_threshold=20.0,
    uncertainty_ratio_threshold=0.50,
    field_disagree_pct_threshold=25.0,
    overforecast_pct_threshold=30.0,
    underforecast_pct_threshold=-30.0,
)
flagged = engine.flag(comparison_df)
# Adds: exc_large_cycle_change, exc_high_uncertainty, exc_field_disagree,
#       exc_overforecast, exc_underforecast, exc_no_prior, has_exception

summary = engine.exception_summary(flagged)
# One row per series; n_weeks_exc_* counts; sorted by total_exception_weeks desc
```

### 9.5 ForecastExplainer (`analytics/explainer.py`)

```python
explainer = ForecastExplainer(season_length=52, trend_window=12)
```

**`decompose(history, forecast, id_col, time_col, target_col, value_col)`**

Returns `[id_col, time_col, value, trend, seasonal, residual, is_forecast]`.

Algorithm:
1. Trend = centered moving average with window = `season_length`; NaN at edges
2. Seasonal = `mean(value - trend)` per seasonal position (0..season_length-1)
3. Residual = value − trend − seasonal
4. Forecast trend = linear fit over last `trend_window` non-NaN trend values; extrapolated by `slope × h`
5. Forecast seasonal = `seasonal_avg[( n + h - 1) % season_length]`

**`explain_ml(model, features_df, id_col, time_col, top_k=5)`**

- Resolves feature names from `model.feature_names_in_` (sklearn) or `model.feature_name_()` (LightGBM native)
- Uses `shap.Explainer(model, X)` to compute SHAP values
- Returns tidy `[id_col, time_col, feature, shap_value, rank]`
- Returns empty DataFrame (with `shap_unavailable` column) if `shap` not installed

**`narrative(decomposition, comparison, id_col, time_col)`**

Template (per series):
```
"Series {sid}: forecast is {X}% above/below last year, primarily driven by {trend|seasonality}.
 System is {Y}% above/below {source} forecast on average.
 Model uncertainty is HIGH — review P10/P90 range."
```

YoY comparison uses last `season_length` periods of history.
Primary driver = trend if |trend_share| ≥ |seasonal_share|, else seasonality.
Uncertainty label: > 0.75 → HIGH, > 0.40 → moderate, else low.

### 9.6 DriftDetector (`analytics/governance.py`)

```python
detector = DriftDetector(
    metric_store,
    warn_multiplier=1.25,
    alert_multiplier=1.50,
    min_live_weeks=4,
)
result = detector.detect(model_id, lob, metric="wmape", n_recent_weeks=None)
# Returns: {model_id, lob, metric, backtest_score, live_score, ratio, status, n_live_weeks}
# status: "ok" | "warning" | "alert" | "insufficient_data"

df = detector.batch_detect(lob, metric="wmape")
# One row per model_id, sorted by ratio descending (most degraded first)
```

Status thresholds:
```
ratio ≤ warn_multiplier                          → "ok"
warn_multiplier < ratio ≤ alert_multiplier       → "warning"
ratio > alert_multiplier                         → "alert"
n_live_weeks < min_live_weeks or no backtest     → "insufficient_data"
```

### 9.7 ModelCard (`analytics/governance.py`)

```python
@dataclass
class ModelCard:
    model_name: str
    lob: str
    training_start: Optional[date]
    training_end: Optional[date]
    n_series: int
    n_observations: int
    backtest_wmape: Optional[float]
    backtest_bias: Optional[float]
    champion_since: Optional[date]
    features: List[str]
    config_hash: str          # MD5[:8] of serialized config
    notes: str

    @classmethod
    def from_backtest(cls, model_name, lob, backtest_results, champion_since,
                      features, config, notes) -> "ModelCard": ...
    def to_dict(self) -> Dict: ...   # ISO date strings; JSON-safe
    def to_frame(self) -> pl.DataFrame: ...  # single-row DataFrame
```

`config_hash` = `MD5(json.dumps(asdict(config), sort_keys=True))[:8]`.

### 9.8 ModelCardRegistry

```python
registry = ModelCardRegistry("data/model_cards/")
registry.register(card)            # upsert by model_name; persists to Parquet
registry.get("lgbm_direct")        # returns ModelCard or None
registry.all_cards()               # pl.DataFrame of all cards
```

Parquet file: `data/model_cards/model_cards.parquet` (single flat file, rewritten on each `register()`).

### 9.9 ForecastLineage

Append-only; one Parquet file per `record()` call:
`data/lineage/lineage_{lob}_{run_date}_{model_id[:12]}.parquet`

```python
lineage = ForecastLineage("data/lineage/")
lineage.record(
    lob, model_id, n_series, horizon_weeks,
    selection_strategy="champion", run_id="", notes="",
    run_date=None,   # defaults to date.today()
)
lineage.history(lob=None, model_id=None) -> pl.DataFrame  # sorted run_date desc
lineage.latest(lob) -> Optional[Dict]
```

Schema: `run_date`, `lob`, `model_id`, `selection_strategy`, `n_series`, `horizon_weeks`, `run_id`, `notes`.

---

## 10. Planner Override Store

```python
store = OverrideStore("data/overrides.duckdb")
store.add_override(series_id, lob, week, override_value, override_type,
                   reason="", created_by="")
store.get_overrides(lob=None, series_id=None, week=None) -> pl.DataFrame
store.get_all() -> pl.DataFrame
store.delete_override(override_id) -> bool
store.close()
```

`override_type`: `"absolute"` (replace forecast with value) or `"percent_change"` (multiply by `1 + value/100`).

---

## 11. SKU Mapping

### 11.1 Mapping Methods

All inherit from `BaseMethod` and implement `run(product_master) -> List[MappingCandidate]`.

| Class | Similarity metric | Key params |
|-------|------------------|-----------|
| `AttributeMatchingMethod` | Cosine similarity on numeric attribute vectors | `launch_window_days=365` |
| `NamingConventionMethod` | Jaccard token overlap + base-name/marker parsing | `min_score` threshold |
| `CurveFittingMethod` | R² of S-curve / step-ramp shape fit to sales trajectory | ramp shape config |
| `TemporalCovementMethod` | Pearson correlation of week-over-week growth rates | `min_overlap_weeks` |

### 11.2 CandidateFusion

```python
fusion = CandidateFusion(weights={"attribute": 0.3, "naming": 0.3,
                                   "curve": 0.2, "temporal": 0.2})
records = fusion.fuse(candidates_by_method)
# Assigns mapping_type: one_to_one | one_to_many
# Assigns proportion (from BayesianProportionEstimator for one_to_many)
# Assigns confidence: high | medium | low
```

### 11.3 BayesianProportionEstimator

For 1-to-many splits (one old SKU → multiple new SKUs):

```python
estimator = BayesianProportionEstimator(concentration=0.5)
updated_records = estimator.estimate(mapping_records)
# Prior: Dirichlet(α = concentration * uniform)
# Update: use method scores as pseudo-observations
# Posterior mean = proportion weights
# Updates MappingRecord.proportion for each candidate in the group
```

---

## 12. REST API

### 12.1 Endpoints

Built with FastAPI; Swagger UI at `/docs`, ReDoc at `/redoc`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe; returns `{"status": "ok", "version": "..."}` |
| GET | `/forecast/{lob}` | Latest forecast Parquet for a LOB; query params: `series_id`, `horizon` |
| GET | `/forecast/{lob}/{series_id}` | Latest forecast for a single series |
| GET | `/metrics/leaderboard/{lob}` | Model leaderboard; query param: `run_type` (default `backtest`) |
| GET | `/metrics/drift/{lob}` | Drift alerts; query params: `run_type`, `baseline_weeks` (26), `recent_weeks` (8) |

### 12.2 Response Schemas

```python
HealthResponse:    status, version
ForecastPoint:     series_id, week, forecast, model, lob
ForecastResponse:  lob, series_count, forecast_origin, points: List[ForecastPoint]
LeaderboardEntry:  model, wmape, normalized_bias, rank, n_series
LeaderboardResponse: lob, run_type, entries: List[LeaderboardEntry]
DriftAlertItem:    series_id, metric, severity, current_value, baseline_value, message
DriftResponse:     lob, n_critical, n_warning, alerts: List[DriftAlertItem]
```

### 12.3 App Factory

```python
from src.api.app import create_app
app = create_app(data_dir="data/", metrics_dir="data/metrics/")
# Or: uvicorn src.api.app:app --port 8000
```

Environment variables: `API_DATA_DIR`, `API_METRICS_DIR`, `API_VERSION`.

Forecast files expected at: `{data_dir}/forecasts/{lob}/forecast_*.parquet` (most recent is served).

---

## 13. Microsoft Fabric / Delta Lake

### 13.1 DeploymentOrchestrator

```python
orchestrator = DeploymentOrchestrator(spark, platform_config, deploy_config)
result: DeploymentResult = orchestrator.run(actuals_sdf)
```

Steps:
1. `_preflight()` — validate config, check Lakehouse connectivity
2. `_resolve_champion()` — run backtest or read existing champion from audit log
3. Run forecast pipeline via `SparkForecastPipeline`
4. `_postrun_check()` — validate output row count
5. `_write_forecasts()` — upsert to Delta via `DeltaWriter`
6. `_write_leaderboard()` — write backtest metrics
7. `_write_audit_log()` — append `DeploymentResult` to audit table

### 13.2 FabricLakehouse

```python
lh = FabricLakehouse(spark, config)
df = lh.read_table("forecasts", schema={"series_id": pl.Utf8, ...})
lh.write_table(df, "forecasts", mode="overwrite")
lh.vacuum("forecasts", retention_hours=168)
lh.optimize("forecasts", z_order_by=["series_id", "week"])
lh.history("forecasts", limit=10)
```

### 13.3 DeltaWriter

```python
writer = DeltaWriter(spark, config)
writer.upsert(sdf, table_name, merge_keys=["series_id", "week"])
writer.overwrite_partition(sdf, table_name, partition_col="lob", partition_val="retail")
writer.append(sdf, table_name)
writer.write_forecasts(forecasts_sdf, lob, forecast_origin)
```

### 13.4 FabricConfig

```python
config = FabricConfig.from_env()       # reads FABRIC_WORKSPACE_ID, FABRIC_LAKEHOUSE_ID, etc.
config = FabricConfig.from_dict(d)
config.abfss_base   # -> "abfss://workspace_id@onelake.dfs.fabric.microsoft.com/lakehouse_id/"
config.table_path("forecasts")
```

---

## 14. Spark Distributed Execution

### 14.1 SparkForecastPipeline

```python
pipeline = SparkForecastPipeline(spark, platform_config)
forecasts_sdf = pipeline.run_forecast(actuals_sdf, champion_model, horizon)
backtest_sdf  = pipeline.run_backtest(actuals_sdf, models, n_folds, horizon)
champion_id   = pipeline.select_champion(backtest_sdf, primary_metric="wmape")
```

### 14.2 SparkSeriesBuilder

```python
builder = SparkSeriesBuilder.from_config(config_dict)
series_sdf = builder.build(raw_sdf)
# Aggregates to weekly panel; fills zero-sales gaps; assigns series_id
```

### 14.3 Utilities (`spark/utils.py`)

```python
polars_to_spark(polars_df, spark)         # preserves schema
spark_to_polars(spark_df)                 # converts to Polars
repartition_by_series(df, series_id_col)  # optimal parallelism
abfss_uri(workspace_id, lakehouse_id, subpath)
```

### 14.4 Session Factory (`spark/session.py`)

```python
spark = get_or_create_spark(app_name="ForecastingPlatform", config_overrides={})
# Auto-detects Fabric environment via notebookutils; configures Delta extensions
```

---

## 15. Configuration Schema (`src/config/schema.py`)

```yaml
lob: retail
horizon_weeks: 13
season_length: 52
min_train_weeks: 104

models:
  - type: lgbm
    lag_weeks: [1, 2, 4, 8, 13, 26, 52]
    rolling_windows: [4, 13, 26]
  - type: naive
  - type: ensemble
    weights: [0.7, 0.3]
    quantiles: [0.1, 0.5, 0.9]

backtest:
  n_folds: 4
  horizon_weeks: 13
  step_weeks: 13
  min_train_weeks: 104

hierarchies:
  - name: product
    levels:
      - name: total
      - name: category
      - name: sku     # leaf
    reconciliation:
      method: mint      # bottom_up | top_down | middle_out | ols | wls | mint
      non_negative: true

transition:
  transition_window_weeks: 8
  ramp_shape: scurve    # linear | scurve | step

output:
  forecast_dir: data/forecasts/
  metrics_dir: data/metrics/
  bi_export_dir: data/bi_exports/
```

Loaded via `load_config(path)` or `load_config_with_overrides(base_path, override_path)`.
All nested dicts deep-merged; override values take precedence.

---

## 16. Test Coverage

| Test file | Tests | Key classes / scenarios |
|-----------|------:|------------------------|
| `test_platform.py` | 85 | Config schema, hierarchy tree/aggregator, reconciliation (basic), metrics, MetricStore, TransitionEngine ramp shapes, ForecasterRegistry, NaiveForecaster, WalkForwardCV, ChampionSelector, end-to-end, DeploymentConfig, REST API, ForecastDriftDetector |
| `test_sku_mapping.py` | 67 | MockGenerator, AttributeMatching, NamingConvention (base+marker extraction), CandidateFusion, MappingWriter, end-to-end pipeline, CurveFitting, TemporalComovement, BayesianProportionEstimator |
| `test_forecast_explainability.py` | 59 | ForecastComparator (11), ExceptionEngine (12), ForecastExplainerDecompose (7), SHAP fallback (2), Narrative (5), ModelCard (5), ModelCardRegistry (4), DriftDetector (6), ForecastLineage (6) |
| `test_intermittent_demand.py` | 55 | SparseDetectorClassify/Split, CrostonForecaster, CrostonSBAForecaster, TSBForecaster, `_croston_fit`, `_tsb_fit`, backtest routing, IntermittentRegistry |
| `test_foundation_models.py` | 41 | ChronosForecaster fit/predict/quantiles, TimeGPTForecaster fit/predict/quantiles, error handling, FoundationModelRegistry, zero-shot property |
| `test_mint_reconciliation.py` | 46 | S-matrix shape/identity/arithmetic, OLS output/non-neg/multi-level, WLS structural/residual, MinT no-residuals fallback/with-residuals/shrinkage/coherence, multi-value-columns, edge cases (single leaf, all-zero, missing levels), G·S=I property |
| `test_probabilistic_ensemble.py` | 24 | NaivePredictQuantiles, WeightedEnsembleForecaster, compute_ensemble_weights, SelectionStrategyConfig, ForecastConfigQuantiles, BacktestPipelineEnsemble |
| `test_feature_engineering.py` | 3 | FeatureEngineer (pandas) |
| `test_metrics.py` | 6 | Metric functions (pandas) |
| **Total** | **386 core + 9 pandas = 391** | |
