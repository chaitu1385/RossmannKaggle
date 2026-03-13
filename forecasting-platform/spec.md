# Forecasting Platform — Technical Specification

Version: Phase 4 complete
Last updated: 2026-03-13

---

## 1. Problem Statement

Weekly sales forecasting at scale for a retail / supply-chain S&OP process. Key requirements:

- Produce coherent weekly forecasts at multiple hierarchy levels (product group → category → SKU; region → country).
- Support new and discontinued SKUs via analogue mapping.
- Run model comparison, champion selection, and automated exception flagging every planning cycle.
- Provide explainability to planners (narrative + decomposition) and data scientists (SHAP).
- Maintain a full governance trail (model cards, lineage, drift monitoring).
- Deploy to Microsoft Fabric / Delta Lake at scale; expose via REST API.

---

## 2. System Layers

```
┌─────────────────────────────────────────────────────────┐
│  REST API  (FastAPI)          src/api/                  │
├─────────────────────────────────────────────────────────┤
│  Analytics Layer              src/analytics/            │
│   ForecastComparator · ExceptionEngine                  │
│   ForecastExplainer · DriftDetector                     │
│   ModelCard · ModelCardRegistry · ForecastLineage       │
├─────────────────────────────────────────────────────────┤
│  Pipeline Layer               src/pipeline/             │
│   BacktestPipeline · ForecastPipeline                   │
├─────────────────────────────────────────────────────────┤
│  Backtest Engine              src/backtesting/          │
│   BacktestEngine · CrossValidator · ChampionSelector    │
├─────────────────────────────────────────────────────────┤
│  Hierarchy Layer              src/hierarchy/            │
│   HierarchyTree · HierarchyAggregator · Reconciler      │
│   (bottom_up · top_down · middle_out · OLS · WLS · MinT)│
├─────────────────────────────────────────────────────────┤
│  Model Library                src/forecasting/          │
│   Naive · Statistical · ML · Foundation                 │
│   Intermittent · Ensemble                               │
├─────────────────────────────────────────────────────────┤
│  SKU Mapping                  src/sku_mapping/          │
│   AttributeMatching · NamingParsing · CurveFitting      │
│   TemporalComovement · BayesianProportions              │
├─────────────────────────────────────────────────────────┤
│  Data Layer                   src/data/ · src/series/   │
│   Loader · Preprocessor · FeatureEngineering            │
│   SeriesBuilder · SparseDetector · TransitionHandler    │
├─────────────────────────────────────────────────────────┤
│  Infrastructure               src/fabric/ · src/spark/  │
│   FabricDeployment · DeltaWriter · SparkPipeline        │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Data Model

### 3.1 Panel DataFrame (core format)

All modules consume and produce Polars DataFrames in panel format:

| Column | Type | Description |
|--------|------|-------------|
| `series_id` | `Utf8` | Unique series identifier (e.g. `SKU_001_STORE_A`) |
| `week` | `Date` | ISO week start date (Monday) |
| `quantity` | `Float64` | Actual sales (history) |
| `forecast` | `Float64` | Point forecast (P50) |
| `p10` | `Float64` | 10th percentile forecast (optional) |
| `p90` | `Float64` | 90th percentile forecast (optional) |
| `lob` | `Utf8` | Line of business / segment |
| `model_id` | `Utf8` | Model that produced the forecast |

### 3.2 Hierarchy DataFrame

| Column | Type | Description |
|--------|------|-------------|
| `node_key` | `Utf8` | Node identifier |
| `level` | `Utf8` | Level name (e.g. `sku`, `category`, `total`) |
| `parent_key` | `Utf8` | Parent node key (null for root) |
| `parent_level` | `Utf8` | Parent level name |

### 3.3 Metric Store Schema

Stored as Parquet partitioned by `run_type` (`backtest` | `live`):

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

---

## 4. Model Library

### 4.1 Common Interface (`src/forecasting/base.py`)

```python
class BaseForecaster:
    def fit(self, data: pl.DataFrame, config: ForecastConfig) -> "BaseForecaster": ...
    def predict(self, horizon: int, series_ids: list[str]) -> pl.DataFrame: ...
    def predict_quantiles(self, horizon: int, quantiles: list[float]) -> pl.DataFrame: ...
```

### 4.2 Model Implementations

| Model | Key Config | Notes |
|-------|-----------|-------|
| `NaiveForecaster` | `season_length` | Last-year same-week seasonal naïve |
| `StatisticalForecaster` | `trend`, `seasonal`, `damping` | ETS with additive/multiplicative options |
| `MLForecaster` | `model_type` (`lgbm`/`xgb`), `horizon`, `lag_weeks`, `rolling_windows` | Direct multi-step with feature engineering |
| `FoundationForecaster` | `model_name`, `batch_size` | Zero-shot; wraps HuggingFace Chronos-style interface |
| `IntermittentForecaster` | `method` (`croston`/`adida`), `alpha` | For CV > 0.49 or ADI > 1.32 series |
| `EnsembleForecaster` | `models`, `weights`, `quantiles` | Weighted mixture; bootstrapped quantiles |

### 4.3 Series Classification (`src/series/sparse_detector.py`)

Series are classified before model selection:

| Class | CV² | ADI | Recommended Model |
|-------|-----|-----|-------------------|
| Smooth | ≤ 0.49 | ≤ 1.32 | Statistical / ML |
| Intermittent | ≤ 0.49 | > 1.32 | Croston |
| Erratic | > 0.49 | ≤ 1.32 | ADIDA |
| Lumpy | > 0.49 | > 1.32 | ADIDA / Ensemble |

---

## 5. Hierarchical Reconciliation

### 5.1 Summing Matrix S

`S` has shape `(n_all × n_leaves)` where `S[i, j] = 1` if leaf `j` is in the subtree of node `i`.
Leaf rows of `S` form the identity matrix.

### 5.2 Linear Reconciliation Formula

```
G = (S′W⁻¹S)⁻¹ S′W⁻¹         # projection onto coherent subspace
P̃_leaf = G · P̂_all            # reconciled leaf forecasts
P̃_all  = S · P̃_leaf           # all-level coherent forecasts
```

Tikhonov regularisation `λ·I` (λ = 1e-6) is added to `(S′W⁻¹S)` for numerical stability.
Non-negativity: `clip(0)` applied to leaf forecasts after reconciliation.

### 5.3 W Matrix by Method

| Method | W |
|--------|---|
| OLS | Identity (equal uncertainty at all levels) |
| WLS-structural | `diag(n_leaf_descendants per node)` — nodes with more leaves get less weight |
| WLS-residual | `diag(per-series residual variance)` from supplied residuals DataFrame |
| MinT | Ledoit–Wolf diagonal shrinkage covariance; falls back to WLS-structural when `T < n` |

### 5.4 MinT Shrinkage

```
λ* = min(1, n / T)             # Ledoit-Wolf intensity, capped at 1
Σ_shrunk = (1 - λ*) · Σ_sample + λ* · diag(Σ_sample)
W = diag(Σ_shrunk)             # diagonal only for stability
```

where `n` = number of series, `T` = number of time observations in residuals.

---

## 6. Backtesting

### 6.1 Walk-Forward Protocol

- **Expanding window**: training set grows with each fold; test set is a fixed horizon.
- Configurable: `n_folds`, `horizon_weeks`, `min_train_weeks`, `step_weeks`.
- Per-fold, per-series metrics are written to `MetricStore`.

### 6.2 Champion Selection

`ChampionSelector` ranks all model candidates by mean WMAPE across folds and series. The lowest WMAPE model is promoted to champion. Tie-breaking uses normalized bias (closer to 0 wins).

---

## 7. Analytics Layer

### 7.1 ForecastComparator

Inputs:
- `model_forecast`: panel DataFrame with `[series_id, week, forecast, p10, p90]`
- `external_forecasts`: dict mapping source name → panel DataFrame
- `prior_model_forecast`: previous cycle's model forecast

Outputs (additional columns on model_forecast):
- `{name}_forecast`, `{name}_gap`, `{name}_gap_pct` — per external source
- `prior_model_forecast`, `cycle_change`, `cycle_change_pct`
- `uncertainty_ratio` = (p90 − p10) / p50

### 7.2 ExceptionEngine

Default thresholds (all configurable at construction):

```python
cycle_change_pct_threshold  = 20.0   # %
uncertainty_ratio_threshold = 0.50
field_disagree_pct_threshold= 25.0   # %
overforecast_pct_threshold  = 30.0   # %
underforecast_pct_threshold = -30.0  # %
```

`flag()` returns the input DataFrame with boolean `exc_*` columns and `has_exception`.
`exception_summary()` returns one row per series with flagged-week counts, sorted by `total_exception_weeks` descending.

### 7.3 ForecastExplainer

**STL decomposition** (classical additive):
1. Trend = centered moving average with window = `season_length` (default 52).
2. Seasonal = mean de-trended value per seasonal position across all years.
3. Residual = value − trend − seasonal.
4. Forecast trend = linear extrapolation from last `trend_window` (default 12) observations.
5. Forecast seasonal = historical seasonal pattern applied forward.

**SHAP** (optional, `pip install shap`):
- Uses `shap.Explainer` with the fitted model and feature matrix.
- Returns tidy DataFrame `[series_id, week, feature, shap_value, rank]`.
- Returns empty DataFrame with warning column if `shap` not installed.

**Narrative template** (per series):
```
"Series {sid}: forecast is {X}% above/below last year, primarily driven by {trend|seasonality}.
 System is {Y}% above/below {source} forecast on average.
 Model uncertainty is HIGH/moderate/low."
```

### 7.4 DriftDetector

```
ratio = live_wmape / backtest_wmape

ratio ≤ warn_multiplier (1.25)              → "ok"
warn_multiplier < ratio ≤ alert_multiplier  → "warning"
ratio > alert_multiplier (1.50)             → "alert"
n_live_weeks < min_live_weeks (4)           → "insufficient_data"
```

`batch_detect()` runs detection for all model IDs found in the MetricStore, returns a DataFrame sorted by ratio descending.

### 7.5 ModelCard

Captures everything needed to reproduce or audit a model:

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | str | Unique model identifier |
| `lob` | str | Line of business |
| `training_start` / `training_end` | date | Training window |
| `n_series` | int | Number of distinct series |
| `n_observations` | int | Total rows used in training |
| `backtest_wmape` | float | Mean WMAPE across all folds |
| `backtest_bias` | float | Mean normalized bias |
| `champion_since` | date | Date promoted to champion |
| `features` | list[str] | Feature names (ML models) |
| `config_hash` | str | MD5[:8] of serialized config |
| `notes` | str | Free-text notes |

Persisted to `data/model_cards/model_cards.parquet` via `ModelCardRegistry`.

### 7.6 ForecastLineage

Append-only log; one Parquet file per run at `data/lineage/lineage_{lob}_{date}_{model_id}.parquet`.

| Field | Description |
|-------|-------------|
| `run_date` | Date the forecast run was executed |
| `lob` | Line of business |
| `model_id` | Champion model used |
| `selection_strategy` | How champion was chosen |
| `n_series` | Number of series forecast |
| `horizon_weeks` | Forecast horizon |
| `run_id` | Optional external run identifier |
| `notes` | Free-text |

`history(lob, model_id)` returns all records filtered and sorted by `run_date` descending.
`latest(lob)` returns the most recent record for a LOB.

---

## 8. SKU Mapping

Handles new-product launches (no history) and discontinuations (multi-mapped successors).

### 8.1 Mapping Methods

| Method | Input | Similarity |
|--------|-------|-----------|
| `AttributeMatching` | Product attribute vectors | Cosine similarity |
| `NamingParsing` | SKU name / description strings | Token overlap (Jaccard) |
| `CurveFitting` | Sales trajectory of candidate | S-curve / step-ramp shape fit (R²) |
| `TemporalComovement` | Historical weekly sales | Pearson correlation of growth rates |

### 8.2 Bayesian Proportion Estimation

For 1-to-many splits (one old SKU → multiple new SKUs):

- Prior: Dirichlet(α = uniform or attribute-informed)
- Likelihood: observed sales in overlap period
- Posterior: Dirichlet; mean used as proportion weights
- Output: proportion DataFrame with credible intervals

### 8.3 Fusion

`MappingScorer` combines method scores with configurable weights → ranked candidate list.
Confidence threshold filters low-confidence mappings, which fall back to category average.

---

## 9. REST API

Built with FastAPI. All endpoints accept/return JSON.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/forecast` | POST | Generate forecast for one or more series |
| `/metrics` | GET | Retrieve stored metrics (backtest or live) |
| `/drift` | GET | Drift status per model / LOB |
| `/lineage` | GET | Forecast lineage audit log |
| `/exceptions` | GET | Latest exception flags for a LOB |
| `/model-cards` | GET | All registered model cards |

Request/response schemas defined in `src/api/schemas.py`.

---

## 10. Microsoft Fabric Deployment

`FabricDeployment` orchestrates:
1. Write reconciled forecasts to Delta Lake via `DeltaWriter`.
2. Write backtest metrics to the Lakehouse.
3. Register model card.
4. Record lineage entry.
5. Trigger downstream Power BI dataset refresh (optional).

Config via `FabricConfig`:
- `workspace_id`, `lakehouse_id`, `capacity_id`
- `forecast_table`, `metrics_table`, `lineage_table`

---

## 11. Configuration Schema (`src/config/schema.py`)

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

backtest:
  n_folds: 4
  step_weeks: 13

reconciliation:
  method: mint           # bottom_up | top_down | middle_out | ols | wls | mint
  non_negative: true

exception_thresholds:
  cycle_change_pct: 20
  uncertainty_ratio: 0.50
  field_disagree_pct: 25
  overforecast_pct: 30
  underforecast_pct: -30

drift:
  warn_multiplier: 1.25
  alert_multiplier: 1.50
  min_live_weeks: 4
```

---

## 12. Test Coverage

| Test File | Tests | Covers |
|-----------|-------|--------|
| `test_platform.py` | ~80 | Core platform integration |
| `test_mint_reconciliation.py` | 46 | OLS / WLS / MinT reconciliation, S-matrix math |
| `test_forecast_explainability.py` | 59 | Comparator, ExceptionEngine, Explainer, Governance |
| `test_probabilistic_ensemble.py` | ~60 | Ensemble quantiles, probabilistic output |
| `test_foundation_models.py` | ~40 | Foundation model interface |
| `test_intermittent_demand.py` | ~40 | Croston, ADIDA, sparse detection |
| `test_sku_mapping.py` | ~60 | All mapping methods + Bayesian fusion |
| `test_feature_engineering.py` | ~20 | Feature engineering (pandas-dependent) |
| `test_metrics.py` | ~20 | Metrics (pandas-dependent) |

**391 tests pass** (excluding 2 pandas-dependent files not available in this environment).

---

## 13. Dependencies

Core (no pandas required in production path):
```
polars
numpy
scipy
lightgbm
xgboost
fastapi
uvicorn
pyyaml
```

Optional:
```
shap           # SHAP explainability
pyspark        # Distributed execution
azure-storage-blob  # Fabric / Delta Lake
chronos-forecasting # Foundation models
```
