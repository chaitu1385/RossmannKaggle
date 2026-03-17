# Forecasting Platform

A production-grade, modular weekly sales forecasting platform. Covers the full lifecycle from raw data ingestion to hierarchically reconciled, explained, and governed forecasts — with a REST API, Microsoft Fabric/Delta Lake deployment layer, Spark distributed execution, and S&OP exception management.

---

## System Layers

```
┌─────────────────────────────────────────────────────────────┐
│  REST API  (FastAPI, JWT-protected)   src/api/              │
│  POST /auth/token                                           │
│  GET /health  /forecast/{lob}  /metrics/leaderboard/{lob}  │
│  GET /forecast/{lob}/{series_id}  /metrics/drift/{lob}     │
│  GET /audit                                                 │
├─────────────────────────────────────────────────────────────┤
│  Auth & Audit                         src/auth/ src/audit/  │
│  RBAC (5 roles, 11 permissions) · JWT tokens                │
│  AuditLogger (append-only, Parquet, date-partitioned)       │
├─────────────────────────────────────────────────────────────┤
│  Analytics Layer                      src/analytics/        │
│  ForecastAnalytics (notebook API)                           │
│  BIExporter (Power BI / Parquet)                            │
│  ForecastComparator · ExceptionEngine                       │
│  ForecastExplainer (STL + SHAP + narrative)                 │
│  DriftDetector · ModelCard · ModelCardRegistry              │
│  ForecastLineage · FVAAnalyzer                              │
├─────────────────────────────────────────────────────────────┤
│  Pipeline Layer                       src/pipeline/         │
│  BacktestPipeline · ForecastPipeline · PipelineManifest     │
├─────────────────────────────────────────────────────────────┤
│  Backtest Engine                      src/backtesting/      │
│  BacktestEngine · WalkForwardCV · ChampionSelector          │
├─────────────────────────────────────────────────────────────┤
│  Override Store                       src/overrides/        │
│  OverrideStore (DuckDB)                                     │
├─────────────────────────────────────────────────────────────┤
│  Hierarchy Layer                      src/hierarchy/        │
│  HierarchyTree · HierarchyNode · HierarchyAggregator       │
│  Reconciler (bottom_up · top_down · middle_out)             │
│            (ols · wls · mint)                               │
├─────────────────────────────────────────────────────────────┤
│  Model Library                        src/forecasting/      │
│  SeasonalNaive · AutoARIMA · AutoETS · AutoTheta · MSTL    │
│  LGBMDirect · XGBoostDirect                                 │
│  Chronos · TimeGPT (zero-shot foundation)                   │
│  N-BEATS · NHITS · TFT (neural)                             │
│  Croston · CrostonSBA · TSB (intermittent)                  │
│  WeightedEnsemble · HierarchicalForecaster                  │
│  ConstrainedDemandEstimator · ForecasterRegistry            │
├─────────────────────────────────────────────────────────────┤
│  Metrics                              src/metrics/          │
│  MetricStore · ForecastDriftDetector · FVA engine           │
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
│  DataLoader · DataPreprocessor · FeatureEngineer            │
│  ExternalRegressorLoader · HolidayCalendar                  │
│  DataValidator · DemandCleanser · RegressorScreen            │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure                       src/fabric/ src/spark/│
│  DeploymentOrchestrator · FabricLakehouse · DeltaWriter     │
│  SparkForecastPipeline · SparkSeriesBuilder                 │
│  SparkFeatureEngineer · SparkDataLoader                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
forecasting-platform/
├── src/
│   ├── analytics/          # Notebook API, BI export, comparators, exception engine, explainability, governance, FVA
│   ├── api/                # FastAPI REST serving layer (auth-protected)
│   ├── audit/              # Append-only Parquet audit log (immutable, date-partitioned)
│   ├── auth/               # RBAC, JWT authentication, role/permission models
│   ├── backtesting/        # Walk-forward backtest engine, champion selection
│   ├── config/             # YAML config schema + loader (incl. external regressor config)
│   ├── data/               # Data loading, preprocessing, validation, demand cleansing, regressor screening, external regressors
│   ├── evaluation/         # Metrics (WMAPE, RMSPE, bias, MAE) + evaluator
│   ├── fabric/             # Microsoft Fabric / Delta Lake deployment
│   ├── forecasting/        # Model implementations + registry (statistical, ML, neural, foundation, intermittent)
│   ├── hierarchy/          # Hierarchy tree, aggregation, reconciliation (OLS/WLS/MinT)
│   ├── metrics/            # MetricStore, drift detection, metric definitions, FVA computation
│   ├── models/             # LightGBM + XGBoost wrappers (legacy)
│   ├── overrides/          # Planner manual override store (DuckDB)
│   ├── pipeline/           # End-to-end backtest + forecast pipelines, provenance manifest
│   ├── series/             # Series builder, sparse detector, lifecycle transitions
│   ├── sku_mapping/        # New/discontinued SKU mapping (4 methods + Bayesian fusion)
│   ├── spark/              # PySpark distributed execution layer
│   └── utils/              # Logger, config utilities
├── tests/                  # 760+ unit + integration tests
├── configs/                # YAML configuration files
├── scripts/                # Entry points (run_backtest, run_forecast, serve, spark_*)
├── notebooks/              # Jupyter notebooks for exploration
├── data/rossmann/          # Rossmann dataset for demos
├── requirements.txt
└── setup.py
```

---

## Modules

### `src/forecasting/` — Model Library

All models implement a common interface (`BaseForecaster`):

```python
def fit(self, data: pl.DataFrame, config: ForecastConfig) -> "BaseForecaster"
def predict(self, horizon: int, ...) -> pl.DataFrame               # P50 point forecast
def predict_quantiles(self, horizon: int, quantiles: list[float]) -> pl.DataFrame
```

**Statistical & Naive:**

| Class | Description |
|-------|-------------|
| `SeasonalNaiveForecaster` | Last-year same-week seasonal naive baseline; `season_length=52` |
| `AutoARIMAForecaster` | Auto-ARIMA via statsforecast; `season_length=52` |
| `AutoETSForecaster` | ETS (Error/Trend/Season) via statsforecast; `season_length=52` |
| `AutoThetaForecaster` | AutoTheta via statsforecast; automatic trend/seasonality decomposition |
| `MSTLForecaster` | Multiple Seasonal-Trend decomposition via statsforecast; multi-period seasonality |

**ML (gradient boosting):**

| Class | Description |
|-------|-------------|
| `LGBMDirectForecaster` | LightGBM direct multi-step with lag/rolling features; supports external regressors |
| `XGBoostDirectForecaster` | XGBoost direct multi-step with lag/rolling features; supports external regressors |

**Neural (deep learning):**

| Class | Description |
|-------|-------------|
| `NBEATSForecaster` | N-BEATS neural forecaster via neuralforecast; basis expansion architecture |
| `NHITSForecaster` | N-HiTS neural forecaster via neuralforecast; hierarchical interpolation |
| `TFTForecaster` | Temporal Fusion Transformer via neuralforecast; attention-based with interpretable components |

**Foundation models (zero-shot):**

| Class | Description |
|-------|-------------|
| `ChronosForecaster` | Zero-shot foundation model (Amazon Chronos); no fine-tuning needed |
| `TimeGPTForecaster` | Zero-shot foundation model (Nixtla TimeGPT) via REST API |

**Intermittent demand:**

| Class | Description |
|-------|-------------|
| `CrostonForecaster` | Croston's method for intermittent demand |
| `CrostonSBAForecaster` | Croston-SBA (bias-corrected variant) |
| `TSBForecaster` | Teunter-Syntetos-Babai method for lumpy demand |

**Ensemble, hierarchical & constrained:**

| Class | Description |
|-------|-------------|
| `WeightedEnsembleForecaster` | Weighted mixture of any base models; bootstrapped P10/P50/P90 |
| `HierarchicalForecaster` | Forecaster that produces coherent hierarchical forecasts with built-in reconciliation |
| `ConstrainedDemandEstimator` | Wraps any base forecaster; enforces non-negativity, capacity limits, aggregate budgets; preserves quantile monotonicity |
| `ForecasterRegistry` | Register, retrieve, and instantiate models by name |

### `src/data/` — Data Layer

#### External Regressors

| Function | Description |
|----------|-------------|
| `load_external_features(path)` | Load external feature data from Parquet or CSV |
| `generate_holiday_calendar(country, start, end)` | Generate weekly holiday flags using the `holidays` library (optional dep) |
| `validate_regressors(features, actuals, columns)` | Validate grain alignment, null checks, future coverage for forecast horizon |

ML models (`LGBMDirectForecaster`, `XGBoostDirectForecaster`) automatically detect and use external feature columns during `fit()` and `predict()`. Statistical and naive models silently ignore them.

#### DataValidator (schema enforcement)

| Class/Dataclass | Description |
|-----------------|-------------|
| `DataValidator` | Validates input DataFrames: schema checks, duplicate detection, frequency validation, value range enforcement, completeness checks |
| `ValidationReport` | Result with `passed` flag, `issues` list, counts for duplicates/negatives/frequency violations |
| `ValidationIssue` | Single issue with `level` (error/warning), `check` name, `message`, optional `series_id` |

Runs as the first step in `SeriesBuilder.build()` when `validation.enabled = True`. Raises `ValueError` on errors in strict mode.

#### DemandCleanser (outlier & stockout correction)

| Class/Dataclass | Description |
|-----------------|-------------|
| `DemandCleanser` | Detects outliers (IQR/z-score), identifies stockout periods, applies corrections per-series |
| `CleansingResult` | Contains cleaned DataFrame + `CleansingReport` summary |
| `CleansingReport` | Counts: outliers, stockout periods/weeks, rows modified, per-series breakdown |

Runs after gap-filling in `SeriesBuilder.build()` when `cleansing.enabled = True`. All statistics computed per-series (not globally). Supports period exclusion (e.g., COVID) with interpolate/drop/flag actions.

#### RegressorScreen (pre-training feature quality)

| Class/Dataclass | Description |
|-----------------|-------------|
| `screen_regressors()` | Screens external features before model training: zero/near-zero variance, high pairwise correlation, optional mutual information with target |
| `RegressorScreenReport` | Result with `screened_columns`, `dropped_columns`, `low_variance_columns`, `high_correlation_pairs`, `low_mi_columns`, per-column stats |

Runs after feature join in `SeriesBuilder.build()` when `external_regressors.screen.enabled = True`. Optionally auto-drops flagged columns (`auto_drop=True`).

### `src/auth/` — RBAC & Authentication

| Class/Function | Description |
|----------------|-------------|
| `Role` (enum) | `ADMIN`, `DATA_SCIENTIST`, `PLANNER`, `MANAGER`, `VIEWER` |
| `Permission` (enum) | 11 permissions: `VIEW_FORECASTS`, `VIEW_METRICS`, `VIEW_AUDIT_LOG`, `CREATE_OVERRIDE`, `DELETE_OVERRIDE`, `APPROVE_OVERRIDE`, `RUN_BACKTEST`, `RUN_PIPELINE`, `PROMOTE_MODEL`, `MODIFY_CONFIG`, `MANAGE_USERS` |
| `ROLE_PERMISSIONS` | Complete role -> permission mapping |
| `User` | Dataclass with `user_id`, `email`, `role`, `is_active`; `has_permission()` method |
| `get_current_user()` | FastAPI dependency — extracts/validates JWT from `Authorization: Bearer` header |
| `require_permission(perm)` | FastAPI dependency factory for fine-grained permission checks |
| `require_role(*roles)` | FastAPI dependency factory for role-based checks |
| `create_token(user_id, email, role)` | Create signed JWT (HS256, 24h default expiry) |
| `decode_token(token, secret)` | Decode and validate JWT; returns `None` on failure |

**Permission matrix:**

| Action | ADMIN | DATA_SCIENTIST | PLANNER | MANAGER | VIEWER |
|--------|-------|----------------|---------|---------|--------|
| View forecasts/metrics | Y | Y | Y | Y | Y |
| Create overrides | Y | Y | Y | N | N |
| Approve overrides | Y | N | N | Y | N |
| Run backtest/pipeline | Y | Y | N | N | N |
| Promote champion model | Y | Y | N | N | N |
| View audit log | Y | Y | N | Y | N |
| Manage users | Y | N | N | N | N |

### `src/audit/` — Audit Trail

| Class | Description |
|-------|-------------|
| `AuditEvent` | Immutable dataclass: `action`, `resource_type`, `resource_id`, `user_id`, `user_role`, `status`, `old_value`, `new_value`, `ip_address`, `request_id`, auto-generated `timestamp` (UTC) and `audit_id` |
| `AuditLogger` | Append-only Parquet-backed logger; date-partitioned (`audit_log/date=YYYY-MM-DD/`) |

```python
logger = AuditLogger("data/audit_log/")
logger.log(event)                       # single event
logger.log_batch(events)                # batch write
logger.query(user_id=..., action=..., start_date=..., limit=100)  # filtered reads
logger.count_by_action()                # aggregation by action + status
```

No UPDATE or DELETE operations — append-only by design for SOX compliance.

### `src/series/` — Series Management

| Class | Description |
|-------|-------------|
| `SeriesBuilder` | Builds weekly panel DataFrames from raw transactional data; fills gaps; integrates validation and cleansing |
| `SparseDetector` | Classifies series as smooth / intermittent / erratic / lumpy using CV-squared and ADI |
| `TransitionEngine` | Handles new-product launches: stitches history, applies linear/S-curve/step ramps |

**Sparse classification thresholds:**

| Class | CV-squared | ADI | Recommended model |
|-------|------------|-----|-------------------|
| Smooth | <= 0.49 | <= 1.32 | Statistical / ML |
| Intermittent | <= 0.49 | > 1.32 | Croston / CrostonSBA |
| Erratic | > 0.49 | <= 1.32 | TSB |
| Lumpy | > 0.49 | > 1.32 | TSB / Ensemble |

**Transition scenarios:**

| Scenario | Condition | Action |
|----------|-----------|--------|
| A — Already launched | `launch_date <= forecast_origin` | Stitch old SKU history onto new SKU |
| B — In horizon | `0 < gap <= transition_window` | Ramp-down old, ramp-up new (linear/scurve/step) |
| C — Beyond horizon | `gap > transition_window` | Forecast old only; new SKU flagged "pending" |

### `src/hierarchy/` — Hierarchical Reconciliation

| Class | Description |
|-------|-------------|
| `HierarchyTree` | Build and traverse a multi-level node hierarchy; exposes `summing_matrix()` |
| `HierarchyNode` | Single node with `descendants()`, `ancestors()`, `leaf_descendants()`, `is_leaf()` |
| `HierarchyAggregator` | `aggregate_to()`, `disaggregate_to()`, `compute_historical_proportions()` |
| `Reconciler` | Six reconciliation methods; non-negativity enforced |

**Reconciliation methods:**

| Method | W matrix | Notes |
|--------|----------|-------|
| `bottom_up` | — | Leaf forecasts are authoritative; upper levels are sums |
| `top_down` | — | Top level disaggregated by historical proportions |
| `middle_out` | — | Mid-level is authoritative; disaggregated down, aggregated up |
| `ols` | Identity | Equal uncertainty at all levels |
| `wls` | `diag(n_leaf_descendants)` or per-series residual variance | Structural or data-driven weights |
| `mint` | Ledoit-Wolf diagonal shrinkage covariance | Falls back to WLS-structural when `T < n` |

**Linear reconciliation formula (OLS/WLS/MinT):**
```
G       = (S'W-1 S)-1 S'W-1       # projection matrix
P_leaf  = G * P_all                # reconciled leaf forecasts
P_all   = S * P_leaf               # all-level coherent forecasts
```
Tikhonov regularisation (lambda = 1e-6) applied for numerical stability.

### `src/backtesting/` — Backtest Engine

| Class | Description |
|-------|-------------|
| `BacktestEngine` | Orchestrates walk-forward folds; writes per-fold, per-series metrics to MetricStore |
| `WalkForwardCV` | Expanding-window fold generator; configurable `n_folds`, `horizon_weeks`, `step_weeks` |
| `CVFold` | Dataclass for one fold: train indices, test indices, fold number |
| `ChampionSelector` | Ranks models by WMAPE; promotes lowest-error model; `compute_ensemble_weights()` |

### `src/metrics/` — Metric Store & Drift

| Class/Function | Description |
|----------------|-------------|
| `MetricStore` | Append-only Parquet store; `write()`, `read()`, `leaderboard()`, `accuracy_over_time()` |
| `ForecastDriftDetector` | Detects accuracy drift, bias drift, and volume anomalies per series |
| `DriftAlert` | `series_id`, `metric`, `severity` (warning/critical), `current_value`, `baseline_value` |
| `DriftConfig` | `baseline_weeks`, `recent_weeks` configuration |
| `wmape()` / `normalized_bias()` / `mape()` / `mae()` / `rmse()` | Standalone metric functions |
| `compute_all_metrics()` | Compute all metrics in one pass |

### `src/analytics/` — Explainability, Governance & BI

#### `ForecastAnalytics` (notebook API)
Notebook-ready queries over the MetricStore:
- `model_leaderboard(lob, run_type, primary_metric)` — rank models by WMAPE
- `model_comparison_by_fold(lob)` — per-fold performance breakdown
- `accuracy_over_time(model, channel)` — trend of accuracy metrics
- `accuracy_by_grain(lob)` — performance by hierarchy level
- `bias_distribution(lob)` — normalized bias distribution
- `transition_impact(lob)` — before/after metrics for SKU transitions
- `backtest_vs_live(lob)` — backtest vs live accuracy comparison

#### `BIExporter`
Writes Parquet in Hive-partitioned layout for direct Power BI consumption:
- `export_forecast_vs_actual(forecasts, actuals, lob)` -> `bi_exports/forecast_vs_actual/lob=.../`
- `export_leaderboard(lob)` -> `bi_exports/model_leaderboard/lob=.../`
- `export_bias_report(lob)` -> `bi_exports/bias_report/lob=.../`
- `export_fva(fva_detail, fva_summary, lob)` -> `bi_exports/fva_detail/lob=.../` and `fva_summary/lob=.../`

#### `ForecastComparator`
Aligns model forecast with external sources and prior cycle:
- `compare(model_forecast, external_forecasts, prior_model_forecast)` — adds `{name}_gap`, `{name}_gap_pct`, `uncertainty_ratio`, `cycle_change`, `cycle_change_pct`
- `summary(comparison)` — one row per series_id

#### `ExceptionEngine`
Business-rule exception flags for S&OP review queues:

| Flag | Default threshold |
|------|------------------|
| `exc_large_cycle_change` | \|cycle_change_pct\| > 20% |
| `exc_high_uncertainty` | uncertainty_ratio > 0.50 |
| `exc_field_disagree` | \|gap_pct\| > 25% vs any external source |
| `exc_overforecast` | gap_pct > 30% |
| `exc_underforecast` | gap_pct < -30% |
| `exc_no_prior` | prior_model_forecast is null |

All thresholds configurable at construction. `exception_summary()` groups by series with flagged-week counts, sorted by `total_exception_weeks` descending.

#### `ForecastExplainer`
- **`decompose(history, forecast)`** — STL-style: trend (centered MA, window=`season_length`) + seasonal (mean de-trended per position) + residual; trend extrapolated linearly over `trend_window` periods into forecast horizon
- **`explain_ml(model, features_df, top_k=5)`** — SHAP attribution for LightGBM/XGBoost; lazy import, graceful empty-DataFrame fallback if `shap` not installed
- **`narrative(decomposition, comparison)`** — templated natural-language string per series (YoY direction, primary driver, gap vs external, uncertainty label)

#### Model Governance (`governance.py`)

| Class | Description |
|-------|-------------|
| `DriftDetector` | Compares live WMAPE to backtest WMAPE; returns `ok` / `warning` / `alert` / `insufficient_data` |
| `ModelCard` | Structured metadata dataclass: training window, series count, backtest metrics, features, config hash |
| `ModelCardRegistry` | In-memory + Parquet-backed registry; `register()`, `get()`, `all_cards()` |
| `ForecastLineage` | Append-only audit log of which model produced each forecast run; `record()`, `history()`, `latest()` |

#### FVA Analysis (`metrics/fva.py` + `analytics/fva_analyzer.py`)

Measures how much accuracy each forecast layer contributes — from naive baseline through statistical, ML, and planner overrides.

**Layers:**

| Layer | Models | Role |
|-------|--------|------|
| L1: Naive | `seasonal_naive` | Baseline (always computed) |
| L2: Statistical | `auto_arima`, `auto_ets`, `croston`, `croston_sba`, `tsb` | Best statistical per series |
| L3: ML | `lgbm_direct`, `xgboost_direct` | Best ML per series |
| L4: Override | Planner-adjusted | If override exists, else L4 = L3 |

**Key functions (`src/metrics/fva.py`):**

| Function | Description |
|----------|-------------|
| `classify_fva(value)` | `ADDS_VALUE` (>2pp), `NEUTRAL`, or `DESTROYS_VALUE` (<-2pp) |
| `compute_layer_metrics(actual, forecast)` | WMAPE, bias, MAE for one layer |
| `compute_fva_between_layers(actual, parent, child)` | Incremental FVA with classification |
| `compute_fva_cascade(actual, forecasts)` | Full cascade metrics across all layers |
| `compute_total_fva(actual, forecasts)` | Total WMAPE reduction baseline -> final |

**FVA Analyzer (`src/analytics/fva_analyzer.py`):**

| Method | Description |
|--------|-------------|
| `compute_fva_detail(backtest_results)` | Per-series, per-fold FVA from backtest results |
| `summarize(fva_detail, group_by)` | Aggregate by layer: mean WMAPE, FVA, % adds/neutral/destroys |
| `layer_leaderboard(fva_detail)` | Rank layers by contribution; recommends Keep/Review/Remove |

### `src/overrides/` — Planner Overrides

| Class | Description |
|-------|-------------|
| `OverrideStore` | DuckDB-backed store for planner **transition overrides** (old_sku -> new_sku proportion, scenario, ramp shape); `add_override()`, `get_overrides()`, `delete_override()` |

### `src/sku_mapping/` — New / Discontinued SKU Mapping

Maps new SKUs to analogues and splits multi-mapped forecasts proportionally.

**Mapping methods:**

| Class | Similarity | Input |
|-------|-----------|-------|
| `AttributeMatchingMethod` | Cosine similarity | Product attribute vectors |
| `NamingConventionMethod` | Token overlap (Jaccard) + base-name parsing | SKU names/descriptions |
| `CurveFittingMethod` | S-curve / step-ramp shape fit (R-squared) | Sales trajectory |
| `TemporalCovementMethod` | Pearson correlation of growth rates | Historical weekly sales |

**Fusion:**

| Class | Description |
|-------|-------------|
| `CandidateFusion` | Combines method scores with configurable weights; confidence classification |
| `BayesianProportionEstimator` | Dirichlet posterior for 1-to-many splits; `concentration=0.5` |

### `src/api/` — REST API

Built with FastAPI. Auto-generated Swagger docs at `/docs`. All data endpoints require JWT authentication when `auth_enabled=True`.

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/health` | GET | Liveness probe | No |
| `/auth/token` | POST | Obtain JWT access token (`username`, `role` params) | No |
| `/forecast/{lob}` | GET | Latest forecasts for a LOB (optional `series_id`, `horizon` query params) | Yes |
| `/forecast/{lob}/{series_id}` | GET | Latest forecast for a single series | Yes |
| `/metrics/leaderboard/{lob}` | GET | Model leaderboard ranked by WMAPE | Yes |
| `/metrics/drift/{lob}` | GET | Drift alerts (`baseline_weeks`, `recent_weeks` params) | Yes |
| `/audit` | GET | Query audit log (`action`, `resource_type`, `limit` params) | Yes (`VIEW_AUDIT_LOG`) |

### `src/fabric/` — Microsoft Fabric / Delta Lake

| Class | Description |
|-------|-------------|
| `DeploymentOrchestrator` | End-to-end pipeline: preflight checks -> champion selection -> forecast -> write -> audit log |
| `FabricLakehouse` | `read_table()`, `write_table()`, `table_exists()`, `vacuum()`, `optimize()`, `history()` |
| `DeltaWriter` | `upsert()`, `overwrite_partition()`, `append()`, `write_forecasts()` |
| `FabricConfig` | Workspace/lakehouse/capacity config; `from_env()`, `from_dict()`, `abfss_base` property |

### `src/spark/` — Distributed Execution

PySpark wrappers for large-scale runs on Fabric/Databricks:

| Class/Function | Description |
|----------------|-------------|
| `SparkForecastPipeline` | `run_forecast()`, `run_backtest()`, `select_champion()` |
| `SparkSeriesBuilder` | Distributed panel construction from raw Spark DataFrames |
| `SparkFeatureEngineer` | `create_temporal_features()`, `create_lag_features()`, `create_rolling_features()` |
| `SparkDataLoader` | Reads actuals, product master, Delta tables, Rossmann data |
| `get_or_create_spark()` | Fabric-aware session factory (detects `notebookutils`) |
| `polars_to_spark()` / `spark_to_polars()` | Schema-preserving conversion utilities |
| `repartition_by_series()` | Optimise parallelism by series count |
| `abfss_uri()` | Build `abfss://` URIs for Fabric/ADLS |

### `src/pipeline/` — End-to-End Pipelines

| Class | Description |
|-------|-------------|
| `BacktestPipeline` | Wires `SeriesBuilder -> BacktestEngine -> MetricStore -> ChampionSelector` |
| `ForecastPipeline` | Wires `SeriesBuilder -> champion model -> Reconciler -> OverrideStore -> BIExporter` |
| `PipelineManifest` | Provenance sidecar for each forecast run: input data hash, cleansing/validation/screen summaries, config hash, champion model, output metadata |
| `build_manifest()` | Collects provenance from SeriesBuilder reports into a single manifest |
| `write_manifest()` | Writes manifest as JSON sidecar next to the forecast Parquet file |
| `read_manifest()` | Loads manifest from JSON file |

---

## Quick Start

```python
from src.forecasting import LGBMDirectForecaster, WeightedEnsembleForecaster, ForecasterRegistry
from src.series import SparseDetector, TransitionEngine
from src.hierarchy import HierarchyTree, HierarchyAggregator, Reconciler
from src.backtesting import BacktestEngine, WalkForwardCV, ChampionSelector
from src.metrics import MetricStore, ForecastDriftDetector
from src.metrics.fva import compute_fva_cascade, classify_fva
from src.analytics import (
    ForecastAnalytics, ForecastComparator, ExceptionEngine,
    ForecastExplainer, DriftDetector, ModelCard, ModelCardRegistry, ForecastLineage,
    BIExporter,
)
from src.analytics.fva_analyzer import FVAAnalyzer
from src.data.regressors import load_external_features, validate_regressors
from src.data.validator import DataValidator
from src.data.cleanser import DemandCleanser
from src.data.regressor_screen import screen_regressors
from src.forecasting.constrained import ConstrainedDemandEstimator
from src.auth.models import User, Role, Permission
from src.auth.token import create_token, decode_token
from src.audit.logger import AuditLogger
from src.overrides import OverrideStore
from src.pipeline import BacktestPipeline, ForecastPipeline
from src.pipeline.manifest import PipelineManifest, read_manifest

# 1. Load and validate external features (promotions, holidays, price)
ext_features = load_external_features("data/external_features.parquet")
warnings = validate_regressors(ext_features, panel_df, config.forecast.external_regressors.feature_columns)

# 2. Classify series and route to appropriate model
detector = SparseDetector()
classification = detector.classify(panel_df)
smooth, intermittent = detector.split(panel_df)

# 3. Backtest all candidate models (with external features)
engine = BacktestEngine(n_folds=4, horizon_weeks=13)
results = engine.run(models=[LGBMDirectForecaster(), SeasonalNaiveForecaster()], data=smooth)

# 4. Select champion
champion = ChampionSelector(config).select(results)

# 5. FVA Analysis — measure value added by each forecast layer
fva_analyzer = FVAAnalyzer()
fva_detail = fva_analyzer.compute_fva_detail(results)
fva_summary = fva_analyzer.summarize(fva_detail)
fva_leaderboard = fva_analyzer.layer_leaderboard(fva_detail)

# 6. Reconcile across hierarchy
tree = HierarchyTree(config.get_hierarchy("product"), hierarchy_df)
reconciler = Reconciler(tree)
reconciled = reconciler.reconcile(forecasts_df, method="mint", residuals=residuals_df)

# 7. Apply planner overrides
store = OverrideStore()
overrides = store.get_overrides(lob="retail", week=current_week)

# 8. Compare vs external forecasts and flag exceptions
comparison = ForecastComparator().compare(
    reconciled,
    external_forecasts={"field": field_df, "financial": finance_df},
    prior_model_forecast=last_cycle_df,
)
flagged = ExceptionEngine().flag(comparison)
actionable = flagged.filter(pl.col("has_exception"))

# 9. Explain
explainer = ForecastExplainer(season_length=52, trend_window=12)
decomp = explainer.decompose(history_df, reconciled)
narratives = explainer.narrative(decomp, comparison)

# 10. Governance
registry = ModelCardRegistry()
registry.register(ModelCard.from_backtest("lgbm_direct", "retail", results))
ForecastLineage().record(lob="retail", model_id="lgbm_direct", n_series=500, horizon_weeks=13)

# 11. Export to Power BI (including FVA tables)
exporter = BIExporter()
exporter.export_forecast_vs_actual(reconciled, actuals_df, lob="retail")
exporter.export_fva(fva_detail, fva_summary, lob="retail")

# 12. Auth — create JWT token and validate
token = create_token(user_id="analyst_1", email="a@co.com", role="data_scientist", secret_key="secret")
user = User(user_id="analyst_1", email="a@co.com", role=Role.DATA_SCIENTIST)
assert user.has_permission(Permission.RUN_BACKTEST)

# 13. Audit — log an action
audit = AuditLogger("data/audit_log/")
from src.audit.schemas import AuditEvent
audit.log(AuditEvent(action="promote_model", resource_type="model_card",
                     resource_id="lgbm_direct", user_id="analyst_1",
                     user_role="data_scientist", user_email="a@co.com", status="SUCCESS"))
```

---

## Configuration

```yaml
lob: retail

forecast:
  horizon_weeks: 39          # default; override per LOB
  forecasters: [lgbm_direct, auto_ets, seasonal_naive]
  quantiles: [0.1, 0.5, 0.9]
  sparse_detection: true
  intermittent_forecasters: [croston_sba, tsb]
  external_regressors:
    enabled: true
    feature_columns:
      - promotion_flag
      - holiday_flag
      - price_index
    future_features_path: data/future_features.parquet
    screen:                    # RegressorScreen
      enabled: false
      variance_threshold: 1.0e-6
      correlation_threshold: 0.95
      mi_enabled: false
      mi_threshold: 0.01
      auto_drop: true
  constraints:                # ConstrainedDemandEstimator
    enabled: false
    min_demand: 0.0
    max_capacity: null
    aggregate_max: null

backtest:
  n_folds: 3
  val_weeks: 13
  gap_weeks: 0
  champion_granularity: lob  # lob | product_group | series
  selection_strategy: champion   # champion | weighted_ensemble

hierarchies:
  - name: product
    levels: [total, category, sku]
    reconciliation:
      method: mint    # bottom_up | top_down | middle_out | ols | wls | mint

transition:
  transition_window_weeks: 13
  ramp_shape: linear   # linear | scurve | step
  enable_overrides: true

data_quality:
  validation:              # DataValidator
    enabled: false
    check_duplicates: true
    check_frequency: true
    check_non_negative: true
    max_missing_pct: 100.0
    strict: false
  cleansing:               # DemandCleanser
    enabled: false
    outlier_method: iqr    # iqr | zscore
    iqr_multiplier: 1.5
    outlier_action: clip   # clip | interpolate | flag_only
    stockout_detection: true
    min_zero_run: 2
```

---

## Testing

```bash
pip install -r forecasting-platform/requirements.txt
python -m pytest forecasting-platform/tests/ \
  --ignore=forecasting-platform/tests/test_metrics.py \
  --ignore=forecasting-platform/tests/test_feature_engineering.py -v
# 760+ tests collected
```

| Test file | Tests | Covers |
|-----------|------:|--------|
| `test_platform.py` | 85 | Config, hierarchy, reconciliation, metrics, transitions, registry, backtest, REST API, deployment, drift |
| `test_forecast_explainability.py` | 59 | Comparator, ExceptionEngine, Explainer (STL+SHAP+narrative), ModelCard, Registry, DriftDetector, Lineage |
| `test_intermittent_demand.py` | 55 | SparseDetector, Croston, CrostonSBA, TSB, backtest routing |
| `test_mint_reconciliation.py` | 46 | S-matrix math, OLS/WLS/MinT correctness, coherence, edge cases |
| `test_foundation_models.py` | 41 | Chronos, TimeGPT fit/predict/quantiles, error handling, zero-shot property |
| `test_nixtla_models.py` | 29 | AutoTheta, MSTL extended statsforecast models |
| `test_data_analyzer.py` | 29 | Data analysis module |
| `test_forecastability.py` | 28 | Forecastability signals and scoring |
| `test_causal_analyzer.py` | 27 | Causal/econometric analysis |
| `test_probabilistic_ensemble.py` | 24 | Ensemble quantiles, weight computation, config |
| `test_demand_cleansing.py` | 24 | Outlier detection, stockout imputation, period exclusion, CleansingReport |
| `test_data_validator.py` | 24 | Schema checks, duplicates, frequency, value range, completeness, builder integration |
| `test_calibration.py` | 20 | Interval calibration and coverage |
| `test_llm_analyzer.py` | 20 | LLM-based analysis integration |
| `test_quality_report.py` | 19 | Data quality reporting |
| `test_hierarchical_forecaster.py` | 17 | Hierarchical forecaster with reconciliation |
| `test_constrained_demand.py` | 16 | Non-negativity, capacity limits, aggregate budgets, quantile monotonicity |
| `test_rbac.py` | 14 | Role permissions, User model, AuditEvent, AuditLogger log/query/filters |
| `test_override_store.py` | 14 | Override CRUD, approval workflow |
| `test_break_detection.py` | 14 | Structural break detection |
| `test_mase.py` | 13 | MASE metric computation |
| `test_statistical_forecasters.py` | 12 | AutoARIMA, AutoETS fit/predict/quantiles |
| `test_fva.py` | 12 | FVA classification, layer metrics, cascade computation, FVAAnalyzer |
| `test_data_preprocessor.py` | 12 | Data preprocessing pipeline |
| `test_ml_forecasters.py` | 11 | LightGBM, XGBoost direct forecasters |
| `test_hierarchy_aggregator.py` | 11 | Aggregation, disaggregation, proportions |
| `test_data_integrity.py` | 9 | Data integrity checks |
| `test_multi_horizon.py` | 8 | Multi-horizon forecast evaluation |
| `test_regressor_screen.py` | 16 | Variance, correlation, MI screening; auto-drop; SeriesBuilder integration |
| `test_pipeline_manifest.py` | 15 | Manifest build, write, read roundtrip; hash determinism; ForecastPipeline integration |
| `test_external_regressors.py` | 6 | Regressor validation, SeriesBuilder with/without features |
| `test_data_loader.py` | 6 | Data loading from files |
| `test_evaluator.py` | 5 | Evaluation orchestration |

---

## Dependencies

**Core** (no pandas in the production path):
```
polars >= 0.20.0
numpy >= 1.21.0
statsforecast >= 1.6.0      # AutoARIMA, AutoETS, AutoTheta, MSTL
mlforecast >= 0.12.0        # ML direct forecasting
lightgbm >= 3.3.0
xgboost >= 1.5.0
scikit-learn >= 1.0.0
rapidfuzz >= 3.0.0          # SKU name matching
duckdb >= 0.9.0             # Override store
pyyaml >= 5.4.0
fastapi + uvicorn           # REST API
```

**Optional:**
```
neuralforecast              # N-BEATS, NHITS, TFT neural forecasters
shap                        # SHAP explainability for ML models
pyjwt >= 2.0.0              # JWT token creation/validation (RBAC)
bcrypt >= 4.0.0             # Password hashing (auth)
holidays >= 0.40            # Holiday calendar generation (external regressors)
pyspark >= 3.4.0            # Distributed execution
delta-spark >= 2.4.0        # Delta Lake (local dev)
azure-identity              # Fabric authentication
pandas >= 1.3.0             # Legacy data layer + evaluation
matplotlib / seaborn        # Visualisation
```
