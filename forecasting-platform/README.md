# Forecasting Platform

A production-grade, modular weekly sales forecasting platform built on Polars and NumPy. Covers the full lifecycle from raw data to reconciled, explained, and governed forecasts — with a REST API, Microsoft Fabric/Spark deployment layer, and S&OP exception management.

---

## Architecture Overview

```
forecasting-platform/
├── src/
│   ├── analytics/          # Explainability, comparators, exception engine, governance
│   ├── api/                # FastAPI REST serving layer
│   ├── backtesting/        # Walk-forward backtest engine + champion selection
│   ├── config/             # YAML config schema + loader
│   ├── data/               # Data loading, preprocessing, feature engineering
│   ├── evaluation/         # Metrics (WMAPE, RMSPE, bias) + evaluator
│   ├── fabric/             # Microsoft Fabric / Delta Lake deployment
│   ├── forecasting/        # Model implementations (statistical, ML, foundation, ensemble)
│   ├── hierarchy/          # Hierarchy tree, aggregation, reconciliation (OLS/WLS/MinT)
│   ├── metrics/            # MetricStore, drift metrics, metric definitions
│   ├── models/             # LightGBM + XGBoost wrappers
│   ├── overrides/          # Planner manual override store
│   ├── pipeline/           # End-to-end backtest + forecast pipelines
│   ├── series/             # Series builder, sparse detector, lifecycle transitions
│   ├── sku_mapping/        # New/discontinued SKU mapping (4 methods + Bayesian fusion)
│   ├── spark/              # PySpark distributed execution layer
│   └── utils/              # Logger, config utilities
├── tests/                  # 391 unit + integration tests
├── requirements.txt
└── setup.py
```

---

## Modules

### `src/forecasting/` — Model Library

| Class | Description |
|-------|-------------|
| `NaiveForecaster` | Seasonal naïve baseline (last-year same-week) |
| `StatisticalForecaster` | Exponential smoothing / ETS |
| `MLForecaster` | LightGBM / XGBoost direct multi-step |
| `FoundationForecaster` | Zero-shot foundation model wrapper (Chronos-style) |
| `IntermittentForecaster` | Croston / ADIDA for sparse/lumpy demand |
| `EnsembleForecaster` | Weighted ensemble with probabilistic (P10/P50/P90) output |
| `ModelRegistry` | Register, retrieve, and version trained models |

### `src/hierarchy/` — Hierarchical Reconciliation

Ensures forecasts are coherent across all levels of a product/geography hierarchy.

| Class | Description |
|-------|-------------|
| `HierarchyTree` | Build and traverse a multi-level hierarchy |
| `HierarchyAggregator` | Bottom-up aggregation of leaf forecasts |
| `Reconciler` | OLS, WLS, MinT reconciliation |

**Reconciliation methods:**
- **Bottom-Up** — leaf forecasts are authoritative; upper levels are sums.
- **Top-Down** — top level disaggregated by historical proportions.
- **Middle-Out** — a mid-level is authoritative; disaggregated down, aggregated up.
- **OLS** — optimal projection with identity error covariance (`W = I`).
- **WLS** — weighted least squares using structural weights (1/n_leaves) or per-series residual variance.
- **MinT** — Minimum Trace (Wickramasuriya et al., 2019); Ledoit–Wolf diagonal shrinkage covariance when residuals are supplied; falls back to WLS-structural otherwise.

All linear methods share:
```
G = (S′W⁻¹S)⁻¹ S′W⁻¹        # projection matrix
P̃_leaf = G · P̂_all            # reconciled leaf forecasts
```
Non-negativity is enforced with `clip(0)` after reconciliation.

### `src/backtesting/` — Backtest Engine

Walk-forward expanding-window cross-validation with per-fold, per-series metric tracking.

| Class | Description |
|-------|-------------|
| `BacktestEngine` | Orchestrates folds, calls model train/predict, writes metrics |
| `CrossValidator` | Fold generator (expanding or sliding window) |
| `ChampionSelector` | Compares models on backtest WMAPE; promotes best to champion |

### `src/analytics/` — Explainability & Governance

#### `ForecastComparator`
Aligns the system forecast with external sources (field, financial, prior cycle) and computes:
- `{name}_gap`, `{name}_gap_pct` vs each external source
- `uncertainty_ratio` = (P90 − P10) / P50
- `cycle_change`, `cycle_change_pct` vs prior model forecast
- `summary()` aggregates to one row per series

#### `ExceptionEngine`
Business-rule exception flags for S&OP review queues:

| Flag | Trigger |
|------|---------|
| `exc_large_cycle_change` | \|cycle_change_pct\| > 20% |
| `exc_high_uncertainty` | uncertainty_ratio > 0.50 |
| `exc_field_disagree` | \|gap_pct\| > 25% vs any external source |
| `exc_overforecast` | gap_pct > 30% |
| `exc_underforecast` | gap_pct < −30% |
| `exc_no_prior` | prior_model_forecast is null |

All thresholds are configurable. `exception_summary()` groups by series with flagged-week counts.

#### `ForecastExplainer`
- **STL-style decomposition**: trend (centered moving average) + seasonal (average de-trended per position) + residual — applied to both history and forecast horizon.
- **SHAP attribution**: per-prediction top-K feature importance for LightGBM/XGBoost. Lazy import — gracefully returns an empty DataFrame if `shap` is not installed.
- **Narrative generator**: templated natural-language strings per series (YoY direction, primary driver, gap vs external, uncertainty label).

#### Model Governance (`governance.py`)

| Class | Description |
|-------|-------------|
| `DriftDetector` | Compares live WMAPE to backtest WMAPE; returns ok / warning / alert |
| `ModelCard` | Structured metadata dataclass (training window, series count, backtest metrics, features, config hash) |
| `ModelCardRegistry` | In-memory registry backed by Parquet; `register()`, `get()`, `all_cards()` |
| `ForecastLineage` | Append-only audit log of which model produced each forecast run |

### `src/sku_mapping/` — New/Discontinued SKU Mapping

Maps new SKUs to analogues and splits multi-mapped forecasts proportionally.

| Method | Description |
|--------|-------------|
| `AttributeMatching` | Cosine similarity on product attributes |
| `NamingParsing` | Token overlap on SKU names / descriptions |
| `CurveFitting` | S-curve / step-ramp lifecycle shape fitting |
| `TemporalComovement` | Correlation of historical sales trajectories |
| `BayesianProportions` | Posterior proportion estimation for 1-to-many splits |
| `MappingScorer` | Fuses method scores with configurable weights |

### `src/api/` — REST Serving Layer

FastAPI application exposing:
- `POST /forecast` — run a forecast for a series or LOB
- `GET /metrics` — retrieve stored backtest / live metrics
- `GET /drift` — drift status per model
- `GET /lineage` — forecast lineage audit log

### `src/fabric/` — Microsoft Fabric / Delta Lake

| Module | Description |
|--------|-------------|
| `FabricDeployment` | End-to-end deployment pipeline to Fabric workspace |
| `LakehouseConnector` | Read/write Fabric Lakehouse tables |
| `DeltaWriter` | Write Polars DataFrames as Delta Lake tables |
| `FabricConfig` | Workspace, lakehouse, and capacity configuration |

### `src/spark/` — Distributed Execution

PySpark wrappers that mirror the single-node Polars layer for large-scale runs:
- `SparkLoader`, `SparkSeriesBuilder`, `SparkFeatureEngineering`
- `SparkForecastPipeline` — distributed model training and scoring

### `src/metrics/` — Metric Store

| Class | Description |
|-------|-------------|
| `MetricStore` | Append-only Parquet store for backtest and live metrics |
| `DriftMetrics` | Compute live drift metrics from actuals |
| `MetricDefinitions` | WMAPE, normalized bias, RMSPE, coverage definitions |

---

## Key Design Principles

- **Polars-first**: all DataFrames are `polars.DataFrame`; no pandas dependency in core modules.
- **Modular**: every layer is independently testable and replaceable.
- **Coherent forecasts**: hierarchical reconciliation guarantees leaf × S = all-level totals.
- **Graceful degradation**: SHAP, foundation models, and Spark are optional; the platform runs without them.
- **Audit-ready**: `ModelCardRegistry` + `ForecastLineage` provide a full governance trail.

---

## Quick Start

```python
from src.forecasting import MLForecaster, EnsembleForecaster
from src.hierarchy import HierarchyTree, Reconciler
from src.backtesting import BacktestEngine, ChampionSelector
from src.analytics import ForecastComparator, ExceptionEngine, ForecastExplainer
from src.analytics import DriftDetector, ModelCard, ModelCardRegistry, ForecastLineage

# 1. Train and backtest
engine = BacktestEngine(n_folds=4, horizon_weeks=13)
results = engine.run(models=[MLForecaster()], data=panel_df)

# 2. Select champion
champion = ChampionSelector().select(results)

# 3. Reconcile across hierarchy
tree = HierarchyTree.from_dataframe(hierarchy_df)
reconciler = Reconciler(tree)
reconciled = reconciler.reconcile(forecasts_df, method="mint", residuals=residuals_df)

# 4. Compare vs external forecasts and flag exceptions
comparison = ForecastComparator().compare(reconciled, external_forecasts={"field": field_df})
flagged = ExceptionEngine().flag(comparison)
actionable = flagged.filter(pl.col("has_exception"))

# 5. Explain
explainer = ForecastExplainer()
decomp = explainer.decompose(history_df, reconciled)
narratives = explainer.narrative(decomp, comparison)

# 6. Governance
ModelCardRegistry().register(ModelCard.from_backtest("lgbm_direct", "retail", results))
ForecastLineage().record(lob="retail", model_id="lgbm_direct", n_series=500, horizon_weeks=13)
```

---

## Testing

```bash
pip install -r requirements.txt
python -m pytest --ignore=tests/test_metrics.py --ignore=tests/test_feature_engineering.py -v
# 391 tests pass
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| WMAPE | Weighted Mean Absolute Percentage Error (primary) |
| Normalized Bias | Mean signed error / mean actuals |
| RMSPE | Root Mean Square Percentage Error |
| Coverage | % of actuals within P10–P90 interval |
