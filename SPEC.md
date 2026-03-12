# Forecasting Platform — Specification Document

> **Living document.** Updated as each feature is built.
> Last updated: 2026-03-12

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Configuration System](#3-configuration-system)
4. [Data Pipeline](#4-data-pipeline)
5. [Forecasting Engine](#5-forecasting-engine)
6. [Hierarchy & Reconciliation](#6-hierarchy--reconciliation)
7. [Backtesting & Model Selection](#7-backtesting--model-selection)
8. [SKU Mapping Discovery](#8-sku-mapping-discovery)
9. [Product Transition Engine](#9-product-transition-engine)
10. [Planner Overrides](#10-planner-overrides)
11. [Cloud Integration](#11-cloud-integration)
12. [Metrics](#12-metrics)
13. [CLI Entry Points](#13-cli-entry-points)
14. [Testing](#14-testing)
15. [Roadmap](#15-roadmap)
16. [Changelog](#16-changelog)

---

## 1. Project Overview

**Forecasting Platform** is a modular, production-ready time series forecasting system for predicting retail store sales. It supports multi-hierarchy aggregation, automated SKU transition detection, walk-forward backtesting, and cloud deployment on Microsoft Fabric / Apache Spark.

| Property        | Value                                   |
|-----------------|-----------------------------------------|
| Domain          | Retail sales forecasting                |
| Forecast horizon| 39 weeks (configurable)                 |
| Granularity     | Product × Geography × Channel          |
| Primary metric  | WMAPE (Weighted Mean Absolute % Error)  |
| Target variable | Weekly sales units                      |
| Language        | Python 3.8+                             |

---

## 2. Architecture Overview

```
configs/              YAML-driven configuration (no hard-coded params)
src/
  config/             Config schema (dataclasses) + YAML loader
  data/               Data loading & preprocessing (CSV, Delta, Polars DF)
  series/             Series builder + product transition logic
  forecasting/        Model registry + Base/ML/Statistical/Naive forecasters
  backtesting/        Walk-forward CV engine + champion selection
  hierarchy/          Tree, aggregation, disaggregation, reconciliation
  metrics/            WMAPE, bias, MAE, MAPE, RMSE
  sku_mapping/        Automated SKU replacement discovery pipeline
  overrides/          Planner override store (DuckDB)
  pipeline/           Forecast & backtest orchestration
  spark/              Spark session factory + Delta Lake utils
  fabric/             Microsoft Fabric / OneLake lakehouse I/O
scripts/              CLI entry points
tests/                Unit & integration tests
notebooks/            Exploratory Python notebooks (data prep, backtest, forecast)
```

### Key Design Principles

- **Config-driven**: all behaviour declared in YAML; no magic constants in code.
- **Registry pattern**: models and SKU-mapping methods registered by name, instantiated from config.
- **Temporal integrity**: all train/validation splits are time-ordered (no random splits).
- **Polars-native core**: vectorised operations; no row-level Python loops in hot paths.
- **Graceful fallbacks**: heavy dependencies (statsforecast, mlforecast, pyspark) degrade safely if unavailable.
- **Dependency injection**: config objects passed explicitly; no module-level singletons.

---

## 3. Configuration System

### Files

| File | Purpose |
|------|---------|
| `configs/platform_config.yaml` | Master config — hierarchies, reconciliation strategy, horizon |
| `configs/base_config.yaml` | Legacy data paths & feature engineering flags |
| `configs/xgboost_config.yaml` | XGBoost hyperparameters |
| `configs/lightgbm_config.yaml` | LightGBM hyperparameters |
| `configs/sku_mapping_config.yaml` | Method weights, confidence thresholds |
| `configs/fabric_config.yaml` | Microsoft Fabric workspace & lakehouse settings |
| `configs/lob/surface.yaml` | Surface LOB example |
| `configs/lob/walmart_example.yaml` | Walmart LOB example |

### Schema (`src/config/schema.py`)

Key dataclasses:

```
PlatformConfig
  ├── HierarchyConfig       (levels, columns, aggregation order)
  ├── ReconciliationConfig  (method: middle-out | bottom-up | top-down)
  ├── BacktestConfig        (n_folds, validation_weeks, metric)
  └── ForecastConfig        (horizon, models, output_path)
```

Config is loaded once at pipeline start and passed as a dependency to all modules.

---

## 4. Data Pipeline

### Loading (`src/data/`)

| Source | Method |
|--------|--------|
| CSV | `pandas.read_csv` / `polars.read_csv` |
| Delta Lake | Spark DataFrame → Polars via Arrow |
| Microsoft Fabric OneLake | `src/fabric/lakehouse.py` |
| In-memory | Polars DataFrame passed directly |

### Preprocessing

- Type casting, null handling, categorical encoding.
- Composite series ID built from hierarchy columns (e.g. `{product}_{store}_{channel}`).
- Missing weeks zero-filled (gap filling).
- Log-transform of target optional via config.

### Series Builder (`src/series/builder.py`)

Builds a tidy Polars DataFrame with columns `[date, series_id, sales, *features]` ready for the forecasting engine.

---

## 5. Forecasting Engine

### Model Registry (`src/forecasting/`)

All models inherit `BaseForecaster` and implement:

```python
fit(df: pl.DataFrame, config: ForecastConfig) -> None
predict(horizon: int) -> pl.DataFrame          # returns date + forecast cols
get_params() -> dict
```

Registered models:

| Key | Class | Backend |
|-----|-------|---------|
| `naive_seasonal` | `SeasonalNaiveForecaster` | manual (52-week repeat) |
| `auto_arima` | `AutoARIMAForecaster` | statsforecast |
| `auto_ets` | `AutoETSForecaster` | statsforecast |
| `xgb_direct` | `XGBoostForecaster` | xgboost + mlforecast |
| `lgbm_direct` | `LightGBMForecaster` | lightgbm + mlforecast |

### Feature Engineering

| Category | Features |
|----------|---------|
| Temporal | year, month, day, dow, woy, quarter, is_weekend, is_month_start/end |
| Lags | configurable (default: 1, 7, 14, 30 days) |
| Rolling | mean & std over configurable windows (default: 7, 14, 30 days) |
| Domain | competition distance, promo flags |

---

## 6. Hierarchy & Reconciliation

### Hierarchy Tree (`src/hierarchy/tree.py`)

Three dimensions supported out-of-the-box:
- **Product** (SKU → Product Family → Segment → Total)
- **Geography** (Store → Region → Country)
- **Channel** (Online, Retail, Wholesale)

### Aggregation / Disaggregation (`src/hierarchy/aggregator.py`)

- **Roll-up**: sum children → parent (fully vectorised with Polars group-by).
- **Disaggregate**: split parent → children using historical proportions (or equal split).

### Reconciliation (`src/hierarchy/reconciler.py`)

| Method | Status |
|--------|--------|
| Bottom-up | ✅ Phase 1 |
| Top-down | ✅ Phase 1 |
| Middle-out | ✅ Phase 1 |
| OLS | ✅ Phase 2 |
| WLS | ✅ Phase 2 |
| MinT (Ledoit-Wolf shrinkage) | ✅ Phase 2 |

---

## 7. Backtesting & Model Selection

### Walk-Forward CV (`src/backtesting/`)

```
| train | val |            fold 1
       | train | val |     fold 2
              | train | val | fold 3
```

- `n_folds`: default 3 (configurable).
- `validation_weeks`: default 13 (configurable).
- Temporal ordering always respected.

### Champion Selection

- Primary: lowest mean WMAPE across folds.
- Secondary (tie-break): normalised bias closest to 0.
- Champion selected per LOB granularity (or per `product_group` / `series_id` if configured).
- Results written to Parquet metric store with `run_id`, `model_id`, `fold`, `series_id`.

---

## 8. SKU Mapping Discovery

### Goal

Automatically identify which discontinued / declining SKUs map to new replacement SKUs, so historical demand can be inherited.

### Pipeline (`src/sku_mapping/pipeline.py`)

```
Load product master
  → Run discovery methods in parallel
      → Attribute Matching
      → Naming Convention Matching
  → Fuse candidates (weighted score + multi-method bonus)
  → Assign confidence level & mapping type
  → Filter by min_confidence threshold
  → Write mapping table to CSV
```

### Discovery Methods

#### Attribute Matching (`src/sku_mapping/methods/attribute_matching.py`)

Pairs `Discontinued/Declining` SKUs with `Active/Planned` SKUs within the same product family/segment, scoring on:

| Signal | Weight |
|--------|--------|
| Price tier match | configurable |
| Form factor match | configurable |
| Category match | configurable |
| Launch gap (recency) | configurable |

#### Naming Convention (`src/sku_mapping/methods/naming_parsing.py`)

Fuzzy string matching via `rapidfuzz` to detect naming patterns (e.g. `Surface Pro 9` → `Surface Pro 10`).

### Fusion & Confidence (`src/sku_mapping/fusion/scorer.py`)

```
final_score = (2/3 × attribute_score) + (1/3 × naming_score)
              + 0.10  if ≥2 methods agree (multi-method bonus)
```

| Level | Threshold |
|-------|-----------|
| High | ≥ 0.75 |
| Medium | ≥ 0.50 |
| Low | ≥ 0.30 |
| Very Low | < 0.30 (filtered out by default) |

### Mapping Types

`1-to-1`, `1-to-Many`, `Many-to-1`, `Many-to-Many` — with proportional weights per target SKU.

### Planned Methods (Phase 2)

- **Curve fitting**: match demand curve shape between old and new SKU.
- **Temporal matching**: detect temporal co-movement at launch/discontinuation events.

---

## 9. Product Transition Engine

### Scenarios (`src/series/transition.py`)

| Scenario | Condition | Behaviour |
|----------|-----------|-----------|
| A | New SKU already launched | Stitch old SKU history onto new SKU series |
| B | New SKU launches within forecast horizon | Ramp-down old + ramp-up new over 13-week window |
| C | New SKU launches beyond horizon | Forecast old SKU only, no transition |

### Ramp Shapes

| Shape | Status |
|-------|--------|
| Linear | ✅ Phase 1 |
| S-curve | 🔲 Phase 2 |
| Step | 🔲 Phase 2 |

---

## 10. Planner Overrides

`src/overrides/` — DuckDB-backed store.

Planners can manually adjust forecasts at any hierarchy level. Overrides are applied post-reconciliation and frozen before publishing.

---

## 11. Cloud Integration

### Microsoft Fabric (`src/fabric/`)

- Auto-detects Fabric environment via env vars.
- Reads/writes Delta tables on OneLake via `lakehouse.py`.
- `fabric_config.yaml` holds workspace, lakehouse, and table names.

### Apache Spark (`src/spark/`)

| Module | Purpose |
|--------|---------|
| `session.py` | Spark session factory (local or cluster) |
| `utils.py` | Polars ↔ Spark DataFrame conversions via Arrow |
| `features.py` | Distributed feature engineering |

Spark scripts: `scripts/spark_forecast.py`, `scripts/spark_backtest.py`.

---

## 12. Metrics

All metrics defined in `src/metrics/definitions.py`.

| Metric | Formula | Primary Use |
|--------|---------|-------------|
| WMAPE | `Σ|actual−forecast| / Σactual` | Champion selection |
| Normalised Bias | `Σ(forecast−actual) / Σactual` | Bias detection |
| MAE | `mean(|actual−forecast|)` | Absolute scale |
| MAPE | `mean(|actual−forecast| / actual)` | Unit-free comparison |
| RMSE | `sqrt(mean((actual−forecast)²))` | Penalises outliers |

---

## 13. CLI Entry Points

| Script | Command | Description |
|--------|---------|-------------|
| `scripts/run_backtest.py` | `python run_backtest.py --config configs/platform_config.yaml` | Walk-forward CV, writes metric store |
| `scripts/run_forecast.py` | `python run_forecast.py --config configs/platform_config.yaml` | Full forecast pipeline, writes Parquet |
| `scripts/run_sku_mapping.py` | `python run_sku_mapping.py --config configs/sku_mapping_config.yaml` | SKU discovery, writes mapping CSV |
| `scripts/spark_forecast.py` | `spark-submit spark_forecast.py` | Distributed forecast on Spark |
| `scripts/spark_backtest.py` | `spark-submit spark_backtest.py` | Distributed backtest on Spark |

---

## 14. Testing

```
tests/
  test_platform.py           End-to-end smoke tests (config, hierarchy, metrics, series, backtest)
  test_sku_mapping.py        SKU discovery unit tests (attribute, naming, fusion, writer)
  test_feature_engineering.py Temporal, lag, rolling feature tests
  test_metrics.py            WMAPE, bias calculation tests
```

Run all tests:

```bash
cd forecasting-platform
pytest tests/ -v
```

---

## 15. Roadmap

### Phase 1 — MVP (Current)

- [x] Config-driven platform architecture
- [x] Seasonal Naive, AutoARIMA, AutoETS, XGBoost, LightGBM forecasters
- [x] Walk-forward backtesting with champion selection
- [x] Bottom-up, top-down, middle-out reconciliation
- [x] SKU mapping: attribute matching + naming convention methods
- [x] Product transition scenarios A / B / C (linear ramp)
- [x] Planner override store (DuckDB)
- [x] Microsoft Fabric / Spark integration (dev-ready)
- [x] Polars-native core data operations

### Phase 2 — Production Hardening

- [x] MinT reconciliation (OLS / WLS)
- [x] SKU mapping: curve-fitting method
- [x] SKU mapping: temporal co-movement method
- [ ] S-curve and step ramp shapes for transitions
- [ ] Enhanced proportion estimation (Bayesian)
- [ ] Production Fabric deployment pipeline
- [ ] Monitoring & drift detection
- [ ] REST API / serving layer

---

## 16. Changelog

| Date | Version | Summary |
|------|---------|---------|
| 2026-03-12 | 0.1.0 | Initial spec written. Phase 1 MVP complete: forecasting engine, backtesting, hierarchy, SKU mapping (2 methods), product transitions, Fabric/Spark integration. |
| 2026-03-12 | 0.2.0 | Phase 2 — MinT reconciliation: added OLS, WLS, and MinT (Ledoit-Wolf shrinkage) methods to `reconciler.py`. 7 new tests. |
| 2026-03-12 | 0.3.0 | Phase 2 — SKU mapping curve-fitting method: `CurveFittingMethod` scores demand transitions via decline/ramp/complementarity/scale signals. `build_phase2_pipeline()` factory added. 13 new tests. |
| 2026-03-12 | 0.4.0 | Phase 2 — SKU mapping temporal co-movement method: `TemporalCovementMethod` scores demand transitions via correlation/overlap/volume signals. `build_phase2_pipeline()` updated to include all 4 methods. 9 new tests. |

---

> **How to update this doc**: When a new feature is merged, add a row to the Changelog, tick off the Roadmap item (or add a new one), and update the relevant section. Keep descriptions concise — this is a spec, not a tutorial.
