# CLAUDE.md

## Project Overview

Weekly sales forecasting platform for retail S&OP. Python 3.8+, built on FastAPI (REST API), PySpark (distributed execution), and Polars (data processing). Combines statistical, ML, neural, and foundation model forecasting with hierarchical reconciliation.

Main code lives in `forecasting-platform/`.

## Common Commands

```bash
# Install dependencies
pip install -r forecasting-platform/requirements.txt

# Run all tests
python -m pytest forecasting-platform/tests/ --ignore=forecasting-platform/tests/test_metrics.py --ignore=forecasting-platform/tests/test_feature_engineering.py -v

# Run a specific test file
python -m pytest forecasting-platform/tests/test_platform.py -v

# Start the REST API server
python forecasting-platform/scripts/serve.py --port 8000 --data-dir data/

# Run forecast pipeline
python forecasting-platform/scripts/run_forecast.py --config forecasting-platform/configs/platform_config.yaml --lob retail

# Run backtest pipeline
python forecasting-platform/scripts/run_backtest.py --config forecasting-platform/configs/platform_config.yaml --lob retail

# Build package
python forecasting-platform/setup.py sdist bdist_wheel
```

## Architecture

```
forecasting-platform/
├── src/                    # Source modules (~20+ modules)
│   ├── ai/                 # AI-native features (Claude-powered)
│   │   ├── base.py         # AIFeatureBase — shared client wrapper
│   │   ├── nl_query.py     # NaturalLanguageQueryEngine — POST /ai/explain
│   │   ├── anomaly_triage.py # AnomalyTriageEngine — POST /ai/triage
│   │   ├── config_tuner.py # ConfigTunerEngine — POST /ai/recommend-config
│   │   └── commentary.py   # CommentaryEngine — POST /ai/commentary
│   ├── api/                # FastAPI REST endpoints (auth-protected)
│   ├── audit/              # Append-only Parquet audit logging
│   ├── auth/               # RBAC (5 roles, 11 permissions), JWT tokens
│   ├── backtesting/        # Walk-forward validation, champion selection
│   ├── config/             # YAML schema + loader (dataclass-driven)
│   ├── data/               # Data loading, preprocessing, validation, demand cleansing, regressor screening, external regressors
│   │   ├── validator.py    # DataValidator — schema enforcement, duplicate/frequency/range checks
│   │   ├── cleanser.py     # DemandCleanser — outlier detection, stockout imputation, period exclusion
│   │   ├── regressor_screen.py # RegressorScreen — variance, correlation, MI screening
│   │   └── regressors.py   # External regressor loader, holiday calendar, validation
│   ├── evaluation/         # Metric computations (WMAPE, RMSPE, bias, MAE)
│   ├── forecasting/        # Model implementations + registry
│   │   ├── naive.py        # SeasonalNaiveForecaster
│   │   ├── statistical.py  # AutoARIMA, AutoETS, AutoTheta, MSTL
│   │   ├── ml.py           # LGBMDirect, XGBoostDirect
│   │   ├── neural.py       # N-BEATS, NHITS, TFT (via neuralforecast)
│   │   ├── foundation.py   # Chronos, TimeGPT (zero-shot)
│   │   ├── intermittent.py # Croston, CrostonSBA, TSB
│   │   ├── ensemble.py     # WeightedEnsembleForecaster
│   │   ├── hierarchical.py # HierarchicalForecaster
│   │   └── constrained.py  # ConstrainedDemandEstimator (capacity/budget constraints)
│   ├── hierarchy/          # Tree structure, aggregation, reconciliation (OLS/WLS/MinT)
│   ├── metrics/            # MetricStore (Parquet), drift detection, FVA
│   ├── overrides/          # Planner manual override store (DuckDB)
│   ├── pipeline/           # End-to-end backtest + forecast pipelines, provenance manifest
│   │   └── manifest.py     # PipelineManifest — provenance sidecar (JSON) for each forecast run
│   ├── series/             # Series builder, sparse detector, SKU transitions
│   ├── sku_mapping/        # New/discontinued SKU mapping
│   ├── spark/              # PySpark distributed execution
│   └── analytics/          # BI export, comparators, explainability, governance, FVA
├── tests/                  # 790+ tests (pytest)
├── configs/                # YAML configuration files
├── scripts/                # Entry points (run_backtest, run_forecast, serve, spark_*)
└── notebooks/              # Jupyter notebooks for exploration
```

## Code Style

- **Classes**: PascalCase (`LGBMDirectForecaster`, `HierarchyTree`)
- **Functions/methods**: snake_case (`compute_wmape`, `fit_model`)
- **Private members**: leading underscore (`_make_weekly_actuals`)
- **Type hints** on all public APIs
- **Docstrings** in Google/NumPy style with Parameters, Returns, Examples
- **Polars DataFrames** throughout production code (not Pandas)
- **Dataclasses** for configuration (`src/config/schema.py`)
- **ABCs** for extensible interfaces (`BaseForecaster`)
- PEP 8 conventions (no explicit linter configured)

## Configuration

YAML-driven config system with dataclass schema validation:
- `configs/base_config.yaml` — platform defaults
- `configs/platform_config.yaml` — standard deployment
- `configs/fabric_config.yaml` — Microsoft Fabric settings
- `configs/lob/` — line-of-business overrides (inherit from base)
- Schema defined in `src/config/schema.py`

Key config dataclasses: `ForecastConfig`, `BacktestConfig`, `DataQualityConfig` (contains `ValidationConfig`, `CleansingConfig`), `ConstraintConfig`, `ExternalRegressorConfig` (contains `RegressorScreenConfig`), `AIConfig`

### Multi-frequency support

The platform supports daily (`"D"`), weekly (`"W"`), monthly (`"M"`), and quarterly (`"Q"`) data frequencies. The `FREQUENCY_PROFILES` dict in `src/config/schema.py` is the single source of truth, mapping each frequency to: `season_length`, `default_lags`, `min_series_length`, `default_val_periods`, `default_horizon`, `statsforecast_freq`, and `timedelta_kwargs`.

Set `frequency` in the YAML config:
```yaml
forecast:
  frequency: "M"          # "D" | "W" | "M" | "Q"
  horizon_periods: 12     # alias for horizon_weeks (backward-compat)
backtest:
  val_periods: 3          # alias for val_weeks (backward-compat)
```

Helper functions: `get_frequency_profile(freq)` returns the profile dict; `freq_timedelta(freq, periods)` returns a `timedelta` for date arithmetic. All models, backtesting, validation, and data processing use these instead of hardcoded weekly values.

## Testing

- Framework: pytest
- Test files mirror source structure with `test_` prefix
- Helper fixtures use `_make_*` factory functions (e.g., `_make_weekly_actuals`)
- Skip `test_metrics.py` and `test_feature_engineering.py` (legacy/slow)
- 860+ tests across 38 test files
- Key test modules: `test_platform.py` (85 tests), `test_ai_*.py` (73), `test_forecast_explainability.py` (59), `test_intermittent_demand.py` (55)

## Key Dependencies

Core: polars, statsforecast, mlforecast, lightgbm, xgboost, scikit-learn, fastapi, pyyaml, duckdb
Optional: neuralforecast, pyspark, shap, pyjwt, bcrypt, holidays, delta-spark, azure-identity

## Documentation Convention

When adding a new module or capability, update these files:

1. **`README.md`** (root) — Add module docs (class table, description). Update test count and dependency list if changed.
2. **`CLAUDE.md`** (root) — Update architecture tree. Update test count. Add new config dataclasses if applicable.
3. **`CONCEPTS.md`** (root) — Add a concept entry if the feature introduces a new "why" that non-domain-experts need to understand (3-4 sentences: what, why, when).
4. **`EDGE_CASES.md`** (root) — Add an entry if the feature handles a new failure mode (what happens, how we handle it, what to watch for).

These are the only documentation files in the repo. All other docs (plans, specs, analyses) are transient working documents — delete them once the work is merged.
