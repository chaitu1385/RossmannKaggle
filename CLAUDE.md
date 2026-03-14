# CLAUDE.md

## Project Overview

Weekly sales forecasting platform for retail S&OP. Python 3.8+, built on FastAPI (REST API), PySpark (distributed execution), and Polars (data processing). Combines statistical, ML, and foundation model forecasting with hierarchical reconciliation.

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
├── src/                    # Source modules (~12K LOC, 20 modules)
│   ├── api/                # FastAPI REST endpoints (auth-protected)
│   ├── audit/              # Append-only Parquet audit logging
│   ├── auth/               # RBAC (5 roles, 11 permissions), JWT tokens
│   ├── backtesting/        # Walk-forward validation, champion selection
│   ├── config/             # YAML schema + loader (dataclass-driven)
│   ├── data/               # Data loading, preprocessing, external regressors
│   ├── evaluation/         # Metric computations (WMAPE, RMSPE, bias, MAE)
│   ├── forecasting/        # Model implementations + registry
│   ├── hierarchy/          # Tree structure, aggregation, reconciliation (OLS/WLS/MinT)
│   ├── metrics/            # MetricStore (Parquet), drift detection, FVA
│   ├── overrides/          # Planner manual override store (DuckDB)
│   ├── pipeline/           # End-to-end backtest + forecast pipelines
│   ├── series/             # Series builder, sparse detector, SKU transitions
│   ├── sku_mapping/        # New/discontinued SKU mapping
│   ├── spark/              # PySpark distributed execution
│   └── analytics/          # BI export, comparators, explainability
├── tests/                  # 423 tests (pytest)
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

## Testing

- Framework: pytest
- Test files mirror source structure with `test_` prefix
- Helper fixtures use `_make_*` factory functions (e.g., `_make_weekly_actuals`)
- Skip `test_metrics.py` and `test_feature_engineering.py` (legacy/slow)
- Key test modules: `test_platform.py` (85 tests), `test_sku_mapping.py` (67), `test_forecast_explainability.py` (59)

## Key Dependencies

Core: polars, statsforecast, mlforecast, lightgbm, xgboost, scikit-learn, fastapi, pyyaml, duckdb
Optional: pyspark, shap, pyjwt, bcrypt, holidays, delta-spark, azure-identity
