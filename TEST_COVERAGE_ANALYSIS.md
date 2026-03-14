# Test Coverage Analysis

**Date:** 2026-03-14
**Overall Coverage:** 64% (1671 of 4673 statements uncovered)
**Test Results:** 422 passed, 10 failed (all API integration tests)

---

## Current Coverage by Module

| Module | Coverage | Notes |
|--------|----------|-------|
| **Config** (schema, loader) | 96% | Well tested |
| **SKU Mapping** (pipeline, methods, fusion) | 90-97% | Excellent |
| **Forecasting** (base, naive, ensemble, intermittent, registry) | 83-97% | Good |
| **Series** (builder, sparse_detector, transition) | 81-92% | Good |
| **Metrics** (definitions, drift, fva, store) | 71-95% | Moderate to good |
| **Hierarchy** (tree, reconciler) | 73-86% | Moderate |
| **Pipeline** (backtest, forecast) | 80-92% | Good |
| **Auth/RBAC** (models, rbac, token) | 91-100% | Good |
| **Audit** (logger, schemas) | 87-93% | Good |
| **Analytics** (explainer, fva_analyzer, comparator, etc.) | 72-100% | Mixed |
| **Forecasting ML** (ml.py) | **22%** | Very low |
| **Forecasting Statistical** (statistical.py) | **28%** | Very low |
| **Models** (lightgbm_model, xgboost_model) | **30-31%** | Very low |
| **Hierarchy Aggregator** | **34%** | Low |
| **Fabric** (delta_writer, deployment, lakehouse) | **14-36%** | Very low |
| **Evaluation** (evaluator) | **38%** | Low |
| **Overrides** (store) | **0%** | No tests |
| **Spark** (all 6 modules) | **0%** | No tests |
| **Utils** (config, logger) | **0%** | No tests |
| **Data** (loader, preprocessor) | Not imported | No tests |

---

## Priority Areas for Test Improvement

### Priority 1 — Critical Business Logic with Low/No Coverage

#### 1. ML Forecasters (`src/forecasting/ml.py` — 22% coverage)
This module contains the LightGBM and XGBoost forecaster wrappers used for production forecasting. The untested code includes:
- `fit()` with external regressors and feature engineering
- `predict()` with future feature handling
- `predict_quantiles()` for probabilistic forecasts
- Direct vs. recursive multi-step strategy selection
- Feature importance extraction

**Recommended tests:**
- Unit test `fit()` / `predict()` with synthetic data
- Test external regressor integration in ML models
- Test quantile prediction output shape and ordering
- Test direct vs. recursive strategy behavior
- Test error handling for missing features

#### 2. Statistical Forecasters (`src/forecasting/statistical.py` — 28% coverage)
AutoARIMA and AutoETS are core statistical models. Nearly all their fit/predict logic is untested.

**Recommended tests:**
- Test `fit()` / `predict()` for AutoARIMA and AutoETS with synthetic seasonal data
- Test that predictions have correct shape and horizon
- Test `predict_quantiles()` output
- Test fallback behavior when statsforecast is unavailable

#### 3. Override Store (`src/overrides/store.py` — 0% coverage)
Planner overrides directly affect forecast outputs in S&OP workflows. This DuckDB-backed store has zero tests.

**Recommended tests:**
- Test CRUD operations (create, read, update, delete overrides)
- Test override approval workflow
- Test override application to forecasts
- Test concurrent access / data integrity
- Test schema validation and rejection of malformed overrides

#### 4. Hierarchy Aggregator (`src/hierarchy/aggregator.py` — 34% coverage)
Aggregation and disaggregation are critical for hierarchical forecasting coherence. The untested portions include `disaggregate()` and proportion-based top-down methods.

**Recommended tests:**
- Test `disaggregate()` with historical proportions
- Test round-trip: aggregate then disaggregate preserves leaf totals
- Test handling of zero-sum parents
- Test middle-out reconciliation flow

---

### Priority 2 — Infrastructure and Deployment Modules

#### 5. Fabric Integration (`src/fabric/` — 14-36% coverage)
Delta Lake writing and Fabric deployment are critical for production. Almost no integration paths are tested.

**Recommended tests (with mocks):**
- Test `DeltaWriter.write()` with mock Delta table
- Test deployment pipeline stages (validate, write, register)
- Test Lakehouse read/write operations with mock Spark session
- Test configuration validation and error paths
- Test schema evolution handling

#### 6. Spark Modules (`src/spark/` — 0% coverage)
Six modules totaling ~306 lines with zero test coverage. These handle distributed execution.

**Recommended tests (with mocks):**
- Test `SparkSeriesBuilder` logic with mock PySpark DataFrames
- Test `SparkFeatureEngineering` produces expected columns
- Test `SparkPipeline` orchestration and stage sequencing
- Test session configuration and Delta Lake integration
- Test Spark utility functions independently

---

### Priority 3 — API and Integration Testing

#### 7. REST API Integration Tests (10 tests failing)
All `TestRestApi` tests are failing. The API layer (`src/api/app.py`) is a critical interface.

**Recommended fixes and additions:**
- Fix the 10 failing tests (likely missing `httpx` or `TestClient` dependency)
- Add tests for auth-protected endpoints (token validation, role enforcement)
- Add tests for error responses (400, 401, 403, 422)
- Test request validation via Pydantic schemas
- Test pagination and filtering parameters

#### 8. Evaluation Module (`src/evaluation/evaluator.py` — 38% coverage)
The `Evaluator` class orchestrates metric computation and model ranking.

**Recommended tests:**
- Test `evaluate()` returns correct metric dictionary
- Test model ranking logic
- Test handling of edge cases (empty predictions, all-zero actuals)

---

### Priority 4 — Edge Cases and Robustness

#### 9. Data Loader & Preprocessor (`src/data/loader.py`, `src/data/preprocessor.py`)
These modules handle raw data ingestion. They're not even imported by current tests.

**Recommended tests:**
- Test CSV/Parquet loading with synthetic files
- Test missing value imputation strategies
- Test categorical encoding
- Test schema validation and column type enforcement
- Test handling of malformed/corrupt input files

#### 10. Utility Modules (`src/utils/` — 0% coverage)
Small but used across the platform.

**Recommended tests:**
- Test logger configuration (log levels, handlers)
- Test config utility functions

---

### Priority 5 — Cross-Cutting Concerns

#### 11. End-to-End Pipeline Tests
The existing pipeline tests use mocked/simplified data. There are no true integration tests that exercise the full data flow.

**Recommended tests:**
- Full pipeline from CSV load through reconciled forecast output
- Pipeline with external regressors enabled
- Pipeline with SKU transitions active
- Pipeline with override application
- Verify audit trail is produced for each pipeline run

#### 12. Error Handling and Boundary Conditions
Across the codebase, error paths are undertested:
- What happens with empty DataFrames?
- What happens with single-row time series?
- What happens when all models fail during backtesting?
- What happens with misaligned date indices?
- What happens with NaN/Inf values in features?

---

## Summary of Recommendations

| Priority | Area | Current | Target | Effort |
|----------|------|---------|--------|--------|
| P1 | ML Forecasters | 22% | 80%+ | Medium |
| P1 | Statistical Forecasters | 28% | 80%+ | Medium |
| P1 | Override Store | 0% | 80%+ | Medium |
| P1 | Hierarchy Aggregator | 34% | 80%+ | Low |
| P2 | Fabric Integration | 14-36% | 70%+ | High (mocking) |
| P2 | Spark Modules | 0% | 60%+ | High (mocking) |
| P3 | REST API (fix + expand) | Failing | 90%+ | Medium |
| P3 | Evaluation Module | 38% | 80%+ | Low |
| P4 | Data Loader/Preprocessor | 0% | 70%+ | Low |
| P4 | Utility Modules | 0% | 80%+ | Low |
| P5 | E2E Pipeline Tests | Partial | Full flow | High |
| P5 | Edge Cases / Robustness | Sparse | Comprehensive | Medium |

Achieving these targets would raise overall coverage from **64% to approximately 85%+** and significantly improve confidence in production deployments.
