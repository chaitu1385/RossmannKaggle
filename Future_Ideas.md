# Forecasting Platform — Future Roadmap

## Tier 2: Next Priority

Features that build directly on Tier 1 and existing infrastructure.

| # | Feature | Why | Depends On |
|---|---------|-----|------------|
| 1 | **Scenario Planning / What-If Engine** | Planners need to simulate "what if promo doubles?" or "what if we lose a store?" without touching production forecasts. High business demand. | External Regressors (Tier 1) |
| 2 | **Automated Hyperparameter Tuning** | Current ML models use static defaults (200 estimators, lr=0.05). Per-LOB or per-series tuning can yield 5-15% WMAPE improvement with no new data. | BacktestEngine |
| 3 | **Multi-Horizon Champion Selection** | Current champion is selected for a single horizon. Retail needs different champions for short-term (1-4 weeks) vs. medium-term (5-13) vs. long-term (14-39). | ChampionSelector |
| 4 | **Notification & Alerting Service** | Drift alerts and exception flags exist but only in DataFrames. Need email/Slack/Teams push for critical drift, FVA degradation, and override approval requests. | RBAC + Audit (Tier 1), FVA (Tier 1) |
| 5 | **Override Impact Tracking** | FVA has L4 (override layer) but doesn't track individual planner performance. Need per-planner accuracy attribution to coach override quality. | FVA (Tier 1), OverrideStore |

---

## Tier 3: Longer-Term / Research

Features that require significant design or new dependencies.

| # | Feature | Why | Complexity |
|---|---------|-----|-----------|
| 1 | **Causal Inference for Promotions** | External regressors capture correlation, not causation. Uplift modeling (DoWhy / CausalML) would isolate true promo effect from baseline. | High — new modeling paradigm |
| 2 | **Automated Feature Selection** | Current ML models use all available features. Boruta/SHAP-based selection per series could reduce noise and improve sparse series accuracy. | Medium — integrates with existing SHAP |
| 3 | **Demand Sensing (Short-Horizon)** | Foundation models (Chronos, TimeGPT) are zero-shot but not fine-tuned on domain data. Fine-tuning or transfer learning on recent weeks could improve 1-4 week accuracy. | High — GPU infra, fine-tuning pipeline |
| 4 | **Probabilistic Reconciliation** | Current MinT reconciliation works on point forecasts. Extending to reconcile full quantile distributions would give coherent prediction intervals. | High — active research area (Wickramasuriya et al.) |
| 5 | **Graph Neural Networks for Hierarchy** | Replace linear reconciliation with GNN that learns cross-series relationships. Promising for large hierarchies (>10K nodes). | Very High — R&D, GPU |
| 6 | **Self-Healing Pipelines** | Auto-detect ingestion failures, switch to fallback sources, auto-retrain when drift exceeds thresholds — fully autonomous operation. | High — requires robust Data Ingestion (Infra A) |
| 7 | **Multi-Tenant SaaS Mode** | Current platform is single-tenant. Multi-tenant isolation (data, config, auth) would enable platform-as-a-service deployment. | High — architecture change |
| 8 | **Real-Time Forecast Updates** | Current batch pipeline runs weekly. Streaming ingestion (Kafka/Event Hub) + incremental model updates for intra-week adjustments. | Very High — architecture change |

---

## Analysis Gaps in Current Implementation

Things the platform does partially or not at all that should be addressed.

### Data Layer Gaps
- **No schema enforcement at ingestion** — `DataLoader` reads CSV blindly; type mismatches cause silent downstream errors
- **No data quality scoring** — Null rates, outlier percentages, freshness checks are not computed or reported
- **No database connectivity** — File-based only; enterprise deployments need SQL/warehouse connectors
- **No data lineage beyond model tracking** — `ForecastLineage` tracks which model ran, but not which data version was used
- **Preprocessor is pandas-only** — `DataPreprocessor` doesn't have a Polars equivalent, creating a pandas→Polars boundary

### Model Layer Gaps
- **No model versioning** — `ModelCardRegistry` stores metadata but not the model artifacts themselves (weights, pickle files)
- **No A/B testing framework** — Champion selection is offline (backtest); no support for live A/B between champion and challenger
- **Ensemble weights are static** — `WeightedEnsembleForecaster` weights are set once at construction; no online weight adaptation
- **No conformal prediction** — Quantile forecasts are model-native or bootstrap-based; conformal prediction intervals would give distribution-free coverage guarantees

### Pipeline Layer Gaps
- **No pipeline DAG / dependency management** — `ForecastPipeline` and `BacktestPipeline` are monolithic; no Airflow/Prefect/Dagster integration
- **No retry/checkpoint logic** — If a pipeline fails mid-run, it restarts from scratch
- **No parallel model training** — Models are trained sequentially within `BacktestEngine`; could parallelize across models and folds

### Governance Gaps
- **Override approval workflow is designed but not wired** — `PLAN.md` describes DRAFT→PENDING→APPROVED flow, but `OverrideStore` doesn't enforce status transitions
- **Audit log has no retention policy enforcement** — Designed for 2-year retention but no cleanup job
- **No model card approval workflow** — `ModelCard` can be registered by anyone with `DATA_SCIENTIST` role; no mandatory review gate

### API / Serving Gaps
- **No batch prediction endpoint** — API serves single-LOB/single-series; no endpoint for bulk forecast generation via API
- **No WebSocket / SSE for long-running jobs** — Backtest runs can take minutes; no progress streaming
- **No rate limiting** — JWT auth exists but no throttling per user/role
- **No API versioning** — Endpoints are unversioned (`/forecast/{lob}` not `/v1/forecast/{lob}`)

### Testing Gaps
- **No integration tests against real data** — All 423 tests use synthetic data; no smoke tests with production-scale datasets
- **No performance/load tests** — No benchmarks for API latency, backtest throughput, or memory usage
- **No contract tests for API** — Pydantic schemas exist but no OpenAPI contract testing

---

## Priority Recommendation

```
NOW (Infrastructure):
  ├── Data Ingestion Robustness (Infra Plan A)
  └── CI/CD + Containerization (Infra Plan B)

NEXT (Tier 2):
  ├── Scenario Planning / What-If
  ├── Hyperparameter Tuning
  └── Notification Service

LATER (Tier 3):
  ├── Causal Inference
  ├── Demand Sensing
  └── Probabilistic Reconciliation

ONGOING (Analysis Gaps):
  ├── Override approval wiring
  ├── Model versioning
  └── Pipeline DAG integration
```
