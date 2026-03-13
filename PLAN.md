# Implementation Plan — Tier 1 Features

## Feature 1: External Regressors / Promotional Calendar

### Goal
Enable ML models (LightGBM, XGBoost) to use external features — promotions, holidays, price, competitor events — for both training and prediction. This is the single biggest accuracy unlock (30-40% WMAPE reduction in retail).

### Current State
- ML models (`src/forecasting/ml.py`) use mlforecast with built-in lags [1,2,4,8,13,26,52] and date features [week, month, quarter]
- `SeriesBuilder.build()` accepts actuals + product_master + mappings + overrides — no external features
- `ForecastPipeline.run()` threads data through SeriesBuilder → fit → predict → reconcile
- Statistical models (ARIMA, ETS) are univariate — they'll ignore external features (that's fine)
- Config schema has no external regressor section

### Files to Modify
1. `src/config/schema.py` — add `ExternalRegressorConfig` dataclass
2. `src/config/loader.py` — parse new config section
3. `src/series/builder.py` — accept and join external features to actuals
4. `src/forecasting/ml.py` — pass external features to mlforecast `fit(X=)` and `predict(X=)`
5. `src/pipeline/forecast.py` — thread external features through pipeline
6. `src/pipeline/backtest.py` — thread external features through backtest
7. `configs/platform_config.yaml` — add example external_regressors section

### New Files
8. `src/data/regressors.py` — loader/validator for external feature data (holiday calendar generator, promo file reader, validation logic)
9. `tests/test_external_regressors.py` — end-to-end test

### Design

**Config addition:**
```yaml
forecast:
  external_regressors:
    enabled: true
    feature_columns:
      - promotion_flag
      - holiday_flag
      - price_index
    future_features_path: data/future_features.parquet  # known-in-advance features for forecast horizon
```

**Data flow:**
```
actuals (series_id, week, quantity)
    +
external_features (series_id, week, promotion_flag, holiday_flag, price_index)
    ↓
SeriesBuilder.build() — left join on [series_id, week], forward-fill nulls
    ↓
ML model — mlforecast.fit(df, X=external_features)
    ↓
ML predict — mlforecast.predict(h=horizon, X=future_features)
```

**Key constraints:**
- All external features must be at same grain (series_id × week) or broadcastable (week-only for holidays)
- Future feature values MUST be provided for the forecast horizon (you can't predict with unknown promos)
- Statistical/naive models silently ignore external features — no code changes needed there
- Backward compatible: if `external_regressors.enabled = false` or section absent, no behavior change

**Regressor loader (`src/data/regressors.py`):**
- `load_external_features(path) -> pl.DataFrame` — reads Parquet/CSV
- `generate_holiday_calendar(country, start, end) -> pl.DataFrame` — uses Python `holidays` lib
- `validate_regressors(features, actuals, config)` — checks grain alignment, no nulls in training window, future values present for horizon

### Implementation Order
1. Config schema + loader (foundation)
2. Regressor loader/validator (data layer)
3. SeriesBuilder changes (join features)
4. ML model changes (pass X to mlforecast)
5. Pipeline threading (forecast + backtest)
6. Config YAML example
7. Tests

### Risks
- mlforecast's `X` parameter behavior may vary across versions — need to pin/test
- Holiday library adds a dependency — make it optional with graceful fallback
- Users may provide features with missing future values — validator must catch this early

---

## Feature 2: RBAC + Audit Trail

### Goal
Add role-based access control and immutable audit logging so the platform meets enterprise procurement requirements (SOX compliance, access control, change attribution).

### Current State
- API (`src/api/app.py`) has 5 GET endpoints, zero authentication
- OverrideStore has optional `created_by` field but it's unenforced
- ForecastLineage records model runs but not WHO triggered them or approved them
- ModelCard/ModelCardRegistry has no ownership or approval workflow
- Logging is transient stdout — no persistent audit trail

### Files to Modify
1. `src/api/app.py` — add auth middleware, inject user context, protect endpoints
2. `src/api/schemas.py` — add user/role response models, audit event schema
3. `src/overrides/store.py` — enforce user context, add approval fields
4. `src/analytics/governance.py` — add user_id to ForecastLineage, approval workflow to ModelCard

### New Files
5. `src/auth/models.py` — User, Role, Permission dataclasses
6. `src/auth/rbac.py` — role definitions, permission checks, FastAPI dependency
7. `src/auth/token.py` — JWT token creation/validation (pluggable for OAuth2 later)
8. `src/audit/logger.py` — append-only audit log (Parquet-backed, immutable)
9. `src/audit/schemas.py` — AuditEvent dataclass
10. `tests/test_rbac.py` — RBAC enforcement tests
11. `tests/test_audit.py` — audit trail tests

### Design

**Role hierarchy:**
```
ADMIN          — full access, manage users, approve model promotions
DATA_SCIENTIST — run backtests, promote models, modify configs
PLANNER        — create/edit overrides, view forecasts, cannot change model config
MANAGER        — approve overrides above threshold, view reports
VIEWER         — read-only access to forecasts, metrics, reports
```

**Permission matrix:**
| Action                    | ADMIN | DATA_SCIENTIST | PLANNER | MANAGER | VIEWER |
|---------------------------|-------|----------------|---------|---------|--------|
| View forecasts/metrics    | Y     | Y              | Y       | Y       | Y      |
| Create overrides          | Y     | Y              | Y       | N       | N      |
| Approve overrides (>X%)   | Y     | N              | N       | Y       | N      |
| Run backtest/pipeline     | Y     | Y              | N       | N       | N      |
| Promote champion model    | Y     | Y              | N       | N       | N      |
| Modify config             | Y     | Y              | N       | N       | N      |
| View audit log            | Y     | Y              | N       | Y       | N      |
| Manage users              | Y     | N              | N       | N       | N      |

**Auth flow (JWT-based, pluggable):**
```
Client → POST /auth/token (username + password) → JWT
Client → GET /forecast/lob (Authorization: Bearer <JWT>) → data
```
- JWT contains: user_id, email, role, exp
- FastAPI `Depends(get_current_user)` extracts and validates on every request
- Token provider is pluggable — swap JWT for OAuth2/SAML later without changing endpoint code

**Audit log schema (append-only Parquet):**
```
audit_id:        str (UUID)
timestamp:       datetime (UTC)
user_id:         str
user_role:       str
action:          str (e.g., "create_override", "promote_model", "view_forecast")
resource_type:   str (e.g., "override", "model_card", "forecast")
resource_id:     str
status:          str (SUCCESS | DENIED | FAILED)
old_value:       str (JSON, nullable — for updates)
new_value:       str (JSON, nullable — for creates/updates)
ip_address:      str (nullable)
request_id:      str (for tracing)
```
- Parquet files partitioned by date: `audit_log/date=2026-03-13/`
- Append-only: no UPDATE or DELETE operations
- Retention: configurable (default 2 years)

**Override approval workflow enhancement:**
```
Planner creates override → status=DRAFT
  → If override proportion > threshold (configurable, e.g., 20%):
      → Requires MANAGER approval → status=PENDING_APPROVAL
      → Manager approves → status=APPROVED (audit logged)
      → Manager rejects → status=REJECTED (audit logged)
  → If below threshold:
      → Auto-approved → status=APPROVED
```

### Implementation Order
1. Auth models + RBAC definitions (foundation)
2. JWT token provider (auth mechanism)
3. Audit logger (Parquet-backed, append-only)
4. API middleware integration (protect endpoints)
5. Override store enhancement (approval workflow)
6. Governance enhancement (user tracking in lineage/model cards)
7. Tests

### Risks
- JWT secret management — use env var, document rotation procedure
- Password storage — use bcrypt; for MVP, a JSON user file; for prod, defer to external IdP
- Audit log volume at scale — Parquet partitioning by date handles this well
- Migration path — existing overrides/lineage records won't have user_id; handle gracefully with "system" default

---

## Feature 3: Forecast Value Add (FVA) Analysis

### Goal
Measure how much accuracy each forecast layer contributes: naive baseline → statistical → ML → post-override. Shows stakeholders which layers justify their cost and which overrides help vs hurt.

### Current State
- BacktestEngine runs each model independently per fold — doesn't capture layered cascade
- MetricStore schema has `model_id` but no `forecast_layer` concept
- ChampionSelector ranks models by WMAPE but doesn't measure incremental improvement
- ForecastComparator aligns system vs external forecasts — not layer vs layer
- BIExporter produces model_leaderboard but no FVA tables
- All metrics (WMAPE, bias, MAE) are already implemented in `src/metrics/definitions.py`

### Files to Modify
1. `src/backtesting/engine.py` — add mode to run models in layer cascade, capture intermediate outputs
2. `src/metrics/store.py` — add optional `forecast_layer` and `parent_forecast` columns
3. `src/analytics/bi_export.py` — add FVA export tables

### New Files
4. `src/metrics/fva.py` — FVA computation engine (incremental WMAPE, bias, value-add classification)
5. `src/analytics/fva_analyzer.py` — FVA aggregation and reporting (by LOB, layer, sparse class, time)
6. `tests/test_fva.py` — FVA computation tests

### Design

**Layer definitions:**
```
L1: Naive       — SeasonalNaiveForecaster (always the baseline)
L2: Statistical — Best of [AutoARIMA, AutoETS] per series (or Croston/TSB for sparse)
L3: ML          — Best of [LightGBM, XGBoost] per series
L4: Override    — Planner-adjusted forecast (if override exists, else L4 = L3)
```

**FVA computation per (series_id, target_week, fold):**
```python
# Layer errors
wmape_naive = wmape(actual, forecast_naive)
wmape_stat  = wmape(actual, forecast_stat)
wmape_ml    = wmape(actual, forecast_ml)
wmape_ovr   = wmape(actual, forecast_override)

# FVA = error reduction from previous layer (positive = improvement)
fva_stat  = wmape_naive - wmape_stat     # stat over naive
fva_ml    = wmape_stat  - wmape_ml       # ml over stat
fva_ovr   = wmape_ml    - wmape_ovr      # override over ml
fva_total = wmape_naive - wmape_ovr      # total improvement over baseline
```

**FVA classification per series per layer:**
```
fva > 0.02  → "ADDS_VALUE"      (layer improves accuracy by >2pp)
fva ∈ [-0.02, 0.02] → "NEUTRAL" (layer roughly same as parent)
fva < -0.02 → "DESTROYS_VALUE"  (layer makes forecast worse)
```

**Output tables:**

1. **fva_detail** — one row per (lob, series_id, target_week, fold, layer):
```
lob, series_id, target_week, fold, layer, actual, forecast,
parent_forecast, wmape, parent_wmape, fva_wmape, bias, parent_bias,
fva_bias, fva_class, sparse_class
```

2. **fva_summary** — one row per (lob, layer):
```
lob, layer, n_series, mean_wmape, mean_fva_wmape,
pct_adds_value, pct_neutral, pct_destroys_value,
mean_bias, total_volume_weighted_fva
```

3. **fva_leaderboard** — layers ranked by aggregate FVA contribution:
```
lob, layer, rank, cumulative_wmape_reduction, robustness_score,
recommendation (e.g., "Keep", "Review", "Remove")
```

**Integration with BacktestEngine:**
- Add `run_fva=True` option to `BacktestEngine.run()`
- When enabled, runs the 3 base layers (naive, best-stat, best-ML) for every fold
- Stores intermediate forecasts with `forecast_layer` tag
- FVA metrics computed after all folds complete

**Integration with BIExporter:**
- Add `export_fva()` method
- Writes `fva_summary/lob=X/` and `fva_detail/lob=X/` Parquet tables
- Power BI can build waterfall charts showing layer contribution

### Implementation Order
1. FVA computation engine (`src/metrics/fva.py`) — pure functions, no dependencies
2. FVA analyzer (`src/analytics/fva_analyzer.py`) — aggregation logic
3. BacktestEngine enhancement — cascade mode
4. MetricStore schema extension — optional columns
5. BIExporter integration — FVA export tables
6. Tests

### Risks
- Cascade backtest is ~3x slower (runs 3 models per fold instead of 1) — make it opt-in
- Sparse series routing complicates layer comparison — use sparse_class to segment
- Override layer (L4) only meaningful if planner overrides exist — handle L4=L3 case gracefully
- Zero actuals make WMAPE undefined — fall back to MAE for those periods

---

## Summary — Implementation Sequence

| Order | Feature | Estimated Scope | Key Deliverable |
|-------|---------|-----------------|-----------------|
| 1     | External Regressors | ~7 files modified, 2 new files | ML models use promos/holidays/price |
| 2     | RBAC + Audit Trail | ~4 files modified, 7 new files | Auth-protected API, immutable audit log |
| 3     | FVA Analysis | ~3 files modified, 3 new files | Layer-by-layer accuracy attribution |

Each feature is independent — no cross-dependencies between the three. They can be implemented in any order, but the sequence above maximizes value delivery: accuracy first, then governance, then measurement.
