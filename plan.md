# Rossmann Notebook Implementation Plan

## Goal
Create `notebooks/rossmann_platform_demo.ipynb` — a second notebook that proves the platform **thinks** (not just runs) by demonstrating 4 hard capabilities on real Rossmann Store Sales data.

---

## Data Strategy

**Source**: Rossmann Store Sales dataset (Kaggle). The platform already has `SparkDataLoader.read_rossmann_*()` but those require PySpark. Instead, we'll download CSVs directly in the notebook (or instruct the user to place them) and load with Polars.

**Scope**: 1,115 stores × ~2.5 years of daily sales. We'll **aggregate to weekly** to match the platform's weekly grain, and **subsample ~50 stores** (stratified by StoreType) to keep notebook runtime under 10 minutes.

**Key columns used**:
- `Store` (int) — series ID
- `Date` → aggregated to `week` (Monday of each week)
- `Sales` → weekly sum → `quantity`
- `Open` — filter out closed days before aggregation
- `Promo` — external regressor (weekly promo-day ratio)
- `StoreType` (from store.csv) — hierarchy level (a/b/c/d)

---

## Notebook Structure (8 sections, ~35-40 cells)

### Section 1: Setup & Data Loading (4 cells)
- Imports (polars, matplotlib, platform modules)
- Load `train.csv` + `store.csv` with Polars
- Filter: `Open == 1`, drop `Sales == 0` rows
- Join store metadata (StoreType, Assortment)
- Show raw data shape & date range

### Section 2: Weekly Aggregation & Subsample (3 cells)
- Add `week` column (Monday of each week): `pl.col("Date").dt.truncate("1w")`
- Aggregate daily → weekly: sum Sales, mean Promo (ratio of promo days per week)
- Stratified subsample: ~12 stores per StoreType (48-50 total) for tractable runtime
- Rename: `Sales` → `quantity`, `Store` → `series_id`
- Validate: print store count by type, date range, total rows
- Plot: 4-panel grid showing 1 representative store per StoreType

### Section 3: Hierarchy — Build & Visualize (3 cells)
**Capability: StoreType → Store hierarchy with MinT reconciliation**
- Build `HierarchyConfig(name="store_hierarchy", levels=["store_type", "store"], id_column="store")`
- Construct `HierarchyTree(config, data)`
- Print tree stats: 4 StoreType nodes → N store leaves
- Build & display summing matrix S (show shape, a snippet)
- Visualize: bar chart of store counts per type

### Section 4: Hierarchy — Forecast & Reconcile (5 cells)
**Prove: reconciled forecasts beat unreconciled**
- Split data: train (first ~2 years), holdout (last 13 weeks)
- Fit `SeasonalNaiveForecaster` on each store series
- Predict 13-week horizon
- **Unreconciled WMAPE**: compute per-store WMAPE, aggregate to StoreType level by summing forecasts vs summing actuals → store-type WMAPE
- **Reconciled**: use `Reconciler(method="mint")` to reconcile store-level forecasts. Re-aggregate → store-type WMAPE
- Also try OLS for comparison
- **Key output**: table showing `[StoreType, WMAPE_unreconciled, WMAPE_OLS, WMAPE_MinT]`
- Plot: grouped bar chart of WMAPE by StoreType × method
- Coherence check: verify `sum(store forecasts) == store_type forecast` after reconciliation

### Section 5: External Regressors — Promo Impact (5 cells)
**Prove: promos improve (or honestly don't improve) WMAPE**
- Prepare promo feature: weekly promo ratio already computed in Section 2
- **Backtest WITHOUT promos**:
  - Use `BacktestEngine` with `LGBMDirectForecaster` (or manual walk-forward if simpler)
  - 2-fold walk-forward, val_weeks=13
  - Collect per-store WMAPE
- **Backtest WITH promos**:
  - Pass promo column as additional feature in the training DataFrame
  - LightGBM picks it up automatically via `_external_feature_cols` detection in `_fit_mlforecast()`
  - Same 2-fold setup
- **Key output**:
  - Table: `[model, mean_wmape, median_wmape, delta]`
  - Per-store scatter: WMAPE_with_promo vs WMAPE_without_promo (45-degree line = no difference)
  - Histogram of per-store WMAPE improvement
- Honest interpretation: if promos don't help at weekly level, explain why (promos may be too correlated with day-of-week patterns that lags already capture)

### Section 6: FVA Cascade — Naive → Statistical → ML (6 cells)
**Prove: which forecasting layer adds value and which doesn't**
- Use the same train/holdout split (or 2-fold backtest)
- Run 3 model layers on all stores:
  1. **Naive**: `SeasonalNaiveForecaster(season_length=52)`
  2. **Statistical**: `AutoETSForecaster(season_length=52)` (or AutoARIMA)
  3. **ML**: `LGBMDirectForecaster()`
- For each store × layer, compute WMAPE on holdout
- Use `compute_fva_cascade()` from `src/metrics/fva.py`:
  - Pass `{"naive": naive_preds, "statistical": stat_preds, "ml": ml_preds}` as Series per store
  - Get FVA classification per layer transition
- Aggregate with `FVAAnalyzer.compute_fva_detail()` or manual aggregation
- **Key outputs**:
  1. FVA waterfall table: `[layer, mean_wmape, fva_wmape, fva_class, pct_adds_value]`
  2. Waterfall bar chart (the chart that "makes supply chain people pay attention")
  3. Layer leaderboard with Keep/Review/Remove recommendations
  4. Per-store FVA distribution (violin or box plot per layer)

### Section 7: Sparse Demand Detection & Routing (5 cells)
**Prove: sparse detector routes irregular stores to Croston/TSB, and it matters**
- Some Rossmann stores have frequent closures (StateHoliday, SchoolHoliday) creating zero-demand periods
- To amplify sparsity signal: include stores where `Open == 0` days are kept as zero-sales weeks (re-aggregate a subset WITHOUT the Open filter)
- Run `SparseDetector().classify()` on all stores
- Show SBC classification matrix: count stores by demand_class (smooth/erratic/intermittent/lumpy)
- **Prove it matters**: for sparse-classified stores, compare:
  - Standard model (SeasonalNaive or LightGBM) WMAPE
  - Croston/CrostonSBA/TSB WMAPE
- **Key outputs**:
  1. SBC classification heatmap (ADI vs CV² with store dots)
  2. Table: `[demand_class, n_stores, best_standard_wmape, best_sparse_wmape, delta]`
  3. Box plot: WMAPE by model type for sparse vs dense stores
  4. If sparse stores exist and Croston/TSB wins → "platform correctly routes these stores"
  5. If no stores are genuinely sparse → honest statement + synthetic demonstration fallback

### Section 8: Summary & Conclusions (2 cells)
- Executive summary table: one row per capability, key metric, verdict
- Narrative: "The platform doesn't just run models — it makes supply chain decisions"
- What each capability proves:
  - Hierarchy → coherent forecasts across org levels
  - Regressors → leverages external signals when they help
  - FVA → eliminates layers that don't add value
  - Sparse routing → right model for the right demand pattern

---

## Technical Decisions

1. **Subsample size**: ~50 stores (12 per StoreType). Enough for statistical validity, fast enough for notebook.
2. **Season length**: 52 weeks (yearly seasonality for weekly retail).
3. **Holdout**: Last 13 weeks (~1 quarter). Standard for S&OP.
4. **Backtest folds**: 2 (to keep runtime manageable; 3 would be better but slow).
5. **Reconciliation**: MinT primary, OLS as comparison. WLS skipped for brevity.
6. **ML model**: LightGBM only (faster than XGBoost, same interface).
7. **Statistical model**: AutoETS (faster than AutoARIMA for 50 series).

## Data Acquisition
The notebook will include a cell that checks for `data/rossmann/train.csv` and `data/rossmann/store.csv`. If missing, it prints download instructions (Kaggle CLI command). We won't auto-download to avoid credential issues.

## Runtime Estimate
- Weekly aggregation + subsample: ~5s
- Naive fit+predict 50 stores: ~2s
- AutoETS fit+predict 50 stores: ~30-60s
- LightGBM fit+predict 50 stores: ~15-30s
- Hierarchy reconciliation: ~2s
- Total: **~2-3 minutes** with 50 stores

## File Output
- `notebooks/rossmann_platform_demo.ipynb` — the notebook
- No other files created
