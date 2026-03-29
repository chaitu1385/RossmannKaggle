# Future Ideas

This document captures potential enhancements, features, and workflows that could be implemented in the forecasting platform.

---

## 1. Key User Scenarios — End-to-End "Wow" Workflows

Define and validate the highest-impact user journeys that make demand planners and data scientists say "this is game-changing." These scenarios represent the product's core value propositions.

### Planner Scenarios

#### P1: "Zero-to-Forecast in 10 Minutes"
**Pain today:** Getting a first forecast takes weeks of data wrangling, model selection, and IT requests.

**Wow moment:** A planner drops 3 CSV files (sales history, product hierarchy, promotions calendar) onto the data onboarding page. The platform auto-classifies each file's role, detects join keys, merges them, runs quality checks, scores forecastability, and generates a recommended YAML config. They click "Run Forecast" and get a production-quality forecast with prediction intervals.

**Endpoints:** `POST /pipeline/analyze-multi-file` → `POST /analyze` → `POST /pipeline/forecast`
**Frontend:** `/data-onboarding` → `/forecast`

#### P2: "Why Did This SKU's Forecast Change?"
**Pain today:** Planners get a new forecast number with no explanation. They lose trust and override everything.

**Wow moment:** A planner sees a SKU's forecast jumped 30%. They click on it and see the STL decomposition (trend rising + seasonal peak approaching). They type "Why did this forecast increase?" in the AI panel and get a narrative explanation citing the detected structural break, seasonal peak, and the champion model's use of a promotional regressor.

**Endpoints:** `POST /forecast/decompose`, `POST /ai/explain`, `POST /series/breaks`
**Frontend:** `/forecast` → AI query panel + decomposition panel

#### P3: "S&OP Meeting Prep in 5 Minutes"
**Pain today:** Planners spend 2 days manually building PowerPoint decks with forecast summaries, exception callouts, and accuracy trends.

**Wow moment:** The AI commentary engine generates an executive summary with accuracy trends, exception callouts, and action items. The planner exports forecast-vs-actual and bias reports as Parquet files that auto-populate their Power BI dashboard.

**Endpoints:** `POST /ai/commentary`, `GET /metrics/{lob}/fva`, `POST /governance/export/{report_type}`
**Frontend:** `/sop`

#### P4: "Managing a Product Transition Without Losing History"
**Pain today:** When an old SKU is replaced by a new one, the new product has zero history. Planners manually estimate demand in spreadsheets for months.

**Wow moment:** A planner creates an override mapping old → new SKU with a linear ramp over 8 weeks. The platform automatically detects the transition scenario, stitches historical demand, applies the ramp-down/ramp-up curve, and produces a blended forecast. The override is tracked with approval status for audit.

**Endpoints:** `POST /overrides`, `POST /sku-mapping/phase1`, `POST /sku-mapping/phase2`
**Frontend:** `/sku-transitions`

#### P5: "Constrained Demand Planning"
**Pain today:** Forecasts ignore real-world constraints (warehouse capacity, budget limits). Planners manually clip numbers in Excel.

**Wow moment:** A planner applies a max capacity constraint across their region and per-SKU minimums. The platform redistributes the excess proportionally across SKUs while respecting the aggregate cap, showing a before/after comparison.

**Endpoints:** `POST /forecast/constrain`
**Frontend:** `/forecast` → constrained forecast panel

### Data Scientist Scenarios

#### DS1: "Backtest → Champion Selection → Production in One Session"
**Pain today:** Model comparison requires custom notebooks, manual metric aggregation, and a separate deployment process.

**Wow moment:** A data scientist configures 8 models in YAML, runs a 5-fold walk-forward backtest. The platform automatically routes sparse/intermittent series to specialized models and normal series to the full model pool. They see a leaderboard (WMAPE), FVA cascade, calibration plots, and SHAP importance. The champion is auto-selected and immediately available for production.

**Endpoints:** `POST /pipeline/backtest`, `GET /metrics/{lob}/fva`, `GET /metrics/{lob}/calibration`, `POST /metrics/{lob}/shap`
**Frontend:** `/backtest`

#### DS2: "AI Tells Me How to Improve My Config"
**Pain today:** Hyperparameter tuning is manual trial-and-error across dozens of config knobs.

**Wow moment:** After a backtest, Claude analyzes the leaderboard, FVA results, and data characteristics, then returns specific YAML config changes with the exact field path, current value, suggested value, expected impact, and risk level.

**Endpoints:** `POST /ai/recommend-config`
**Frontend:** `/backtest` → AI config tuner panel

#### DS3: "Multi-Frequency Forecasting Across Business Units"
**Pain today:** Different BUs need daily vs weekly vs monthly forecasts, requiring separate codebases.

**Wow moment:** The same pipeline runs for retail (weekly), e-commerce (daily), and wholesale (monthly) by simply changing `forecast.frequency` in each LOB's config. The platform automatically adjusts seasonal cycles, lag structures, validation windows, and model hyperparameters via `FREQUENCY_PROFILES`.

**Config:** `configs/lob/` per-LOB overrides with different `frequency` settings

#### DS4: "Hierarchical Reconciliation That Actually Works"
**Pain today:** Bottom-up forecasts don't add up to top-down targets. Planners lose trust.

**Wow moment:** A 4-level hierarchy (National → Region → Category → SKU) is built from data. After SKU-level forecasting, MinT reconciliation adjusts all levels simultaneously so they're mathematically coherent. Side-by-side comparison of bottom-up vs MinT vs WLS with before/after totals.

**Endpoints:** `POST /hierarchy/build`, `POST /hierarchy/reconcile`
**Frontend:** `/hierarchy` with sunburst visualization

#### DS5: "Drift Detection and Proactive Alerting"
**Pain today:** Model degradation goes unnoticed for weeks until planners complain about bad numbers.

**Wow moment:** The platform continuously compares live WMAPE against backtest baselines. When a SKU-group drifts above 1.5x baseline, an alert fires to Slack. The AI triage engine ranks all active alerts by business impact and suggests remediation actions.

**Endpoints:** `GET /metrics/drift/{lob}`, `POST /ai/triage`
**Frontend:** `/health` → drift histogram + AI triage panel

#### DS6: "Causal Analytics for Pricing & Promotions"
**Pain today:** Forecasts don't account for price changes or promotional cannibalization. Separate econometric models are needed.

**Wow moment:** The causal analytics module estimates price elasticity per SKU, detects cannibalization between products (post-detrending correlation), and estimates promotional lift. These insights feed directly into the ML models as validated regressors.

**Code:** `src/analytics/causal.py` — `PriceElasticityEstimator`, `CannibalizationDetector`, `PromotionalLiftEstimator`

### Cross-Persona Scenarios

#### X1: "Full Audit Trail for Compliance"
**Wow moment:** During an audit, the team can trace any forecast number back to: which model produced it (lineage), what config was used (config hash on model card), what data it was trained on (training window), what overrides were applied (override store with approval status), and who approved it. All stored in append-only Parquet.

**Endpoints:** `GET /governance/lineage`, `GET /governance/model-cards`, `GET /audit`

#### X2: "Cost Visibility Per Forecast Run"
**Wow moment:** The platform tracks compute cost per model per series. A data scientist sees that neural models cost 15x more than statistical ones but only improve WMAPE by 0.3pp. They remove the expensive model from config and cut pipeline runtime by 60%.

**Endpoints:** `GET /pipeline/costs`
**Frontend:** `/health` → cost tracking panel

### Implementation Approach
To validate these scenarios:
1. **Integration tests:** Create `test_user_scenarios.py` with one test class per scenario exercising the real API endpoints (AI endpoints mocked)
2. **End-to-end demo:** Upload sample CSV via `/data-onboarding`, run backtest, view results, generate AI commentary
3. **API smoke tests:** Hit each endpoint with sample payloads via Swagger UI at `/docs`
