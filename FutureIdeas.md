# Future Ideas

This document captures potential enhancements, features, and workflows that could be implemented in the forecasting platform.

---

## 1. Key User Scenarios ΓÇö End-to-End "Wow" Workflows

Define and validate the highest-impact user journeys that make demand planners and data scientists say "this is game-changing." These scenarios represent the product's core value propositions.

### Planner Scenarios

#### P1: "Zero-to-Forecast in 10 Minutes"
**Pain today:** Getting a first forecast takes weeks of data wrangling, model selection, and IT requests.

**Wow moment:** A planner drops 3 CSV files (sales history, product hierarchy, promotions calendar) onto the data onboarding page. The platform auto-classifies each file's role, detects join keys, merges them, runs quality checks, scores forecastability, and generates a recommended YAML config. They click "Run Forecast" and get a production-quality forecast with prediction intervals.

**Endpoints:** `POST /pipeline/analyze-multi-file` ΓåÆ `POST /analyze` ΓåÆ `POST /pipeline/forecast`
**Frontend:** `/data-onboarding` ΓåÆ `/forecast`

#### P2: "Why Did This SKU's Forecast Change?"
**Pain today:** Planners get a new forecast number with no explanation. They lose trust and override everything.

**Wow moment:** A planner sees a SKU's forecast jumped 30%. They click on it and see the STL decomposition (trend rising + seasonal peak approaching). They type "Why did this forecast increase?" in the AI panel and get a narrative explanation citing the detected structural break, seasonal peak, and the champion model's use of a promotional regressor.

**Endpoints:** `POST /forecast/decompose`, `POST /ai/explain`, `POST /series/breaks`
**Frontend:** `/forecast` ΓåÆ AI query panel + decomposition panel

#### P3: "S&OP Meeting Prep in 5 Minutes"
**Pain today:** Planners spend 2 days manually building PowerPoint decks with forecast summaries, exception callouts, and accuracy trends.

**Wow moment:** The AI commentary engine generates an executive summary with accuracy trends, exception callouts, and action items. The planner exports forecast-vs-actual and bias reports as Parquet files that auto-populate their Power BI dashboard.

**Endpoints:** `POST /ai/commentary`, `GET /metrics/{lob}/fva`, `POST /governance/export/{report_type}`
**Frontend:** `/sop`

#### P4: "Managing a Product Transition Without Losing History"
**Pain today:** When an old SKU is replaced by a new one, the new product has zero history. Planners manually estimate demand in spreadsheets for months.

**Wow moment:** A planner creates an override mapping old ΓåÆ new SKU with a linear ramp over 8 weeks. The platform automatically detects the transition scenario, stitches historical demand, applies the ramp-down/ramp-up curve, and produces a blended forecast. The override is tracked with approval status for audit.

**Endpoints:** `POST /overrides`, `POST /sku-mapping/phase1`, `POST /sku-mapping/phase2`
**Frontend:** `/sku-transitions`

#### P5: "Constrained Demand Planning"
**Pain today:** Forecasts ignore real-world constraints (warehouse capacity, budget limits). Planners manually clip numbers in Excel.

**Wow moment:** A planner applies a max capacity constraint across their region and per-SKU minimums. The platform redistributes the excess proportionally across SKUs while respecting the aggregate cap, showing a before/after comparison.

**Endpoints:** `POST /forecast/constrain`
**Frontend:** `/forecast` ΓåÆ constrained forecast panel

### Data Scientist Scenarios

#### DS1: "Backtest ΓåÆ Champion Selection ΓåÆ Production in One Session"
**Pain today:** Model comparison requires custom notebooks, manual metric aggregation, and a separate deployment process.

**Wow moment:** A data scientist configures 8 models in YAML, runs a 5-fold walk-forward backtest. The platform automatically routes sparse/intermittent series to specialized models and normal series to the full model pool. They see a leaderboard (WMAPE), FVA cascade, calibration plots, and SHAP importance. The champion is auto-selected and immediately available for production.

**Endpoints:** `POST /pipeline/backtest`, `GET /metrics/{lob}/fva`, `GET /metrics/{lob}/calibration`, `POST /metrics/{lob}/shap`
**Frontend:** `/backtest`

#### DS2: "AI Tells Me How to Improve My Config"
**Pain today:** Hyperparameter tuning is manual trial-and-error across dozens of config knobs.

**Wow moment:** After a backtest, Claude analyzes the leaderboard, FVA results, and data characteristics, then returns specific YAML config changes with the exact field path, current value, suggested value, expected impact, and risk level.

**Endpoints:** `POST /ai/recommend-config`
**Frontend:** `/backtest` ΓåÆ AI config tuner panel

#### DS3: "Multi-Frequency Forecasting Across Business Units"
**Pain today:** Different BUs need daily vs weekly vs monthly forecasts, requiring separate codebases.

**Wow moment:** The same pipeline runs for retail (weekly), e-commerce (daily), and wholesale (monthly) by simply changing `forecast.frequency` in each LOB's config. The platform automatically adjusts seasonal cycles, lag structures, validation windows, and model hyperparameters via `FREQUENCY_PROFILES`.

**Config:** `configs/lob/` per-LOB overrides with different `frequency` settings

#### DS4: "Hierarchical Reconciliation That Actually Works"
**Pain today:** Bottom-up forecasts don't add up to top-down targets. Planners lose trust.

**Wow moment:** A 4-level hierarchy (National ΓåÆ Region ΓåÆ Category ΓåÆ SKU) is built from data. After SKU-level forecasting, MinT reconciliation adjusts all levels simultaneously so they're mathematically coherent. Side-by-side comparison of bottom-up vs MinT vs WLS with before/after totals.

**Endpoints:** `POST /hierarchy/build`, `POST /hierarchy/reconcile`
**Frontend:** `/hierarchy` with sunburst visualization

#### DS5: "Drift Detection and Proactive Alerting"
**Pain today:** Model degradation goes unnoticed for weeks until planners complain about bad numbers.

**Wow moment:** The platform continuously compares live WMAPE against backtest baselines. When a SKU-group drifts above 1.5x baseline, an alert fires to Slack. The AI triage engine ranks all active alerts by business impact and suggests remediation actions.

**Endpoints:** `GET /metrics/drift/{lob}`, `POST /ai/triage`
**Frontend:** `/health` ΓåÆ drift histogram + AI triage panel

#### DS6: "Causal Analytics for Pricing & Promotions"
**Pain today:** Forecasts don't account for price changes or promotional cannibalization. Separate econometric models are needed.

**Wow moment:** The causal analytics module estimates price elasticity per SKU, detects cannibalization between products (post-detrending correlation), and estimates promotional lift. These insights feed directly into the ML models as validated regressors.

**Code:** `src/analytics/causal.py` ΓÇö `PriceElasticityEstimator`, `CannibalizationDetector`, `PromotionalLiftEstimator`

### Cross-Persona Scenarios

#### X1: "Full Audit Trail for Compliance"
**Wow moment:** During an audit, the team can trace any forecast number back to: which model produced it (lineage), what config was used (config hash on model card), what data it was trained on (training window), what overrides were applied (override store with approval status), and who approved it. All stored in append-only Parquet.

**Endpoints:** `GET /governance/lineage`, `GET /governance/model-cards`, `GET /audit`

#### X2: "Cost Visibility Per Forecast Run"
**Wow moment:** The platform tracks compute cost per model per series. A data scientist sees that neural models cost 15x more than statistical ones but only improve WMAPE by 0.3pp. They remove the expensive model from config and cut pipeline runtime by 60%.

**Endpoints:** `GET /pipeline/costs`
**Frontend:** `/health` ΓåÆ cost tracking panel

### Implementation Approach
To validate these scenarios:
1. **Integration tests:** Create `test_user_scenarios.py` with one test class per scenario exercising the real API endpoints (AI endpoints mocked)
2. **End-to-end demo:** Upload sample CSV via `/data-onboarding`, run backtest, view results, generate AI commentary
3. **API smoke tests:** Hit each endpoint with sample payloads via Swagger UI at `/docs`

---

# AI-Native Supply Chain Product Roadmap

## Context

This platform is a production-ready multi-frequency sales forecasting product with 16 algorithms, 4 Claude-powered AI engines, hierarchical reconciliation, and full governance. The competitive landscape has shifted dramatically: **o9 Solutions** deploys Enterprise Knowledge Graphs with GenAI Composite Agents, **Auger** (founded 2024, ex-Amazon leadership, $100M funding) is building an "AI-native from the ground up" Supply Chain Autonomy Platform on Microsoft Fabric, **Blue Yonder** launched 5 domain-specific AI agents with a Supply Chain Knowledge Graph, **Kinaxis** fuses heuristics+optimization+ML automatically, and **Anaplan** offers role-based AI agents with a conversational CoPlanner.

The common thread: **the industry is moving from AI-assisted forecasting to agentic autonomous planning**. This plan adds 7 major capabilities across 5 phases to position the platform competitively.

---

## Phase 1: Agentic Supply Chain Copilot (Highest Impact)

**Why**: Every competitor (o9, Anaplan, AWS) now has a conversational planning copilot. Our NL Query engine only answers questions about individual series. A copilot that can reason across the entire platform ΓÇö running analyses, comparing scenarios, triggering pipelines ΓÇö is table stakes.

### 1A. Multi-Turn Planning Copilot

**What**: An agentic Claude-powered copilot that can hold multi-turn conversations, call platform APIs as tools, and execute multi-step analytical workflows autonomously.

**Key Components**:
- `src/ai/copilot.py` ΓÇö `PlanningCopilot` class with:
  - Tool registry mapping natural language intents to platform API calls
  - Conversation memory (multi-turn context)
  - Plan-then-execute pattern: Claude creates a plan ΓåÆ user approves ΓåÆ copilot executes steps
  - Tools available to Claude: run_backtest, get_forecast, compare_models, explain_series, get_drift_alerts, triage_anomalies, get_fva, apply_override, get_hierarchy, run_decomposition
- `src/ai/tools.py` ΓÇö Tool definitions formatted for Claude tool_use (wrapping existing API logic from `src/api/routers/`)
- Streaming responses via SSE for real-time feedback

**API Endpoints**:
- `POST /ai/copilot/chat` ΓÇö Send message, get streaming response with tool calls
- `GET /ai/copilot/sessions` ΓÇö List active sessions
- `DELETE /ai/copilot/sessions/{id}` ΓÇö End session

**Frontend**:
- `CopilotPanel` ΓÇö Persistent side panel with chat interface, tool execution visibility, approval prompts
- Inline references to charts/data that copilot is discussing

**Extends**: `src/ai/base.py` (AIFeatureBase), reuses all existing AI engines as callable tools

### 1B. Role-Based Personas (inspired by Anaplan)

**What**: Pre-configured copilot personas that tailor behavior, tool access, and communication style to the user's role.

**Personas**:
| Persona | Role | Focus | Available Tools |
|---------|------|-------|-----------------|
| Demand Planner | Analyst | Series-level accuracy, overrides, anomalies | All forecast + override tools |
| S&OP Executive | Manager | Portfolio view, executive summaries, FVA | Commentary, FVA, drift, hierarchy |
| Data Scientist | Admin | Model performance, config tuning, backtests | All tools including config + backtest |
| Supply Planner | Analyst | Constrained forecasts, capacity, inventory | Constrained, hierarchy, overrides |

**Implementation**: Persona config in `src/ai/personas.py` ΓÇö system prompt templates + tool allowlists per role, integrates with existing RBAC in `src/auth/`

---

## Phase 2: Scenario Planning & What-If Engine

**Why**: Google Cloud Supply Chain Twin, o9, and Coupa/LLamasoft all offer what-if simulation. This is the #1 gap ΓÇö planners need to model "what happens if demand drops 20%" or "what if we lose supplier X" before it happens.

### 2A. Scenario Engine

**What**: Create, compare, and manage named forecast scenarios with parameter overrides.

**Key Components**:
- `src/scenarios/engine.py` ΓÇö `ScenarioEngine` class:
  - Create scenario from base config with parameter overrides (demand shocks, model changes, constraint changes)
  - Run scenario (triggers backtest/forecast pipeline with modified config)
  - Compare scenarios side-by-side (metrics, forecasts, costs)
  - Persist scenarios to Parquet (reusable, shareable)
- `src/scenarios/types.py` ΓÇö `Scenario` dataclass: name, description, base_config, overrides (dict of config path ΓåÆ value), demand_adjustments (series-level multipliers/offsets), constraint_overrides
- `src/scenarios/comparator.py` ΓÇö `ScenarioComparator`: diff two scenario outputs (forecast deltas, metric deltas, cost impact)

**Scenario Types**:
1. **Demand shock**: Apply multiplier/offset to specific series/categories/regions
2. **Model swap**: Run same data through different model configurations
3. **Constraint change**: Modify capacity/budget constraints
4. **Regressor shift**: Modify external regressor assumptions (price changes, promo changes)
5. **Hierarchy change**: Add/remove nodes, test different aggregation levels

**API Endpoints**:
- `POST /scenarios` ΓÇö Create scenario
- `POST /scenarios/{id}/run` ΓÇö Execute scenario
- `GET /scenarios/{id}/results` ΓÇö Get results
- `POST /scenarios/compare` ΓÇö Compare 2+ scenarios
- `GET /scenarios` ΓÇö List all scenarios

**Frontend**:
- `ScenarioBuilder` ΓÇö Visual scenario configuration with parameter sliders
- `ScenarioComparison` ΓÇö Side-by-side charts and metric tables
- `ScenarioLibrary` ΓÇö Browse/clone/share saved scenarios

**AI Integration**: Copilot can create and compare scenarios via natural language: "What if holiday demand is 15% higher than last year?"

### 2B. Impact Simulation (inspired by Google Supply Chain Twin)

**What**: Cascading impact analysis ΓÇö when one variable changes, show the ripple effects through the hierarchy.

- `src/scenarios/impact.py` ΓÇö `ImpactSimulator`: Given a demand change at leaf level, propagate through hierarchy using reconciliation, show effects on parent nodes, constrained plans, and cost estimates
- Visualized as a Sankey diagram or hierarchy diff in frontend

---

## Phase 3: Autonomous Planning Agents

**Why**: Auger's core value prop is "autonomous execution in seconds." Blue Yonder has 5 specialized agents. This is the frontier ΓÇö agents that monitor, detect, decide, and act within guardrails.

### 3A. Agent Framework

**What**: A lightweight agent framework where specialized agents continuously monitor metrics and take autonomous actions within configurable guardrails.

**Key Components**:
- `src/agents/base.py` ΓÇö `SupplyChainAgent` ABC:
  - `monitor()` ΓÇö Check conditions (called on schedule)
  - `evaluate()` ΓÇö Decide if action needed (returns ActionProposal)
  - `execute()` ΓÇö Take action (with guardrail checks)
  - `report()` ΓÇö Log what happened to audit trail
- `src/agents/guardrails.py` ΓÇö `GuardrailEngine`:
  - Configurable thresholds per agent (max forecast change %, max overrides per day, etc.)
  - Approval routing: auto-approve within guardrails, escalate to human outside
  - Full audit trail of all agent decisions via existing `src/audit/`
- `src/agents/scheduler.py` ΓÇö Extends existing `PipelineScheduler` for agent loops

### 3B. Specialized Agents

| Agent | Monitors | Actions | Guardrails |
|-------|----------|---------|------------|
| **DriftResponseAgent** | `src/metrics/` drift alerts | Re-triggers backtest, swaps champion model, notifies planner | Max 1 model swap per series per week |
| **OverrideDecayAgent** | `src/overrides/` age | Expires stale overrides, proposes fresh adjustments based on actuals | Only expires overrides older than N cycles |
| **DataQualityAgent** | `src/data/validator.py` signals | Auto-cleanses new data, flags anomalies, adjusts cleansing params | Cannot delete data, only flag/impute |
| **RebalanceAgent** | Constrained forecast vs actuals | Adjusts capacity allocations, redistributes budget across hierarchy | Max N% reallocation per cycle |
| **NewProductAgent** | `src/sku_mapping/` transitions | Auto-maps new SKUs using similarity, bootstraps initial forecasts | Requires human approval for high-revenue SKUs |

**API Endpoints**:
- `GET /agents` ΓÇö List agents and their status
- `POST /agents/{id}/enable` / `POST /agents/{id}/disable`
- `GET /agents/{id}/history` ΓÇö Action history
- `PUT /agents/{id}/guardrails` ΓÇö Configure guardrails
- `POST /agents/{id}/approve/{action_id}` ΓÇö Approve pending action

**Frontend**:
- `AgentDashboard` ΓÇö Agent status, recent actions, pending approvals
- `GuardrailConfig` ΓÇö Visual guardrail editor per agent
- `AgentTimeline` ΓÇö Chronological view of all agent actions with audit trail

**AI Integration**: Each agent uses Claude for reasoning about edge cases ΓÇö e.g., DriftResponseAgent asks Claude "Should I swap this model given the recent holiday period?" before acting

---

## Phase 4: Supply Chain Knowledge Graph

**Why**: o9's Enterprise Knowledge Graph and Blue Yonder's Supply Chain Knowledge Graph are major differentiators. A knowledge graph connects entities (products, suppliers, customers, events, forecasts) enabling causal reasoning and cross-functional queries that flat tables cannot support.

### 4A. Entity-Relationship Graph

**What**: A lightweight knowledge graph built on NetworkX (no heavy graph DB dependency) that connects platform entities.

**Key Components**:
- `src/knowledge/graph.py` ΓÇö `SupplyChainGraph`:
  - Nodes: Series, Products, Categories, Regions, Suppliers, Customers, Events, Models, Forecasts
  - Edges: belongs_to, supplied_by, sold_to, affected_by, forecasted_by, similar_to
  - Auto-populated from hierarchy tree, SKU mappings, regressor metadata
  - Queryable: "What series are affected by Supplier X?" "What products share demand patterns?"
- `src/knowledge/builder.py` ΓÇö `GraphBuilder`: Constructs graph from existing data sources
  - Ingests hierarchy from `src/hierarchy/`
  - Ingests SKU relationships from `src/sku_mapping/`
  - Ingests regressor associations from `src/data/regressors.py`
  - Infers similarity edges from forecast correlation patterns

### 4B. Causal Reasoning Layer

**What**: Extend the existing `CausalAnalyzer` with graph-powered causal inference.

- `src/knowledge/causal.py` ΓÇö `CausalReasoningEngine`:
  - "Why did forecast change?" ΓåÆ Trace graph edges to find upstream causes (regressor shifts, supplier changes, seasonal events)
  - "What will be affected?" ΓåÆ Forward propagation through graph
  - Claude interprets causal chains into natural language narratives

**API Endpoints**:
- `GET /knowledge/graph/{lob}` ΓÇö Get graph structure
- `POST /knowledge/query` ΓÇö Natural language graph query
- `POST /knowledge/impact` ΓÇö "What-if" through graph (e.g., "What if Supplier X delays 2 weeks?")
- `GET /knowledge/similar/{series_id}` ΓÇö Find similar series via graph proximity

**Frontend**:
- `KnowledgeGraph` ΓÇö Interactive graph visualization (force-directed or hierarchical)
- `CausalChain` ΓÇö Visual cause-effect chain for forecast explanations
- Integrated into copilot: "Show me everything connected to SKU-1234"

---

## Phase 5: External Signal Integration & Demand Sensing

**Why**: Kinaxis, AWS, and o9 all fuse 100+ external signals (weather, POS, social media, news, economic indicators) for real-time demand sensing. This turns periodic forecasting into continuous adaptive planning.

### 5A. Signal Registry

**What**: A pluggable registry for external data signals that feed into demand sensing.

- `src/signals/registry.py` ΓÇö `SignalRegistry`:
  - Register signal sources with metadata (name, frequency, latency, coverage)
  - Built-in connectors: weather APIs, Google Trends, economic indicators (FRED), news sentiment
  - Each signal implements `SignalSource` ABC: `fetch(date_range, entities)` ΓåÆ Polars DataFrame
- `src/signals/connectors/` ΓÇö Individual connector modules:
  - `weather.py` ΓÇö OpenWeatherMap / Visual Crossing
  - `trends.py` ΓÇö Google Trends via pytrends
  - `economic.py` ΓÇö FRED API (CPI, unemployment, consumer sentiment)
  - `news.py` ΓÇö News API + Claude-powered sentiment scoring
  - `custom.py` ΓÇö User-uploaded signal files

### 5B. Demand Sensing Engine

**What**: Short-horizon demand adjustment using leading indicators.

- `src/signals/sensing.py` ΓÇö `DemandSensingEngine`:
  - Ingests latest signals from registry
  - Trains lightweight gradient-boosted adjustment model (extends `src/forecasting/ml.py`)
  - Outputs adjustment factors applied to base statistical/ML forecast
  - Updates continuously as new signals arrive (not just at reforecast time)

### 5C. AI-Powered Signal Interpretation

- Claude analyzes incoming signals and explains their demand implications
- "Breaking news about port strikes ΓåÆ Claude explains expected impact on Category X supply"
- Integrated with copilot and alert system

**API Endpoints**:
- `GET /signals` ΓÇö List registered signals
- `POST /signals/register` ΓÇö Register new signal source
- `POST /signals/sense` ΓÇö Run demand sensing for LOB
- `GET /signals/impact/{lob}` ΓÇö Signal attribution (which signals moved the forecast)

**Frontend**:
- `SignalDashboard` ΓÇö Signal health, latency, coverage
- `DemandSensing` ΓÇö Overlay base forecast with signal-adjusted forecast
- `SignalAttribution` ΓÇö Which external signals are driving adjustments

---

## Implementation Priority & Dependencies

```
Phase 1 (Copilot)          ΓåÉ Builds on existing src/ai/, highest ROI
  Γåô
Phase 2 (Scenarios)        ΓåÉ Builds on existing pipeline, config system
  Γåô
Phase 3 (Agents)           ΓåÉ Requires Phases 1+2 for full value
  Γåô
Phase 4 (Knowledge Graph)  ΓåÉ Requires hierarchy, SKU mapping, regressors
  Γåô
Phase 5 (Demand Sensing)   ΓåÉ Requires signal infrastructure, can start independently
```

Phases 1 and 2 can be built in parallel. Phase 5 signal connectors can start independently.

## Key Files to Extend

| Existing Module | Extension |
|----------------|-----------|
| `src/ai/base.py` | Add tool-use support for copilot |
| `src/ai/nl_query.py` | Upgrade to multi-turn with tool calling |
| `src/api/app.py` | Register new routers (copilot, scenarios, agents, knowledge, signals) |
| `src/config/schema.py` | Add `ScenarioConfig`, `AgentConfig`, `SignalConfig` dataclasses |
| `src/pipeline/backtest.py` | Accept scenario overrides |
| `src/pipeline/forecast.py` | Accept scenario overrides, signal adjustments |
| `src/hierarchy/tree.py` | Expose to knowledge graph builder |
| `src/sku_mapping/` | Expose similarity data to knowledge graph |
| `src/data/regressors.py` | Integrate with signal registry |
| `src/observability/alerts.py` | Agent action notifications |
| `src/audit/logger.py` | Agent decision audit trail |
| `src/auth/rbac.py` | Agent permissions, persona-based access |

## New Modules to Create

| Module | Purpose |
|--------|---------|
| `src/ai/copilot.py` | Multi-turn planning copilot |
| `src/ai/tools.py` | Tool definitions for Claude tool_use |
| `src/ai/personas.py` | Role-based persona configs |
| `src/scenarios/` | Scenario engine, comparator, impact simulator |
| `src/agents/` | Agent framework, specialized agents, guardrails |
| `src/knowledge/` | Knowledge graph, causal reasoning |
| `src/signals/` | Signal registry, demand sensing, connectors |
| `src/api/routers/copilot.py` | Copilot chat endpoints |
| `src/api/routers/scenarios.py` | Scenario management endpoints |
| `src/api/routers/agents.py` | Agent management endpoints |
| `src/api/routers/knowledge.py` | Knowledge graph endpoints |
| `src/api/routers/signals.py` | Signal endpoints |

## Competitive Positioning

| Capability | o9 | Auger | Blue Yonder | Kinaxis | **This Platform** |
|-----------|-----|-------|-------------|---------|-------------------|
| Statistical/ML/Neural Models | Yes | Yes | Yes | Yes | **16 algorithms** |
| Foundation Models | No | No | No | No | **Chronos + TimeGPT** |
| LLM Copilot | Composite Agents | TBD | Cognitive | No | **Phase 1** |
| Scenario Planning | Yes | Yes | Yes | Yes | **Phase 2** |
| Autonomous Agents | Limited | Core | 5 Agents | No | **Phase 3** |
| Knowledge Graph | EKG | No | SCKG | No | **Phase 4** |
| Demand Sensing | Yes | Yes | Yes | Demand.AI | **Phase 5** |
| Hierarchical Reconciliation | Limited | No | Yes | Yes | **6 methods (existing)** |
| Explainability (SHAP+STL+NL) | Limited | No | Limited | Visual | **Deep (existing)** |
| Open/Self-hosted | No | No | No | No | **Yes (differentiator)** |

**Unique differentiators vs. competitors**:
1. **Open/self-hosted** ΓÇö Every competitor is SaaS-only. This platform can be deployed on-prem or in customer's cloud
2. **Foundation model integration** ΓÇö Only platform with Chronos + TimeGPT zero-shot forecasting
3. **Deep explainability** ΓÇö SHAP + STL + FVA + Claude narratives is deeper than any competitor
4. **Multi-frequency native** ΓÇö D/W/M/Q from a single codebase with frequency profiles
5. **Claude-native AI** ΓÇö Deepest LLM integration (not just a chatbot layer) with domain-specific reasoning

## Competitive Research Sources

- **o9 Solutions**: Enterprise Knowledge Graph (EKG), GenAI Composite Agents, Leader in 2025 Gartner Magic Quadrant
- **Auger**: Founded 2024 by Dave Clark (ex-Amazon/Flexport), $100M from Oak HC/FT, built on Microsoft Fabric
- **Blue Yonder**: 5 AI Agents (Inventory/Shelf/Logistics/Warehouse/Network Ops), Supply Chain Knowledge Graph with Snowflake & RelationalAI
- **Kinaxis**: Planning.AI (heuristics+optimization+ML fusion), Demand.AI with 100+ external signals
- **Anaplan**: CoPlanner conversational AI, role-based AI agents (workforce analyst, model builder, issue detective)
- **RELEX Solutions**: Multi-technique AI (ML + optimization + heuristics + Monte Carlo probabilistic planning)
- **ToolsGroup**: Probabilistic forecasting with ML-refined distributions, 20-30% inventory cost reduction
- **Google Cloud**: Supply Chain Twin with what-if simulation and cascading impact analysis
- **AWS Supply Chain**: Amazon Q assistant, 30 years of Amazon demand sensing models, 200+ external signals
- **Coupa/LLamasoft**: Digital twin network modeling, $9.5T spend intelligence dataset

---

# Future Ideas

## 1. Knowledge Graph for Supply Chain Forecasting

### Context

The platform currently models supply chain entities (products, geographies, channels, SKUs) as **isolated hierarchical trees** and **Polars join-based relationships**. There is no unified graph representation connecting products to their regressors, transitions, channels, geographies, pipeline runs, or similar series. This limits the platform in several ways:

1. **Forecasting models treat each series independently** ΓÇö no cross-series information flow (e.g., sibling products, cannibalization, transitions)
2. **AI engines lack structural context** ΓÇö Claude answers questions about a series without knowing its graph neighborhood (what it transitions to, what drives it, what competes with it)
3. **No impact analysis** ΓÇö cannot answer "if we discontinue SKU-X, what series are affected?" without manual investigation
4. **No what-if scenarios** ΓÇö cannot propagate a demand change through the supply chain network

### Industry Research

- **o9 Solutions**: Enterprise Knowledge Graph (EKG) with Graph-Cube technology ΓÇö a graph-based digital twin connecting products, customers, suppliers, facilities across supply chain, finance, and commercial domains. Powers AI agents that traverse the graph for cross-functional queries. (Sense -> Model -> Simulate -> Decide -> Execute -> Learn cycle)
- **Auger** (Dave Clark, ex-Amazon): "Augentic ontology" that encodes constraints, rules, and relationships, normalizing disparate data sources into a single executable source of truth. Deep integration with Microsoft Fabric for analytics.
- **Academic research**: KG-GCN-LSTM models show ~50% of forecast accuracy improvement comes from the knowledge graph alone. GraphSAGE demonstrates that cross-series information flow improves forecasts without additional business rules.

### Ontology Design

#### Node Types

| Node Type | Source in Current Code | Key Attributes |
|---|---|---|
| `Product` | HierarchyTree leaf nodes (product dim) | sku_id, description, family, category, form_factor, price_tier, status, launch_date, eol_date |
| `ProductGroup` | HierarchyTree intermediate nodes | level, key |
| `Geography` | HierarchyTree (geography dim) | country, subregion, region |
| `Channel` | HierarchyTree (channel dim) | channel_name |
| `Series` | SeriesBuilder composite IDs | series_id, frequency, is_sparse |
| `Regressor` | ExternalRegressorConfig | name, type (known_ahead / contemporaneous) |
| `ForecastModel` | Model registry | model_name, model_family |
| `PipelineRun` | PipelineContext | run_id, timestamp, lob |
| `Supplier` | Future ingestion | supplier_id, name, location |
| `Customer` | Future ingestion | customer_id, segment |

#### Edge Types

| Edge Type | From -> To | Source | Temporal? |
|---|---|---|---|
| `child_of` | ProductGroup -> ProductGroup | HierarchyTree parent-child | No |
| `belongs_to` | Product -> ProductGroup | Hierarchy leaf membership | No |
| `sold_in` | Product -> Geography | Data grain | No |
| `via_channel` | Product -> Channel | Data grain | No |
| `generates` | Product -> Series | SeriesBuilder | No |
| `transitions_to` | Product -> Product | MappingRecord (old->new) | Yes |
| `cannibalizes` | Product -> Product | CausalAnalyzer | Yes |
| `driven_by` | Series -> Regressor | ExternalRegressorConfig | Yes |
| `forecast_by` | Series -> ForecastModel | Champion selection | Yes |
| `supplied_by` | Product -> Supplier | Future ingestion | Yes |
| `similar_to` | Series -> Series | Correlation/DTW | Yes |
| `produced_by` | Series -> PipelineRun | LineageTracker | No |

### Module Structure

All new code under `forecasting-product/src/knowledge_graph/`:

```
src/knowledge_graph/
    __init__.py          # Public API: KnowledgeGraph, build_graph
    ontology.py          # GraphNode, GraphEdge dataclasses, EntityType/RelationType enums
    graph.py             # KnowledgeGraph ΓÇö wraps NetworkX MultiDiGraph
    builder.py           # KnowledgeGraphBuilder ΓÇö populates from HierarchyTree, MappingRecords, etc.
    temporal.py          # TemporalEdgeStore ΓÇö edge validity intervals, point-in-time snapshots
    features.py          # GraphFeatureExtractor ΓÇö derives numeric features for forecasting models
    traversal.py         # GraphTraversal ΓÇö path finding, impact radius, neighbor queries
    similarity.py        # SeriesSimilarity ΓÇö correlation/DTW, builds similar_to edges
    scenarios.py         # ScenarioEngine ΓÇö what-if impact propagation
    ai_context.py        # GraphContextProvider ΓÇö structured context for AI engines
    config.py            # KnowledgeGraphConfig dataclass
```

### Key Design Decisions

1. **NetworkX (not Neo4j)** ΓÇö keeps the platform self-contained. Suitable for graphs up to ~100K nodes. The `graph.py` wrapper abstracts the backend so it can be swapped later if needed.

2. **Graph features injected via existing regressor path** ΓÇö `GraphFeatureExtractor` returns a Polars DataFrame with `series_id` column. This joins into `SeriesBuilder.build()` using the same mechanism as external regressors. LightGBM, XGBoost, and neural models all benefit with zero model code changes.

3. **Opt-in via config** ΓÇö `knowledge_graph.enabled = False` by default. Each sub-feature (feature extraction, similarity, temporal snapshots) has its own toggle.

4. **AI grounding before GNN training** ΓÇö immediate value comes from providing structured graph context to Claude via `AIFeatureBase` pattern (lower risk, faster value). GNN embeddings are Phase 4.

5. **Temporal edges via Polars** ΓÇö edge versions stored in a Polars DataFrame for efficient time-range filtering, consistent with the platform's Polars-native approach.

### Graph Features for Forecasting (the "KG-GCN-LSTM" approach)

Per-series features extracted from graph structure:

| Feature | Description | How Computed |
|---|---|---|
| `kg_degree` | Connectivity of the product node | Count of all edges (channels, geos, regressors) |
| `kg_transition_proximity` | Hops to nearest transitioning SKU | BFS from product node to any `transitions_to` edge |
| `kg_sibling_mean` | Mean demand of sibling series | Average demand of series sharing same parent ProductGroup |
| `kg_sibling_std` | Demand volatility across siblings | Std dev of sibling demands |
| `kg_cannibal_exposure` | Cannibalization pressure | Sum of `cannibalizes` edge weights |
| `kg_regressor_count` | Number of connected regressors | Count of `driven_by` edges |
| `kg_similar_series_mean` | Consensus from similar series | Mean demand from `similar_to` neighbors |

These feed into ML models as additional columns alongside existing regressors.

### Integration Points (existing files to modify)

| File | Change |
|---|---|
| `src/config/schema.py` | Add `KnowledgeGraphConfig` dataclass to `PlatformConfig` |
| `src/series/builder.py` | After external feature join, optionally join graph features from `GraphFeatureExtractor` |
| `src/ai/nl_query.py` | Modify `_gather_context()` to include graph neighborhood from `GraphContextProvider` |
| `src/ai/anomaly_triage.py` | Include graph impact context when triaging drift alerts |
| `src/ai/commentary.py` | Include transition/cannibalization context in commentary |
| `src/api/app.py` | Register new `knowledge_graph` router in `_register_routers()` |
| `src/api/schemas.py` | Add Pydantic models for graph API requests/responses |

### API Endpoints

New router: `src/api/routers/knowledge_graph.py`

| Method | Path | Description |
|---|---|---|
| POST | `/graph/build` | Build/rebuild graph from current platform state |
| GET | `/graph/stats` | Node/edge counts by type |
| GET | `/graph/node/{node_id}` | Node details + immediate neighbors |
| GET | `/graph/node/{node_id}/neighborhood` | N-hop subgraph |
| GET | `/graph/series/{series_id}/context` | Full graph context for a series |
| GET | `/graph/path/{source_id}/{target_id}` | Shortest path between entities |
| POST | `/graph/impact` | Impact analysis for a node change |
| POST | `/graph/scenario` | What-if scenario simulation |
| GET | `/graph/features/{lob}` | Graph-derived features for all series |
| GET | `/graph/export` | Export graph as JSON for frontend |

### Frontend Concepts

1. **Graph Explorer Page** ΓÇö Force-directed visualization (react-force-graph-2d or d3-force). Nodes colored by entity type, edges styled by relationship type. Click-to-inspect, search, filter by type.
2. **Series Context Panel** ΓÇö New tab on forecast detail page showing 2-hop neighborhood as mini-graph + table of graph features.
3. **Impact Analysis Widget** ΓÇö Select a node, specify a change (e.g., discontinue SKU-123), see affected series highlighted with estimated demand impact.
4. **Temporal Slider** ΓÇö Date slider on graph explorer showing how the graph evolves (transitions appearing/completing, new products launching).

### Phased Implementation

#### Phase 1: Foundation (2-3 weeks)
- `ontology.py`, `graph.py`, `builder.py`, `config.py`
- Build graph from HierarchyTree, SKU mappings, regressor config
- Add `KnowledgeGraphConfig` to `PlatformConfig`
- JSON persistence
- API: `/graph/stats`, `/graph/node/{id}`, `/graph/build`
- Unit tests: `test_knowledge_graph.py`

#### Phase 2: Feature Extraction + AI Integration (2-3 weeks)
- `features.py` ΓÇö structural graph features
- Integrate into `SeriesBuilder.build()` as additional features
- `ai_context.py` ΓÇö graph context for NL query engine
- `traversal.py` ΓÇö path finding, impact radius
- API: `/graph/features/{lob}`, `/graph/series/{id}/context`
- Backtest comparison: with vs. without graph features

#### Phase 3: Temporal + Scenarios (2-3 weeks)
- `temporal.py` ΓÇö edge versioning, point-in-time snapshots
- `scenarios.py` ΓÇö what-if propagation
- `similarity.py` ΓÇö correlation-based `similar_to` edges
- API: `/graph/impact`, `/graph/scenario`, `/graph/path`
- Integrate into `AnomalyTriageEngine` and `CommentaryEngine`

#### Phase 4: Advanced (3-4 weeks)
- GNN features: Node2Vec embeddings or lightweight GCN (torch-geometric)
- Supplier/Customer node ingestion from new data sources
- Frontend graph visualization components
- Temporal slider
- Performance optimization for large graphs (10K+ nodes)

### Dependencies

- **Phase 1-3**: `networkx>=3.0` (pure Python, no external deps)
- **Phase 4 only**: `node2vec>=0.4` or `torch-geometric` for graph embeddings

No external database required. Graph persists as JSON/Parquet files.

### Testing

- `tests/test_knowledge_graph.py` ΓÇö ontology, graph CRUD, builder
- `tests/test_graph_features.py` ΓÇö feature extraction correctness
- `tests/test_graph_traversal.py` ΓÇö path finding, impact analysis
- `tests/test_graph_scenarios.py` ΓÇö what-if propagation
- `tests/test_graph_ai_context.py` ΓÇö AI context generation
- `tests/test_graph_api.py` ΓÇö API endpoint integration

Fixture: `make_knowledge_graph()` factory using existing `_make_weekly_actuals()` and hierarchy fixtures.

### Research Sources

- [o9 Digital Brain Platform](https://o9solutions.com/digital-brain)
- [o9 Enterprise Knowledge Graph](https://o9solutions.com/videos/explaining-o9s-highly-differentiated-ekg)
- [Auger Supply Chain OS](https://www.oakhcft.com/news/leveraging-ai-to-redefine-supply-chain-tech-with-auger)
- [Auger + Microsoft Fabric Integration](https://blog.fabric.microsoft.com/en-US/blog/from-unified-data-to-decisive-action-advancing-supply-chain-autonomy-with-microsoft-fabric-and-auger/)
- [Knowledge Graph for Pharmaceutical Demand Forecasting (KG-GCN-LSTM)](https://www.nature.com/articles/s41598-026-35113-4)
- [GraphSAGE for Manufacturing Demand Forecasting](https://towardsdatascience.com/time-series-isnt-enough-how-graph-neural-networks-change-demand-forecasting/)
- [GNN Supply Chain Survey](https://arxiv.org/html/2411.08550v1)
- [Neo4j Supply Chain Use Cases](https://neo4j.com/use-cases/supply-chain-management/)
- [Ontology-based KG for Supply Chain Mapping](https://ipmu2024.inesc-id.pt/files/paper_2195.pdf)
- [KG Reasoning for Supply Chain Risk Management](https://www.tandfonline.com/doi/full/10.1080/00207543.2022.2100841)
- [Enhancing Supply Chain Visibility with KG + LLMs](https://arxiv.org/html/2408.07705v1)
- [Temporal Knowledge Graphs for Supply Chain Recommendations](https://www.mdpi.com/2079-9292/14/2/222)