# AI-Native Supply Chain Product Roadmap

## Context

This platform is a production-ready multi-frequency sales forecasting product with 16 algorithms, 4 Claude-powered AI engines, hierarchical reconciliation, and full governance. The competitive landscape has shifted dramatically: **o9 Solutions** deploys Enterprise Knowledge Graphs with GenAI Composite Agents, **Auger** (founded 2024, ex-Amazon leadership, $100M funding) is building an "AI-native from the ground up" Supply Chain Autonomy Platform on Microsoft Fabric, **Blue Yonder** launched 5 domain-specific AI agents with a Supply Chain Knowledge Graph, **Kinaxis** fuses heuristics+optimization+ML automatically, and **Anaplan** offers role-based AI agents with a conversational CoPlanner.

The common thread: **the industry is moving from AI-assisted forecasting to agentic autonomous planning**. This plan adds 7 major capabilities across 5 phases to position the platform competitively.

---

## Phase 1: Agentic Supply Chain Copilot (Highest Impact)

**Why**: Every competitor (o9, Anaplan, AWS) now has a conversational planning copilot. Our NL Query engine only answers questions about individual series. A copilot that can reason across the entire platform — running analyses, comparing scenarios, triggering pipelines — is table stakes.

### 1A. Multi-Turn Planning Copilot

**What**: An agentic Claude-powered copilot that can hold multi-turn conversations, call platform APIs as tools, and execute multi-step analytical workflows autonomously.

**Key Components**:
- `src/ai/copilot.py` — `PlanningCopilot` class with:
  - Tool registry mapping natural language intents to platform API calls
  - Conversation memory (multi-turn context)
  - Plan-then-execute pattern: Claude creates a plan → user approves → copilot executes steps
  - Tools available to Claude: run_backtest, get_forecast, compare_models, explain_series, get_drift_alerts, triage_anomalies, get_fva, apply_override, get_hierarchy, run_decomposition
- `src/ai/tools.py` — Tool definitions formatted for Claude tool_use (wrapping existing API logic from `src/api/routers/`)
- Streaming responses via SSE for real-time feedback

**API Endpoints**:
- `POST /ai/copilot/chat` — Send message, get streaming response with tool calls
- `GET /ai/copilot/sessions` — List active sessions
- `DELETE /ai/copilot/sessions/{id}` — End session

**Frontend**:
- `CopilotPanel` — Persistent side panel with chat interface, tool execution visibility, approval prompts
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

**Implementation**: Persona config in `src/ai/personas.py` — system prompt templates + tool allowlists per role, integrates with existing RBAC in `src/auth/`

---

## Phase 2: Scenario Planning & What-If Engine

**Why**: Google Cloud Supply Chain Twin, o9, and Coupa/LLamasoft all offer what-if simulation. This is the #1 gap — planners need to model "what happens if demand drops 20%" or "what if we lose supplier X" before it happens.

### 2A. Scenario Engine

**What**: Create, compare, and manage named forecast scenarios with parameter overrides.

**Key Components**:
- `src/scenarios/engine.py` — `ScenarioEngine` class:
  - Create scenario from base config with parameter overrides (demand shocks, model changes, constraint changes)
  - Run scenario (triggers backtest/forecast pipeline with modified config)
  - Compare scenarios side-by-side (metrics, forecasts, costs)
  - Persist scenarios to Parquet (reusable, shareable)
- `src/scenarios/types.py` — `Scenario` dataclass: name, description, base_config, overrides (dict of config path → value), demand_adjustments (series-level multipliers/offsets), constraint_overrides
- `src/scenarios/comparator.py` — `ScenarioComparator`: diff two scenario outputs (forecast deltas, metric deltas, cost impact)

**Scenario Types**:
1. **Demand shock**: Apply multiplier/offset to specific series/categories/regions
2. **Model swap**: Run same data through different model configurations
3. **Constraint change**: Modify capacity/budget constraints
4. **Regressor shift**: Modify external regressor assumptions (price changes, promo changes)
5. **Hierarchy change**: Add/remove nodes, test different aggregation levels

**API Endpoints**:
- `POST /scenarios` — Create scenario
- `POST /scenarios/{id}/run` — Execute scenario
- `GET /scenarios/{id}/results` — Get results
- `POST /scenarios/compare` — Compare 2+ scenarios
- `GET /scenarios` — List all scenarios

**Frontend**:
- `ScenarioBuilder` — Visual scenario configuration with parameter sliders
- `ScenarioComparison` — Side-by-side charts and metric tables
- `ScenarioLibrary` — Browse/clone/share saved scenarios

**AI Integration**: Copilot can create and compare scenarios via natural language: "What if holiday demand is 15% higher than last year?"

### 2B. Impact Simulation (inspired by Google Supply Chain Twin)

**What**: Cascading impact analysis — when one variable changes, show the ripple effects through the hierarchy.

- `src/scenarios/impact.py` — `ImpactSimulator`: Given a demand change at leaf level, propagate through hierarchy using reconciliation, show effects on parent nodes, constrained plans, and cost estimates
- Visualized as a Sankey diagram or hierarchy diff in frontend

---

## Phase 3: Autonomous Planning Agents

**Why**: Auger's core value prop is "autonomous execution in seconds." Blue Yonder has 5 specialized agents. This is the frontier — agents that monitor, detect, decide, and act within guardrails.

### 3A. Agent Framework

**What**: A lightweight agent framework where specialized agents continuously monitor metrics and take autonomous actions within configurable guardrails.

**Key Components**:
- `src/agents/base.py` — `SupplyChainAgent` ABC:
  - `monitor()` — Check conditions (called on schedule)
  - `evaluate()` — Decide if action needed (returns ActionProposal)
  - `execute()` — Take action (with guardrail checks)
  - `report()` — Log what happened to audit trail
- `src/agents/guardrails.py` — `GuardrailEngine`:
  - Configurable thresholds per agent (max forecast change %, max overrides per day, etc.)
  - Approval routing: auto-approve within guardrails, escalate to human outside
  - Full audit trail of all agent decisions via existing `src/audit/`
- `src/agents/scheduler.py` — Extends existing `PipelineScheduler` for agent loops

### 3B. Specialized Agents

| Agent | Monitors | Actions | Guardrails |
|-------|----------|---------|------------|
| **DriftResponseAgent** | `src/metrics/` drift alerts | Re-triggers backtest, swaps champion model, notifies planner | Max 1 model swap per series per week |
| **OverrideDecayAgent** | `src/overrides/` age | Expires stale overrides, proposes fresh adjustments based on actuals | Only expires overrides older than N cycles |
| **DataQualityAgent** | `src/data/validator.py` signals | Auto-cleanses new data, flags anomalies, adjusts cleansing params | Cannot delete data, only flag/impute |
| **RebalanceAgent** | Constrained forecast vs actuals | Adjusts capacity allocations, redistributes budget across hierarchy | Max N% reallocation per cycle |
| **NewProductAgent** | `src/sku_mapping/` transitions | Auto-maps new SKUs using similarity, bootstraps initial forecasts | Requires human approval for high-revenue SKUs |

**API Endpoints**:
- `GET /agents` — List agents and their status
- `POST /agents/{id}/enable` / `POST /agents/{id}/disable`
- `GET /agents/{id}/history` — Action history
- `PUT /agents/{id}/guardrails` — Configure guardrails
- `POST /agents/{id}/approve/{action_id}` — Approve pending action

**Frontend**:
- `AgentDashboard` — Agent status, recent actions, pending approvals
- `GuardrailConfig` — Visual guardrail editor per agent
- `AgentTimeline` — Chronological view of all agent actions with audit trail

**AI Integration**: Each agent uses Claude for reasoning about edge cases — e.g., DriftResponseAgent asks Claude "Should I swap this model given the recent holiday period?" before acting

---

## Phase 4: Supply Chain Knowledge Graph

**Why**: o9's Enterprise Knowledge Graph and Blue Yonder's Supply Chain Knowledge Graph are major differentiators. A knowledge graph connects entities (products, suppliers, customers, events, forecasts) enabling causal reasoning and cross-functional queries that flat tables cannot support.

### 4A. Entity-Relationship Graph

**What**: A lightweight knowledge graph built on NetworkX (no heavy graph DB dependency) that connects platform entities.

**Key Components**:
- `src/knowledge/graph.py` — `SupplyChainGraph`:
  - Nodes: Series, Products, Categories, Regions, Suppliers, Customers, Events, Models, Forecasts
  - Edges: belongs_to, supplied_by, sold_to, affected_by, forecasted_by, similar_to
  - Auto-populated from hierarchy tree, SKU mappings, regressor metadata
  - Queryable: "What series are affected by Supplier X?" "What products share demand patterns?"
- `src/knowledge/builder.py` — `GraphBuilder`: Constructs graph from existing data sources
  - Ingests hierarchy from `src/hierarchy/`
  - Ingests SKU relationships from `src/sku_mapping/`
  - Ingests regressor associations from `src/data/regressors.py`
  - Infers similarity edges from forecast correlation patterns

### 4B. Causal Reasoning Layer

**What**: Extend the existing `CausalAnalyzer` with graph-powered causal inference.

- `src/knowledge/causal.py` — `CausalReasoningEngine`:
  - "Why did forecast change?" → Trace graph edges to find upstream causes (regressor shifts, supplier changes, seasonal events)
  - "What will be affected?" → Forward propagation through graph
  - Claude interprets causal chains into natural language narratives

**API Endpoints**:
- `GET /knowledge/graph/{lob}` — Get graph structure
- `POST /knowledge/query` — Natural language graph query
- `POST /knowledge/impact` — "What-if" through graph (e.g., "What if Supplier X delays 2 weeks?")
- `GET /knowledge/similar/{series_id}` — Find similar series via graph proximity

**Frontend**:
- `KnowledgeGraph` — Interactive graph visualization (force-directed or hierarchical)
- `CausalChain` — Visual cause-effect chain for forecast explanations
- Integrated into copilot: "Show me everything connected to SKU-1234"

---

## Phase 5: External Signal Integration & Demand Sensing

**Why**: Kinaxis, AWS, and o9 all fuse 100+ external signals (weather, POS, social media, news, economic indicators) for real-time demand sensing. This turns periodic forecasting into continuous adaptive planning.

### 5A. Signal Registry

**What**: A pluggable registry for external data signals that feed into demand sensing.

- `src/signals/registry.py` — `SignalRegistry`:
  - Register signal sources with metadata (name, frequency, latency, coverage)
  - Built-in connectors: weather APIs, Google Trends, economic indicators (FRED), news sentiment
  - Each signal implements `SignalSource` ABC: `fetch(date_range, entities)` → Polars DataFrame
- `src/signals/connectors/` — Individual connector modules:
  - `weather.py` — OpenWeatherMap / Visual Crossing
  - `trends.py` — Google Trends via pytrends
  - `economic.py` — FRED API (CPI, unemployment, consumer sentiment)
  - `news.py` — News API + Claude-powered sentiment scoring
  - `custom.py` — User-uploaded signal files

### 5B. Demand Sensing Engine

**What**: Short-horizon demand adjustment using leading indicators.

- `src/signals/sensing.py` — `DemandSensingEngine`:
  - Ingests latest signals from registry
  - Trains lightweight gradient-boosted adjustment model (extends `src/forecasting/ml.py`)
  - Outputs adjustment factors applied to base statistical/ML forecast
  - Updates continuously as new signals arrive (not just at reforecast time)

### 5C. AI-Powered Signal Interpretation

- Claude analyzes incoming signals and explains their demand implications
- "Breaking news about port strikes → Claude explains expected impact on Category X supply"
- Integrated with copilot and alert system

**API Endpoints**:
- `GET /signals` — List registered signals
- `POST /signals/register` — Register new signal source
- `POST /signals/sense` — Run demand sensing for LOB
- `GET /signals/impact/{lob}` — Signal attribution (which signals moved the forecast)

**Frontend**:
- `SignalDashboard` — Signal health, latency, coverage
- `DemandSensing` — Overlay base forecast with signal-adjusted forecast
- `SignalAttribution` — Which external signals are driving adjustments

---

## Implementation Priority & Dependencies

```
Phase 1 (Copilot)          ← Builds on existing src/ai/, highest ROI
  ↓
Phase 2 (Scenarios)        ← Builds on existing pipeline, config system
  ↓
Phase 3 (Agents)           ← Requires Phases 1+2 for full value
  ↓
Phase 4 (Knowledge Graph)  ← Requires hierarchy, SKU mapping, regressors
  ↓
Phase 5 (Demand Sensing)   ← Requires signal infrastructure, can start independently
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
1. **Open/self-hosted** — Every competitor is SaaS-only. This platform can be deployed on-prem or in customer's cloud
2. **Foundation model integration** — Only platform with Chronos + TimeGPT zero-shot forecasting
3. **Deep explainability** — SHAP + STL + FVA + Claude narratives is deeper than any competitor
4. **Multi-frequency native** — D/W/M/Q from a single codebase with frequency profiles
5. **Claude-native AI** — Deepest LLM integration (not just a chatbot layer) with domain-specific reasoning

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
