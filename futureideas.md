# Future Ideas

## 1. Knowledge Graph for Supply Chain Forecasting

### Context

The platform currently models supply chain entities (products, geographies, channels, SKUs) as **isolated hierarchical trees** and **Polars join-based relationships**. There is no unified graph representation connecting products to their regressors, transitions, channels, geographies, pipeline runs, or similar series. This limits the platform in several ways:

1. **Forecasting models treat each series independently** — no cross-series information flow (e.g., sibling products, cannibalization, transitions)
2. **AI engines lack structural context** — Claude answers questions about a series without knowing its graph neighborhood (what it transitions to, what drives it, what competes with it)
3. **No impact analysis** — cannot answer "if we discontinue SKU-X, what series are affected?" without manual investigation
4. **No what-if scenarios** — cannot propagate a demand change through the supply chain network

### Industry Research

- **o9 Solutions**: Enterprise Knowledge Graph (EKG) with Graph-Cube technology — a graph-based digital twin connecting products, customers, suppliers, facilities across supply chain, finance, and commercial domains. Powers AI agents that traverse the graph for cross-functional queries. (Sense -> Model -> Simulate -> Decide -> Execute -> Learn cycle)
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
    graph.py             # KnowledgeGraph — wraps NetworkX MultiDiGraph
    builder.py           # KnowledgeGraphBuilder — populates from HierarchyTree, MappingRecords, etc.
    temporal.py          # TemporalEdgeStore — edge validity intervals, point-in-time snapshots
    features.py          # GraphFeatureExtractor — derives numeric features for forecasting models
    traversal.py         # GraphTraversal — path finding, impact radius, neighbor queries
    similarity.py        # SeriesSimilarity — correlation/DTW, builds similar_to edges
    scenarios.py         # ScenarioEngine — what-if impact propagation
    ai_context.py        # GraphContextProvider — structured context for AI engines
    config.py            # KnowledgeGraphConfig dataclass
```

### Key Design Decisions

1. **NetworkX (not Neo4j)** — keeps the platform self-contained. Suitable for graphs up to ~100K nodes. The `graph.py` wrapper abstracts the backend so it can be swapped later if needed.

2. **Graph features injected via existing regressor path** — `GraphFeatureExtractor` returns a Polars DataFrame with `series_id` column. This joins into `SeriesBuilder.build()` using the same mechanism as external regressors. LightGBM, XGBoost, and neural models all benefit with zero model code changes.

3. **Opt-in via config** — `knowledge_graph.enabled = False` by default. Each sub-feature (feature extraction, similarity, temporal snapshots) has its own toggle.

4. **AI grounding before GNN training** — immediate value comes from providing structured graph context to Claude via `AIFeatureBase` pattern (lower risk, faster value). GNN embeddings are Phase 4.

5. **Temporal edges via Polars** — edge versions stored in a Polars DataFrame for efficient time-range filtering, consistent with the platform's Polars-native approach.

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

1. **Graph Explorer Page** — Force-directed visualization (react-force-graph-2d or d3-force). Nodes colored by entity type, edges styled by relationship type. Click-to-inspect, search, filter by type.
2. **Series Context Panel** — New tab on forecast detail page showing 2-hop neighborhood as mini-graph + table of graph features.
3. **Impact Analysis Widget** — Select a node, specify a change (e.g., discontinue SKU-123), see affected series highlighted with estimated demand impact.
4. **Temporal Slider** — Date slider on graph explorer showing how the graph evolves (transitions appearing/completing, new products launching).

### Phased Implementation

#### Phase 1: Foundation (2-3 weeks)
- `ontology.py`, `graph.py`, `builder.py`, `config.py`
- Build graph from HierarchyTree, SKU mappings, regressor config
- Add `KnowledgeGraphConfig` to `PlatformConfig`
- JSON persistence
- API: `/graph/stats`, `/graph/node/{id}`, `/graph/build`
- Unit tests: `test_knowledge_graph.py`

#### Phase 2: Feature Extraction + AI Integration (2-3 weeks)
- `features.py` — structural graph features
- Integrate into `SeriesBuilder.build()` as additional features
- `ai_context.py` — graph context for NL query engine
- `traversal.py` — path finding, impact radius
- API: `/graph/features/{lob}`, `/graph/series/{id}/context`
- Backtest comparison: with vs. without graph features

#### Phase 3: Temporal + Scenarios (2-3 weeks)
- `temporal.py` — edge versioning, point-in-time snapshots
- `scenarios.py` — what-if propagation
- `similarity.py` — correlation-based `similar_to` edges
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

- `tests/test_knowledge_graph.py` — ontology, graph CRUD, builder
- `tests/test_graph_features.py` — feature extraction correctness
- `tests/test_graph_traversal.py` — path finding, impact analysis
- `tests/test_graph_scenarios.py` — what-if propagation
- `tests/test_graph_ai_context.py` — AI context generation
- `tests/test_graph_api.py` — API endpoint integration

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
