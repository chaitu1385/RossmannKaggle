# AI Features Guide

The Forecasting Platform integrates Claude (Anthropic) to provide four AI-powered capabilities. All are optional — the platform works fully without an API key, falling back to template-based or rule-based responses.

---

## Setup

1. Get an API key from [console.anthropic.com](https://console.anthropic.com)
2. Set the environment variable:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   ```
3. Enable in config:
   ```yaml
   ai:
     enabled: true
     api_key_env_var: ANTHROPIC_API_KEY
     model: claude-sonnet-4-20250514
     max_tokens: 2000
   ```

All four engines inherit from `AIFeatureBase`, which manages the Anthropic client, prompt formatting, and response parsing.

---

## 1. Natural Language Query (`/ai/explain`)

**What it does:** Answers natural-language questions about a specific series forecast — "Why did SKU_001 forecast spike?" or "Is this forecast reliable?"

**How it works:**
1. Gathers context: series history stats (mean, std, min, max, recent values), forecast summary, accuracy metrics, model comparison data
2. Sends to Claude with a demand-planning-focused system prompt
3. Parses response into answer + confidence level + supporting data

**API:**

```bash
curl -X POST http://localhost:8000/ai/explain \
  -H "Content-Type: application/json" \
  -d '{"series_id": "sku_001", "question": "Why is next month forecast 30% higher?", "lob": "retail"}'
```

```json
{
  "answer": "The forecast increase is driven by...",
  "confidence": "medium",
  "supporting_data": {"trend_direction": "increasing", "seasonal_peak": true},
  "sources_used": ["history_stats", "forecast_summary"]
}
```

**Confidence levels:** `high` (strong data support), `medium` (partial evidence), `low` (limited data, speculative).

---

## 2. Executive Commentary (`/ai/commentary`)

**What it does:** Generates VP-level executive commentary for S&OP meetings — summarizing forecast performance, exceptions, and action items.

**How it works:**
1. Gathers context: overall WMAPE/bias, drift alerts (critical + warning counts, top 5 details), model leaderboard (top 5), FVA cascade summary
2. Sends to Claude with an S&OP-focused system prompt
3. Parses into executive summary, key metrics, exceptions, and action items

**API:**

```bash
curl -X POST http://localhost:8000/ai/commentary \
  -H "Content-Type: application/json" \
  -d '{"lob": "retail", "run_type": "backtest", "period_start": "2026-01-01", "period_end": "2026-03-31"}'
```

```json
{
  "executive_summary": "Retail forecasting performance improved in Q1...",
  "key_metrics": [
    {"name": "Overall WMAPE", "value": 0.142, "trend": "improving"}
  ],
  "exceptions": ["SKU_042 bias exceeds 15% for 4 consecutive weeks"],
  "action_items": ["Review SKU_042 for potential promotion not captured in model"]
}
```

**Fallback:** When Claude is unavailable, returns a template-based summary from the raw metrics.

---

## 3. Alert Triage (`/ai/triage`)

**What it does:** Prioritizes drift alerts by business impact and suggests corrective actions — turning a noisy alert list into an actionable ranked queue.

**How it works:**
1. Takes drift alerts (up to 50) with optional series context (volume, revenue weight)
2. Sends to Claude with a triage-analyst system prompt
3. Returns ranked alerts with business impact scores (0–100), suggested actions, and reasoning

**API:**

```bash
curl -X POST http://localhost:8000/ai/triage \
  -H "Content-Type: application/json" \
  -d '{"lob": "retail", "severity_filter": "critical", "max_alerts": 20}'
```

```json
{
  "executive_summary": "3 critical alerts require immediate attention...",
  "total_alerts": 20,
  "critical_count": 3,
  "warning_count": 17,
  "ranked_alerts": [
    {
      "series_id": "sku_001",
      "business_impact_score": 92,
      "suggested_action": "Retrain with last 26 weeks; check for product reformulation",
      "reasoning": "High-volume SKU with accuracy degradation of 133%"
    }
  ]
}
```

**Fallback:** Returns alerts in original severity order without scoring.

---

## 4. Config Tuner (`/ai/recommend-config`)

**What it does:** Analyzes backtest results and recommends specific YAML configuration changes to improve accuracy. Conservative — only recommends changes with clear evidence.

**How it works:**
1. Gathers context: current config (forecasters, horizon, frequency, intermittent settings), leaderboard (top 10), FVA cascade, champion table, forecastability distribution
2. Sends to Claude with a configuration-expert system prompt (3000 max tokens)
3. Returns recommendations with exact YAML paths, current vs. recommended values, reasoning, expected impact, and risk level

**API:**

```bash
curl -X POST http://localhost:8000/ai/recommend-config \
  -H "Content-Type: application/json" \
  -d '{"lob": "retail"}'
```

```json
{
  "overall_assessment": "The current configuration is solid but...",
  "risk_summary": "All recommendations are low-to-medium risk...",
  "recommendations": [
    {
      "field_path": "forecast.forecasters",
      "current_value": ["naive_seasonal", "lgbm_direct"],
      "recommended_value": ["naive_seasonal", "lgbm_direct", "ets"],
      "reasoning": "ETS would provide better coverage for smooth series (42% of portfolio)",
      "expected_impact": "1-3% WMAPE improvement on smooth series",
      "risk": "low"
    }
  ]
}
```

---

## Python API

All engines can be used directly in Python:

```python
from forecasting_product.src.ai.nl_query import NaturalLanguageQueryEngine
from forecasting_product.src.ai.commentary import CommentaryEngine
from forecasting_product.src.ai.anomaly_triage import AnomalyTriageEngine
from forecasting_product.src.ai.config_tuner import ConfigTunerEngine

# All share the same constructor pattern
engine = NaturalLanguageQueryEngine(
    api_key="sk-ant-...",                  # or set ANTHROPIC_API_KEY env var
    model="claude-sonnet-4-20250514",           # default
    max_tokens=2000,                       # default (3000 for ConfigTuner)
)

# Check availability
if engine.available:
    result = engine.query(
        series_id="sku_001",
        question="Why did the forecast change?",
        lob="retail",
        history=history_df,
        forecast=forecast_df,
        metrics_df=metrics_df,
    )
    print(result.answer)
    print(result.confidence)
```

---

## Cost Considerations

| Engine | Typical Tokens In | Typical Tokens Out | Cost per Call (~) |
|--------|-------------------|--------------------|--------------------|
| NL Query | 500–1,500 | 200–500 | $0.002–0.005 |
| Commentary | 1,000–3,000 | 500–1,000 | $0.005–0.012 |
| Triage | 1,500–5,000 | 500–1,500 | $0.006–0.020 |
| Config Tuner | 2,000–5,000 | 1,000–3,000 | $0.010–0.025 |

Costs based on Claude Sonnet pricing. Commentary and Triage scale with alert count.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Claude is not available" | Check `ANTHROPIC_API_KEY` is set and valid |
| Slow responses | Reduce `max_alerts` for triage; reduce context window |
| Poor quality answers | Ensure backtest metrics exist — AI engines need data context |
| Rate limits | Anthropic rate limits apply; add retry logic for batch usage |

See also [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for general error handling.
