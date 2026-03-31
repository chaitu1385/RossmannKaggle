# Foundation Models

The platform supports two zero-shot foundation models for time series forecasting: **Amazon Chronos** and **Nixtla TimeGPT**. Both integrate via the standard `BaseForecaster` / registry pattern and work alongside statistical and ML models in backtests.

---

## Amazon Chronos

[Chronos](https://github.com/amazon-science/chronos-forecasting) is a family of pretrained time series models based on T5 and Bolt architectures. Runs locally — no API key required.

### Installation

```bash
pip install chronos-forecasting torch
```

For GPU inference:
```bash
pip install chronos-forecasting torch --extra-index-url https://download.pytorch.org/whl/cu118
```

### Configuration

```yaml
forecast:
  forecasters:
    - naive_seasonal
    - lgbm_direct
    - chronos

# Optional: model-specific params
  model_params:
    chronos:
      model_name: amazon/chronos-t5-small   # default: chronos-t5-tiny
      device: cuda                           # default: cpu
      num_samples: 20                        # Monte Carlo samples for quantiles
      torch_dtype: bfloat16                  # default: bfloat16
      frequency: W                           # default: W

parallelism:
  gpu: true                                  # Enable GPU device placement
```

### Available Models

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| `amazon/chronos-t5-tiny` | 8M | Fastest | Good | Development, quick iteration |
| `amazon/chronos-t5-small` | 20M | Fast | Better | Default for production |
| `amazon/chronos-t5-base` | 200M | Medium | Strong | High-value series |
| `amazon/chronos-t5-large` | 710M | Slow | Strongest | Benchmark, critical series |
| `amazon/chronos-bolt-tiny` | 9M | Fastest | Good | Real-time serving |
| `amazon/chronos-bolt-small` | 48M | Fast | Better | Balanced |
| `amazon/chronos-bolt-base` | 205M | Medium | Strong | Production recommended |

Bolt models are faster due to architectural optimizations (direct multi-step output vs autoregressive).

### How It Works

1. **`fit()`** — Stores historical data as context (no gradient-based training)
2. **`predict()`** — Lazy-loads the HuggingFace pipeline (`ChronosPipeline.from_pretrained`), runs batch inference across all series
3. **`predict_quantiles()`** — Generates `num_samples` Monte Carlo trajectories and extracts arbitrary quantiles

### Resource Requirements

| Model | VRAM (GPU) | RAM (CPU) | Inference Time (1,000 series) |
|-------|-----------|-----------|-------------------------------|
| tiny | 0.5 GB | 1 GB | ~30s (GPU) / ~5min (CPU) |
| small | 1 GB | 2 GB | ~1min (GPU) / ~10min (CPU) |
| base | 2 GB | 4 GB | ~3min (GPU) / ~30min (CPU) |
| large | 4 GB | 8 GB | ~10min (GPU) / ~2hr (CPU) |

---

## Nixtla TimeGPT

[TimeGPT](https://docs.nixtla.io/) is Nixtla's proprietary foundation model, accessed via API. No local compute required.

### Installation

```bash
pip install nixtla
```

### Setup

1. Get an API key from [dashboard.nixtla.io](https://dashboard.nixtla.io)
2. Set the environment variable:
   ```bash
   export NIXTLA_API_KEY=nixtla-...
   ```

### Configuration

```yaml
forecast:
  forecasters:
    - naive_seasonal
    - lgbm_direct
    - timegpt

  model_params:
    timegpt:
      model: timegpt-1                    # default
      frequency: W                         # default: W
      # api_key: nixtla-...               # or set NIXTLA_API_KEY env var
```

### Available Models

| Model | Max History | Best For |
|-------|------------|----------|
| `timegpt-1` | 512 time steps | Most use cases (≤36 period horizon) |
| `timegpt-1-long-horizon` | 512 time steps | Long horizons (>36 periods) |

### How It Works

1. **`fit()`** — Stores training data as a Polars DataFrame (defers pandas conversion to predict time)
2. **`predict()`** — Converts to pandas, calls `NixtlaClient.forecast(...)` 
3. **`predict_quantiles()`** — Maps quantile levels to Nixtla's `level` parameter (e.g., quantiles `[0.1, 0.9]` → `level=[80]`)

### Cost Considerations

TimeGPT pricing is per-series-per-call. For backtests with multiple folds:
- 500 series × 3 folds = 1,500 API calls
- Consider running TimeGPT only on final evaluation, not every fold

---

## Using in Backtests

Both foundation models participate in the standard model tournament:

```python
from forecasting_product.src.pipeline.backtest import BacktestPipeline
from forecasting_product.src.config.loader import load_config

config = load_config("config.yaml")  # with chronos/timegpt in forecasters list
pipeline = BacktestPipeline(config)
results = pipeline.run(actuals)

# Foundation models appear in leaderboard alongside statistical/ML models
for entry in results["leaderboard"]:
    print(f"{entry['model_id']}: WMAPE={entry['wmape']:.3f}")
```

The champion selector picks the best model regardless of type — a foundation model can win if it outperforms.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: chronos` | `pip install chronos-forecasting torch` |
| `ModuleNotFoundError: nixtla` | `pip install nixtla` |
| Chronos OOM on GPU | Use a smaller model (tiny/bolt-tiny) or set `device: cpu` |
| Chronos slow on CPU | Use bolt variants which are 3–5x faster |
| TimeGPT auth error | Check `NIXTLA_API_KEY` env var |
| TimeGPT timeout | Reduce series count or add retry logic |
| Poor foundation model accuracy | These are zero-shot — no domain tuning. Compare WMAPE against tuned statistical models before deciding. |

See also [Adding Custom Models](ADDING_MODELS.md) for how these integrate with the `BaseForecaster` / registry system.
