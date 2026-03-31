# Adding Custom Forecasting Models

This guide shows how to add a new forecasting model to the platform using the `BaseForecaster` abstract class and the `ForecasterRegistry`.

---

## Overview

Every model in the platform:
1. Extends `BaseForecaster`
2. Registers with `@registry.register("name")`
3. Implements `fit()` and `predict()`
4. Optionally implements `predict_quantiles()` for probabilistic forecasts

Once registered, the model is automatically available in configs, backtests, and the API.

---

## Step-by-Step

### 1. Create the model file

Create `src/forecasting/my_model.py`:

```python
import polars as pl
from typing import Any, Dict, List

from .base import BaseForecaster
from .registry import registry


@registry.register("my_model")
class MyModelForecaster(BaseForecaster):
    """One-line description of your model."""

    name = "my_model"

    def __init__(self, season_length: int = 52, my_param: float = 0.5):
        self.season_length = season_length
        self.my_param = my_param
        self._fitted_data = None

    def fit(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> None:
        """Fit on panel data. Store whatever state your model needs."""
        # validate_and_prepare is called automatically by the pipeline,
        # but you can call it here for standalone usage:
        df = self.validate_and_prepare(df, target_col, time_col, id_col)
        self._fitted_data = df
        # ... your training logic ...

    def predict(
        self,
        horizon: int,
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """Generate point forecasts. Must return [id_col, time_col, 'forecast']."""
        # ... your prediction logic ...
        return pl.DataFrame({
            id_col: series_ids,
            time_col: future_dates,
            "forecast": predictions,
        })

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float],
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """Optional: probabilistic forecasts.
        Returns columns: forecast_p10, forecast_p50, forecast_p90, etc.
        If not overridden, the base class returns degenerate intervals
        (point forecast copied to all quantile columns).
        """
        point = self.predict(horizon, id_col, time_col)
        # ... your interval logic ...
        return point

    def get_params(self) -> Dict[str, Any]:
        """Return model parameters for logging and reproducibility."""
        return {
            "season_length": self.season_length,
            "my_param": self.my_param,
        }
```

### 2. Register the import

Add an import to `src/forecasting/__init__.py` so the module loads at startup:

```python
from . import my_model  # noqa: F401
```

### 3. Add to your config

```yaml
forecast:
  forecasters:
    - naive_seasonal
    - lgbm_direct
    - my_model              # <-- your new model
```

With per-model parameters:

```yaml
forecast:
  forecasters:
    - naive_seasonal
    - lgbm_direct
    - my_model
  model_params:
    my_model:
      season_length: 52
      my_param: 0.75
```

The registry's `build()` method uses `inspect.signature` to silently drop unknown kwargs, so mismatched param names won't crash — they'll just use defaults.

### 4. Test

Run a backtest to verify your model participates in the tournament:

```bash
curl -X POST http://localhost:8000/pipeline/backtest \
  -F "file=@actuals.csv" \
  -F "lob=test"
```

Or via Python:

```python
from forecasting_product.src.pipeline.backtest import BacktestPipeline
from forecasting_product.src.config.loader import load_config

config = load_config("config.yaml")
pipeline = BacktestPipeline(config)
results = pipeline.run(actuals)

# Check your model appears in the leaderboard
for entry in results["leaderboard"]:
    print(f"{entry['model_id']}: WMAPE={entry['wmape']:.3f}")
```

---

## BaseForecaster API

| Method | Required | Description |
|--------|----------|-------------|
| `fit(df, target_col, time_col, id_col)` | **Yes** | Fit on panel training data (Polars DataFrame) |
| `predict(horizon, id_col, time_col)` | **Yes** | Return `[id_col, time_col, "forecast"]` DataFrame |
| `predict_quantiles(horizon, quantiles, id_col, time_col)` | No | Return quantile columns (`forecast_p{q}`). Default: degenerate intervals |
| `validate_and_prepare(df, target_col, time_col, id_col)` | No | Pre-fit validation hook. Default: no-op |
| `fill_weekly_gaps(df, time_col, id_col, target_col, strategy)` | No | Static utility to fill missing periods |
| `get_params()` | No | Return params dict for logging |

### Input Conventions

- All data is `pl.DataFrame` (Polars)
- Panel format: multiple series identified by `id_col`, time indexed by `time_col`
- `fit()` receives the full panel; your model handles multi-series internally
- `predict()` returns forecasts for all fitted series

### Output Requirements

`predict()` must return exactly these columns:

| Column | Type | Description |
|--------|------|-------------|
| `{id_col}` | `str` | Series identifier |
| `{time_col}` | `date` | Future date |
| `forecast` | `float` | Point forecast |

`predict_quantiles()` adds columns named `forecast_p{int(q*100)}` (e.g., `forecast_p10`, `forecast_p90`).

---

## ForecasterRegistry API

```python
from forecasting_product.src.forecasting.registry import registry

# List all registered models
registry.available  # ['naive_seasonal', 'ses', 'ets', 'lgbm_direct', ...]

# Get class by name
cls = registry.get("my_model")

# Instantiate with params
model = registry.build("my_model", season_length=52, my_param=0.75)

# Batch instantiate
models = registry.build_from_config(
    names=["my_model", "lgbm_direct"],
    params={"my_model": {"my_param": 0.5}},
)
```

---

## Tips

- **Sparse/intermittent models:** Add your model name to `forecast.intermittent_forecasters` in config. The pipeline auto-routes lumpy/intermittent series (SBC classification) to these models.
- **Regressors:** Access external features via the DataFrame passed to `fit()` — `SeriesBuilder` joins regressors before fitting.
- **GPU:** Set `parallelism.gpu: true` and handle device placement in your `fit()`/`predict()`.
- **Logging:** Use `from ..observability.context import PipelineContext` for structured logging within pipeline runs.

---

## Existing Models

Use `registry.available` or check `src/forecasting/` for all built-in models:

| Name | Type | Description |
|------|------|-------------|
| `naive_seasonal` | Statistical | Last-season repeat |
| `ses` | Statistical | Simple exponential smoothing |
| `ets` | Statistical | Error-Trend-Seasonal (Holt-Winters) |
| `arima` | Statistical | Auto ARIMA via StatsForecast |
| `theta` | Statistical | Theta method |
| `lgbm_direct` | ML | LightGBM direct multi-step |
| `xgboost_direct` | ML | XGBoost direct multi-step |
| `chronos` | Foundation | Amazon Chronos (zero-shot) |
| `timegpt` | Foundation | Nixtla TimeGPT (API-based) |

See [Foundation Models](FOUNDATION_MODELS.md) for Chronos and TimeGPT setup.
