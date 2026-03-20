# Troubleshooting

Common errors, frequently asked questions, and debugging tips.

---

## Common Errors

### "Missing columns: ['week', 'quantity']"

```
ValueError: [series_id, week, quantity] missing: ['week', 'quantity']
```

**Cause:** Your CSV column names don't match what the platform expects.

**Fix:** Either rename your columns or configure the expected names in YAML:

```yaml
forecast:
  time_column: Date           # match your CSV's date column
  target_column: Sales        # match your CSV's target column
  series_id_column: Store     # match your CSV's ID column
```

---

### "Forecaster 'xxx' not registered"

```
KeyError: "Forecaster 'unknown_model' not registered. Available: [...]"
```

**Cause:** Misspelled model name in config or the model's dependency isn't installed.

**Fix:** Check the available model names:

| Name | Requires |
|------|----------|
| `naive_seasonal` | (built-in) |
| `auto_arima`, `auto_ets`, `auto_theta`, `mstl` | `statsforecast` |
| `lgbm_direct` | `lightgbm`, `mlforecast` |
| `xgboost_direct` | `xgboost`, `mlforecast` |
| `nbeats`, `nhits`, `tft` | `neuralforecast` |
| `chronos` | `chronos` |
| `timegpt` | `nixtla` |
| `croston`, `croston_sba`, `tsb` | `statsforecast` |

---

### "Unsupported frequency 'X'"

```
ValueError: "Unsupported frequency 'X'. Choose from: ['D', 'W', 'M', 'Q']"
```

**Cause:** Invalid frequency code in config.

**Fix:** Use one of: `D` (daily), `W` (weekly), `M` (monthly), `Q` (quarterly).

---

### "No forecast data found for LOB 'retail'"

```
HTTPException 404: "No forecast data found for LOB 'retail'. Expected directory: ..."
```

**Cause:** You're querying the API before running the forecast pipeline.

**Fix:** Run the forecast pipeline first:
```bash
python forecasting-platform/scripts/run_forecast.py \
  --config configs/platform_config.yaml --lob retail
```

---

### "Authentication required" / "Invalid or expired token"

```
HTTPException 401: "Authentication required. Provide a Bearer token."
HTTPException 401: "Invalid or expired token."
```

**Cause:** Auth is enabled but no valid token was provided.

**Fix:**
1. Get a token: `curl -X POST "http://localhost:8000/auth/token?username=user&role=data_scientist"`
2. Use it: `curl -H "Authorization: Bearer <token>" http://localhost:8000/forecast/retail`
3. Tokens expire after 24 hours — request a new one.

---

### "Permission 'xxx' required"

```
HTTPException 403: "Permission 'run_backtest' required. Your role 'viewer' does not have this permission."
```

**Cause:** Your user role doesn't have the required permission.

**Fix:** Request a token with the appropriate role. See [DEPLOYMENT.md](DEPLOYMENT.md) for the role-permission matrix.

---

### "Unsupported file format: .xlsx"

```
ValueError: "Unsupported file format: .xlsx. Use .parquet or .csv"
```

**Cause:** The platform only accepts CSV and Parquet files.

**Fix:** Save your Excel file as CSV first.

---

### "Claude client not available" / "anthropic package not installed"

```
RuntimeError: "Claude client not available"
RuntimeError: "anthropic package not installed"
```

**Cause:** AI features require the `anthropic` package and an API key.

**Fix:**
```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

AI features degrade gracefully when Claude is unavailable — they return template-based responses using the same data structures. No data is lost.

---

### "PyJWT is required for token operations"

```
RuntimeError: "PyJWT is required for token operations. Install with: pip install PyJWT"
```

**Fix:** `pip install PyJWT`

---

## Frequently Asked Questions

### How many weeks of history do I need?

Minimum history depends on frequency:

| Frequency | Minimum | Recommended |
|-----------|---------|-------------|
| Daily | 90 days | 180+ days |
| Weekly | 52 weeks | 104+ weeks |
| Monthly | 24 months | 36+ months |
| Quarterly | 8 quarters | 12+ quarters |

Foundation models (Chronos, TimeGPT) can work with less history because they don't learn seasonality from scratch.

---

### Can I use daily data?

Yes. Set `frequency: "D"` in your config. The platform auto-adjusts season lengths (7 for daily), lag features, and minimum history requirements.

```yaml
forecast:
  frequency: "D"
  horizon_weeks: 90    # 90-day horizon
```

---

### How do I add a new model?

Register it with the `ForecasterRegistry`:

```python
from src.forecasting.registry import registry
from src.forecasting.base import BaseForecaster

class MyForecaster(BaseForecaster):
    def fit(self, data, config):
        # your training logic
        return self

    def predict(self, horizon, **kwargs):
        # your prediction logic
        return forecast_df

registry.register("my_model", MyForecaster)
```

Then reference it in config:
```yaml
forecast:
  forecasters: [my_model, lgbm_direct]
```

---

### Why are my forecasts flat?

Common causes:
1. **Constant series** — If your data has no variation, models produce flat forecasts. Check for placeholder data.
2. **Insufficient history** — With very short series, models default to the mean. Provide more history or use foundation models.
3. **All-zero series after cleansing** — Aggressive outlier cleansing may remove real demand signals. Check the cleansing report.
4. **Wrong frequency** — Monthly data parsed as weekly creates gaps that get filled with zeros.

---

### Why are some series missing from the forecast output?

Series can be dropped during processing:
1. **Too short** — Below `min_series_length_weeks` (default: 52 for weekly)
2. **All zeros** — Dropped when `drop_zero_series: true`
3. **Model failure** — The model failed to fit a specific series. Check `engine.failures` for details.
4. **Parallel batch failure** — A worker process crashed. Check logs and compare input vs output series counts.

---

### How do I handle sparse/intermittent demand?

Enable sparse detection:

```yaml
forecast:
  sparse_detection: true
  intermittent_forecasters: [croston_sba, tsb]
```

The platform auto-classifies each series using the Syntetos-Boylan-Croston (SBC) matrix:

| Classification | Condition | Routed To |
|---------------|-----------|-----------|
| Smooth | CV² < 0.49, ADI < 1.32 | Regular models |
| Erratic | CV² >= 0.49, ADI < 1.32 | Regular models |
| Intermittent | CV² < 0.49, ADI >= 1.32 | Intermittent models |
| Lumpy | CV² >= 0.49, ADI >= 1.32 | Intermittent models |

---

### Can I use the platform without the dashboard?

Yes. The platform is fully usable via:
- **CLI scripts** — `run_backtest.py`, `run_forecast.py`
- **Python API** — Import classes directly (see Quick Start in README)
- **REST API** — All endpoints available via HTTP

The Streamlit dashboard is a convenience layer, not a requirement.

---

## Debugging Tips

### Enable structured logging

```yaml
observability:
  log_format: json
  log_level: DEBUG
```

JSON logs include `run_id` for correlating events across a pipeline run.

### Check the pipeline manifest

Every forecast run produces a manifest JSON file with full provenance — what data, config, and model produced the forecast. Look for:
- `validation_warnings` — Data quality issues that were non-blocking
- `outliers_clipped` — How many values were modified by cleansing
- `regressors_dropped` — Features that failed screening
- `champion_model_id` — Which model actually ran

### Use drift alerts for monitoring

Set up webhook alerts to catch accuracy degradation early:

```yaml
observability:
  alerts:
    channels: [webhook]
    webhook_url: "https://hooks.slack.com/services/..."
    min_severity: critical
```

Start with `critical` severity only. Lower to `warning` after tuning thresholds.

### Check for model failures

After a backtest:

```python
failures = engine.get_failure_summary()
if len(failures) > 0:
    print(failures)   # DataFrame with model_name, fold, error_type, message
```

---

## Performance Issues

### Backtests are slow

- **Reduce models** — Start with 2-3 candidates, not all 18
- **Reduce folds** — `n_folds: 2` instead of 5 (less robust but faster)
- **Skip neural models** — `nbeats`, `nhits`, `tft` are orders of magnitude slower than statistical/ML
- **Enable parallelism** — `n_workers: -1` uses all CPU cores
- **Use batch processing** — `batch_size: 500` to chunk large datasets

### Out of memory

- **Reduce `n_workers`** — Each worker loads a full model + data copy
- **Increase `batch_size`** — Smaller chunks per worker
- **Use Spark** — `backend: spark` for datasets that don't fit in memory
- **Drop neural models** — They have the largest memory footprint

### Docker container crashes

- Provision at least **2 GB RAM** for datasets with 1000+ series
- The Rossmann demo dataset (1M rows) runs comfortably in 2 GB
- For larger datasets, increase container memory limits

---

## Environment-Specific Issues

### Microsoft Fabric

**Issue:** DuckDB, neuralforecast, and PySpark are not available in the Fabric Spark runtime.

**What happens:**
- Override store auto-falls back to Parquet (no concurrent write support)
- Neural models (`nbeats`, `nhits`, `tft`) are unavailable
- `pip install` only persists for the session, not across cluster restarts

**Fix:**
- Use `requirements-fabric.txt` (excludes incompatible packages)
- Pin libraries in Fabric workspace settings for persistent installs
- Use a single orchestrator notebook for override management

### Docker volume mounts

**Issue:** Forecast/metric files aren't visible between API and Streamlit containers.

**Fix:** Both services must share the same data volume:
```yaml
volumes:
  - ./forecasting-platform/data:/app/forecasting-platform/data
```

### Import errors for optional dependencies

**Issue:** `ImportError` or `ModuleNotFoundError` for packages like `neuralforecast`, `shap`, `holidays`.

**What happens:** The platform uses optional dependency guards — it won't crash on import, but the feature will be unavailable at runtime.

**Fix:** Install the specific package:
```bash
pip install neuralforecast    # for N-BEATS, N-HiTS, TFT
pip install shap              # for SHAP explainability
pip install holidays          # for holiday calendar generation
pip install PyJWT             # for JWT authentication
pip install anthropic         # for AI features
```

---

## Next.js Frontend Issues

### Frontend can't connect to API

**Issue:** Pages show "Network Error" or data doesn't load.

**Fix:**
1. Ensure the FastAPI backend is running (`python forecasting-platform/scripts/serve.py --port 8000`)
2. Check `.env.local` has `NEXT_PUBLIC_API_URL=http://localhost:8000`
3. If running on different hosts/ports, check for CORS — the API must allow the frontend's origin

### `npm install` fails

**Issue:** Dependency installation errors.

**Fix:** Ensure Node.js 18+ is installed (`node --version`). Delete `node_modules` and `package-lock.json`, then retry:
```bash
rm -rf node_modules package-lock.json
npm install
```

### Login doesn't work

**Issue:** Login form submits but redirects back or shows error.

**Fix:**
1. Ensure the API is running and `POST /auth/token` returns a response
2. Check that `NEXTAUTH_SECRET` is set in `.env.local` for production
3. In development, NextAuth uses a default secret — verify the API returns valid JWT tokens
