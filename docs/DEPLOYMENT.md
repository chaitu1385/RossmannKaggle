# Deployment Guide

How to run the Forecasting Platform — from local development to production deployment.

---

## Quick Start: Docker Compose

The fastest path to a running platform:

```bash
git clone https://github.com/chaitu1385/Forecasting-Platform.git
cd Forecasting-Platform
docker compose up
```

This starts two services:

| Service | Port | URL |
|---------|------|-----|
| REST API (FastAPI) | 8000 | http://localhost:8000/docs (Swagger UI) |
| Dashboard (Streamlit) | 8501 | http://localhost:8501 |

Both services share a data volume at `./forecasting-platform/data`.

### Docker Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_DATA_DIR` | `forecasting-platform/data/` | Root data directory for the API |
| `API_METRICS_DIR` | `forecasting-platform/data/metrics/` | Metric store location |
| `ANTHROPIC_API_KEY` | (empty) | Claude API key for AI features (optional) |

Set AI features:
```bash
ANTHROPIC_API_KEY=sk-ant-... docker compose up
```

### Docker Health Check

The API service has a built-in health check hitting `GET /health` every 30 seconds. The Streamlit service waits for the API to be healthy before starting.

---

## Local Python Setup

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
# Full install (all features)
pip install -r forecasting-platform/requirements.txt

# Fabric-compatible subset (no DuckDB, PySpark, neuralforecast)
pip install -r forecasting-platform/requirements-fabric.txt
```

### Start the API Server

```bash
python forecasting-platform/scripts/serve.py --port 8000 --data-dir data/
```

CLI arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Bind port |
| `--data-dir` | `data/` | Root data directory |
| `--metrics-dir` | `data/metrics/` | Metric store directory |
| `--reload` | off | Enable hot-reload (development only) |
| `--workers` | `1` | Number of Uvicorn worker processes |

Environment variables `API_DATA_DIR` and `API_METRICS_DIR` override their CLI counterparts.

### Start the Dashboard

```bash
streamlit run forecasting-platform/streamlit/app.py
```

Opens at http://localhost:8501.

---

## Running Pipelines

### Backtest Pipeline

Evaluates all configured models via walk-forward cross-validation and selects a champion:

```bash
python forecasting-platform/scripts/run_backtest.py \
  --config forecasting-platform/configs/platform_config.yaml \
  --lob retail
```

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | Yes | Path to platform config YAML |
| `--lob-override` | No | LOB-specific config override YAML |
| `--data` | Yes | Path to actuals (Parquet or CSV) |
| `--product-master` | No | Path to product master table |
| `--mapping-table` | No | Path to SKU mapping table |

### Forecast Pipeline

Generates forecasts using the champion model:

```bash
python forecasting-platform/scripts/run_forecast.py \
  --config forecasting-platform/configs/platform_config.yaml \
  --lob retail
```

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | Yes | Path to platform config YAML |
| `--lob-override` | No | LOB-specific config override YAML |
| `--data` | Yes | Path to actuals (Parquet or CSV) |
| `--champion` | No | Champion model name (default: `naive_seasonal`) |
| `--product-master` | No | Path to product master table |
| `--mapping-table` | No | Path to SKU mapping table |

---

## Configuration

The platform uses a YAML config system with layered overrides.

### Config Files

| File | Purpose |
|------|---------|
| `configs/base_config.yaml` | Platform defaults |
| `configs/platform_config.yaml` | Standard deployment config |
| `configs/fabric_config.yaml` | Microsoft Fabric settings |
| `configs/lob/<name>.yaml` | Line-of-business overrides (inherit from base) |

### Key Settings to Customize

```yaml
# What to forecast
forecast:
  frequency: W                          # D | W | M | Q
  horizon_weeks: 39                     # forecast horizon (periods)
  forecasters: [lgbm_direct, auto_ets, seasonal_naive]
  quantiles: [0.1, 0.5, 0.9]           # prediction intervals

# How to evaluate
backtest:
  n_folds: 3                            # walk-forward CV folds
  val_weeks: 13                          # validation window
  champion_granularity: lob              # lob | product_group | series
  selection_strategy: champion           # champion | weighted_ensemble

# Data quality
data_quality:
  validation:
    enabled: true                        # schema + duplicate checks
    strict: false                        # true = halt on errors
  cleansing:
    enabled: true                        # outlier + stockout correction
    outlier_method: iqr                  # iqr | zscore

# Performance
parallelism:
  backend: local                         # local | spark
  n_workers: -1                          # -1 = all CPU cores
  batch_size: 0                          # 0 = all series at once
```

See [DATA_FORMAT.md](DATA_FORMAT.md) for complete config schema details.

---

## Authentication Setup

Authentication is disabled by default for development. To enable:

### 1. Enable Auth in Config

The API checks if auth is enabled. When disabled, all requests are treated as admin.

### 2. Create Tokens

```bash
# Via the API
curl -X POST "http://localhost:8000/auth/token?username=analyst&role=data_scientist"
# Returns: {"access_token": "eyJ...", "token_type": "bearer"}
```

### 3. Use Tokens

```bash
curl -H "Authorization: Bearer eyJ..." http://localhost:8000/forecast/retail
```

### Role Permissions

| Action | admin | data_scientist | planner | manager | viewer |
|--------|-------|----------------|---------|---------|--------|
| View forecasts/metrics | Y | Y | Y | Y | Y |
| Create overrides | Y | Y | Y | - | - |
| Approve overrides | Y | - | - | Y | - |
| Run backtest/pipeline | Y | Y | - | - | - |
| Promote champion model | Y | Y | - | - | - |
| View audit log | Y | Y | - | Y | - |
| Manage users | Y | - | - | - | - |

Required package: `pip install PyJWT`

---

## Observability Setup

### Logging

```yaml
observability:
  log_format: json       # "text" for human-readable, "json" for structured
  log_level: INFO         # DEBUG | INFO | WARNING | ERROR
```

JSON logs include `run_id`, `lob`, and timestamps for every pipeline operation.

### Metrics

```yaml
observability:
  metrics_backend: log      # "log" = JSON lines, "statsd" = UDP to StatsD
  statsd_host: localhost
  statsd_port: 8125
  metrics_prefix: forecast_platform
```

### Drift Alerts

```yaml
observability:
  alerts:
    channels: [log, webhook]
    webhook_url: "https://hooks.slack.com/services/..."    # Slack, Teams, PagerDuty
    min_severity: warning    # "warning" | "critical"
```

Alerts fire when forecast accuracy drifts beyond configured thresholds. Start with `min_severity: critical` and lower to `warning` once you've tuned thresholds for your data.

### Cost Tracking

```yaml
observability:
  cost_per_second: 0.0001    # $/second for compute cost estimation
```

Tracks per-model compute time and estimates cloud costs per run.

---

## Scaling

### Local (ProcessPool)

```yaml
parallelism:
  backend: local
  n_workers: -1              # -1 = all CPU cores
  batch_size: 500            # chunk series into batches of 500
```

- Reduce `n_workers` if hitting memory limits
- Increase `batch_size` to reduce memory per worker (each worker loads its batch + model)

### PySpark (Distributed)

For large-scale runs on Databricks or Fabric:

```bash
python forecasting-platform/scripts/spark_forecast.py \
  --config forecasting-platform/configs/platform_config.yaml
```

Uses `SparkForecastPipeline` for distributed series processing. Configure via:

```yaml
parallelism:
  backend: spark
```

---

## Microsoft Fabric Deployment

### Prerequisites

- Fabric workspace with a Lakehouse
- `requirements-fabric.txt` installed in notebook environment

### Limitations

Fabric's Spark runtime does **not** include:
- DuckDB (override store auto-falls back to Parquet)
- neuralforecast / PyTorch (neural models unavailable)
- FastAPI (API runs externally, not in Fabric)

### Notebook Deployment

```python
from src.fabric.deployment import DeploymentOrchestrator, DeploymentConfig

cfg = DeploymentConfig(
    lob="retail",
    workspace="my-workspace",
    lakehouse="my-lakehouse",
    max_staleness_days=14,     # actuals must be within 14 days
    write_mode="upsert",       # "upsert" | "overwrite_partition" | "append"
)

orch = DeploymentOrchestrator(spark, config=platform_config, deploy_config=cfg)
result = orch.run(actuals_sdf=actuals_spark_df)
```

The orchestrator runs pre-flight checks (data freshness, schema), backtests (optional), forecasts, writes to Delta tables, and logs to the deploy audit table.

### Fabric Config

```yaml
# configs/fabric_config.yaml
fabric:
  workspace: my-workspace
  lakehouse: my-lakehouse
```

---

## Production Checklist

Before going to production, verify:

- [ ] **Auth enabled** — JWT authentication active, secrets rotated
- [ ] **HTTPS** — TLS termination configured (reverse proxy or load balancer)
- [ ] **Logging** — `log_format: json` for structured log aggregation
- [ ] **Alerts** — Webhook configured for drift alerts (Slack/Teams/PagerDuty)
- [ ] **Data validation** — `validation.enabled: true` with `strict: true`
- [ ] **Demand cleansing** — `cleansing.enabled: true` for outlier/stockout handling
- [ ] **Parallelism** — `n_workers` tuned for available CPU/memory
- [ ] **Batch size** — Set `batch_size > 0` for memory-constrained environments
- [ ] **Monitoring** — Health check endpoint (`/health`) monitored
- [ ] **Backups** — Metric store and audit log directories backed up
- [ ] **Cost tracking** — `cost_per_second` set if tracking cloud spend
- [ ] **API key** — `ANTHROPIC_API_KEY` set if using AI features
