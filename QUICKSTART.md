# Quick Start

Get the forecasting platform running in your browser. Pick one path:

## Path A — Docker (recommended)

No Python setup needed. Just Docker.

```bash
git clone https://github.com/chaitu1385/Forecasting-Platform.git
cd Forecasting-Platform
docker compose up
```

Open **http://localhost:8501** in your browser. That's it.

<!-- screenshot: Streamlit landing page after docker compose up -->

## Path B — Local Python

Requires Python 3.8+.

```bash
git clone https://github.com/chaitu1385/Forecasting-Platform.git
cd Forecasting-Platform
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r forecasting-platform/requirements.txt
streamlit run forecasting-platform/streamlit/app.py
```

Open **http://localhost:8501** in your browser.

<!-- screenshot: Streamlit landing page running locally -->

## What to do next

Once the app is open, follow these four pages in order:

### 1. Data Onboarding

Click **Data Onboarding** in the sidebar. Then click **Use sample data** — no download needed, the platform generates a built-in retail dataset automatically.

You'll see: schema detection (columns, frequency, date range), a forecastability gauge, demand classification, and a recommended model configuration. Click **Download Config YAML** to save it.

<!-- screenshot: Data Onboarding page with forecastability gauge and config -->

### 2. Backtest Results

After running a backtest (`python forecasting-platform/scripts/run_backtest.py`), this page shows the model leaderboard, an FVA cascade chart showing which model layers add or destroy value, and a per-series champion map.

<!-- screenshot: FVA cascade bar chart with ADDS_VALUE / DESTROYS_VALUE annotations -->

### 3. Forecast Viewer

Upload forecast output to see an interactive chart with P10/P90 confidence intervals. Add actuals to overlay historical data and see the seasonal decomposition (trend, seasonal, residual).

<!-- screenshot: Fan chart with actuals overlay -->

### 4. Platform Health

Monitor pipeline runs, drift alerts, data quality, and compute cost. Drift alerts are colour-coded by severity (warning, critical).

<!-- screenshot: Drift alerts table with severity colouring -->

## Optional: AI features

Set `ANTHROPIC_API_KEY` to enable Claude-powered features (natural-language explanations, config recommendations, executive commentary). Everything works without it — AI features degrade gracefully.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Next steps

- [README.md](README.md) — full architecture and module reference
- [CONCEPTS.md](CONCEPTS.md) — why each component exists
- [EDGE_CASES.md](EDGE_CASES.md) — failure modes and how the platform handles them
