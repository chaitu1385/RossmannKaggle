# Fabric Notebook: 04 — One-Cell Production Deployment
# =====================================================
# This notebook demonstrates the full forecast cycle using
# FabricNotebookAdapter — a single-class entry point that eliminates
# the boilerplate across notebooks 01-03.
#
# Prerequisites:
#   - Actuals Delta table must exist in the Lakehouse (created by 01_data_prep.py).
#   - Set FABRIC_WORKSPACE, FABRIC_LAKEHOUSE, and FORECAST_LOB env vars
#     (or pass them directly to the adapter).
#
# Usage:
#   1. Open in a Fabric Lakehouse Notebook.
#   2. Attach to a Spark pool (Medium or larger recommended).
#   3. Run the single cell below.

# %% [markdown]
# ## Full Cycle: Load → Backtest → Champion → Forecast → Write → Optimize

# %%
from src.fabric.notebook_adapter import FabricNotebookAdapter

# ── 1. Initialize adapter (auto-detects Fabric env vars) ─────────────────────
adapter = FabricNotebookAdapter(
    config_path="configs/platform_config.yaml",
    fabric_config_path="configs/fabric_config.yaml",
    # Override env vars if needed:
    # lob="rossmann",
    # workspace="my-workspace",
    # lakehouse="my-lakehouse",
)
adapter.summary()

# ── 2. Load actuals from Lakehouse ───────────────────────────────────────────
actuals_sdf = adapter.load_actuals("actuals")
actuals_series = adapter.build_series(actuals_sdf)
print(f"Series count: {actuals_series.select('series_id').distinct().count():,}")

# ── 3. Run backtest and select champion ──────────────────────────────────────
backtest_sdf = adapter.run_backtest(actuals_series)
champion = adapter.select_champion(backtest_sdf)
print(f"Champion model: {champion}")

# ── 4. Generate production forecasts ─────────────────────────────────────────
forecasts_sdf = adapter.run_forecast(actuals_series, champion=champion)
forecasts_sdf.cache()
print(f"Forecast rows: {forecasts_sdf.count():,}")

# ── 5. Write forecasts to Lakehouse ──────────────────────────────────────────
adapter.write_forecasts(forecasts_sdf)

# ── 6. Optimize for BI query performance ─────────────────────────────────────
adapter.optimize_table("forecasts", z_order_by=["series_id", "week"])

# ── 7. Spot-check ────────────────────────────────────────────────────────────
from pyspark.sql import functions as F
from datetime import date

(
    adapter.lakehouse.read_table("forecasts")
    .filter(
        (F.col("lob") == adapter.lob)
        & (F.col("forecast_origin") == date.today().isoformat())
    )
    .orderBy("series_id", "week")
    .show(20)
)

print("=" * 60)
print(f"Deployment complete: lob={adapter.lob}, champion={champion}")
