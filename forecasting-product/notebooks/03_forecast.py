# Fabric Notebook: 03 — Distributed Production Forecast
# =======================================================
# Run this notebook (typically weekly) to:
#   1. Read the champion model from the leaderboard table.
#   2. Fit the champion on all available actuals.
#   3. Generate forecasts for the full horizon.
#   4. Write forecasts to the Lakehouse Delta table (upsert).
#   5. (Optional) Optimize the forecasts table for BI query performance.
#
# Prerequisites: 01_data_prep.py and 02_backtest.py must have run.

# %% [markdown]
# ## 0 — Configuration

# %%
import os
import sys
import yaml
from datetime import date

PLATFORM_ROOT = "/lakehouse/default/Files/forecasting-product"
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)

WORKSPACE         = os.environ.get("FABRIC_WORKSPACE", "my-workspace")
LAKEHOUSE         = os.environ.get("FABRIC_LAKEHOUSE", "my-lakehouse")
LOB               = os.environ.get("FORECAST_LOB", "rossmann")
CONFIG_PATH       = os.environ.get("FORECAST_CONFIG", "configs/platform_config.yaml")
ENVIRONMENT       = os.environ.get("FABRIC_ENVIRONMENT", "development")
FABRIC_CFG_PATH   = os.environ.get("FABRIC_CONFIG", "configs/fabric_config.yaml")
# Override champion model (empty = read from leaderboard table)
CHAMPION_OVERRIDE = os.environ.get("CHAMPION_MODEL", "")

with open(FABRIC_CFG_PATH) as _f:
    fabric_yaml = yaml.safe_load(_f)

FORECAST_ORIGIN = date.today().isoformat()

print(f"Workspace       : {WORKSPACE}")
print(f"LOB             : {LOB}")
print(f"Forecast origin : {FORECAST_ORIGIN}")
print(f"Champion override: {CHAMPION_OVERRIDE or '(from leaderboard)'}")

# %% [markdown]
# ## 1 — Spark Session

# %%
from src.spark.session import get_or_create_spark

spark = get_or_create_spark(app_name=f"ForecastingPlatform-Forecast-{LOB}")
spark.sparkContext.setLogLevel("WARN")
print(f"Spark version: {spark.version}")

# %% [markdown]
# ## 2 — Platform config

# %%
from src.config.loader import load_config

config = load_config(CONFIG_PATH)
config.lob = LOB
print(f"Horizon : {config.forecast.horizon_weeks} weeks")

# %% [markdown]
# ## 3 — Fabric Lakehouse client

# %%
from src.fabric.config import FabricConfig
from src.fabric.lakehouse import FabricLakehouse

fabric_cfg = FabricConfig(workspace=WORKSPACE, lakehouse=LAKEHOUSE, environment=ENVIRONMENT)
lh = FabricLakehouse(spark, fabric_cfg)

# %% [markdown]
# ## 4 — Resolve champion model

# %%
from pyspark.sql import functions as F

if CHAMPION_OVERRIDE:
    champion_model = CHAMPION_OVERRIDE
    print(f"Using overridden champion: {champion_model}")
else:
    leaderboard_sdf = lh.read_table("leaderboard")
    champion_model = (
        leaderboard_sdf
        .filter(F.col("lob") == LOB)
        .orderBy(F.col("run_date").desc(), F.col("rank").asc())
        .select("champion_model")
        .limit(1)
        .collect()[0][0]
    )
    print(f"Champion model from leaderboard: {champion_model}")

# %% [markdown]
# ## 5 — Load actuals and build canonical series panel
#
# Column mapping is read from fabric_config.yaml → series_builder.

# %%
from src.spark.series_builder import SparkSeriesBuilder

actuals_raw_sdf = lh.read_table("actuals")

builder = SparkSeriesBuilder.from_config(fabric_yaml["series_builder"])
actuals_series_sdf = builder.build(actuals_raw_sdf)

print(f"Actuals rows   : {actuals_series_sdf.count():,}")
print(f"Series count   : {actuals_series_sdf.select('series_id').distinct().count():,}")
actuals_series_sdf.show(5)

# %% [markdown]
# ## 6 — Run distributed forecast

# %%
from src.spark.pipeline import SparkForecastPipeline

pipeline = SparkForecastPipeline(spark, config)

forecasts_sdf = pipeline.run_forecast(
    actuals_sdf=actuals_series_sdf,
    champion_model=champion_model,
    horizon=config.forecast.horizon_weeks,
)
forecasts_sdf.cache()
print(f"Forecast rows: {forecasts_sdf.count():,}")
forecasts_sdf.show(10)

# %% [markdown]
# ## 7 — Write forecasts to Lakehouse

# %%
from src.fabric.delta_writer import DeltaWriter

writer = DeltaWriter(spark, fabric_cfg)

writer.write_forecasts(
    df=forecasts_sdf,
    lob=LOB,
    forecast_origin=FORECAST_ORIGIN,
    mode="upsert",
)
print(f"Forecasts written: lob={LOB}, origin={FORECAST_ORIGIN}")

# %% [markdown]
# ## 8 — Optimize forecasts table for BI

# %%
lh.optimize(
    "forecasts",
    z_order_by=["series_id", "week"],
)
print("OPTIMIZE complete.")

# %% [markdown]
# ## 9 — Spot-check: sample forecast output

# %%
(
    lh.read_table("forecasts")
    .filter((F.col("lob") == LOB) & (F.col("forecast_origin") == FORECAST_ORIGIN))
    .orderBy("series_id", "week")
    .show(20)
)

# %% [markdown]
# ## 10 — Summary

# %%
forecast_count = (
    lh.read_table("forecasts")
    .filter((F.col("lob") == LOB) & (F.col("forecast_origin") == FORECAST_ORIGIN))
    .count()
)
print("=" * 60)
print(f"LOB             : {LOB}")
print(f"Champion model  : {champion_model}")
print(f"Forecast origin : {FORECAST_ORIGIN}")
print(f"Forecast rows   : {forecast_count:,}")
print("Forecast notebook complete.")
