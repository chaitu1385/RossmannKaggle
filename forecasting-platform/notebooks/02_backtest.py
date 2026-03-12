# Fabric Notebook: 02 — Distributed Backtest & Champion Selection
# ================================================================
# Run this notebook after 01_data_prep.py to:
#   1. Read processed actuals from the Lakehouse Delta table.
#   2. Run walk-forward backtest across all configured models in parallel.
#   3. Select the champion model per LOB.
#   4. Write backtest results and leaderboard to the Lakehouse.
#
# Prerequisites: 01_data_prep.py must have run successfully.

# %% [markdown]
# ## 0 — Configuration

# %%
import os
import sys

PLATFORM_ROOT = "/lakehouse/default/Files/forecasting-platform"
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)

WORKSPACE   = os.environ.get("FABRIC_WORKSPACE", "my-workspace")
LAKEHOUSE   = os.environ.get("FABRIC_LAKEHOUSE", "my-lakehouse")
LOB         = os.environ.get("FORECAST_LOB", "rossmann")
CONFIG_PATH = os.environ.get("FORECAST_CONFIG", "configs/platform_config.yaml")
ENVIRONMENT = os.environ.get("FABRIC_ENVIRONMENT", "development")

print(f"Workspace  : {WORKSPACE}")
print(f"LOB        : {LOB}")
print(f"Config     : {CONFIG_PATH}")

# %% [markdown]
# ## 1 — Spark Session

# %%
from src.spark.session import get_or_create_spark

spark = get_or_create_spark(app_name=f"ForecastingPlatform-Backtest-{LOB}")
spark.sparkContext.setLogLevel("WARN")
print(f"Spark version: {spark.version}")

# %% [markdown]
# ## 2 — Load platform config

# %%
from src.config.loader import load_config

config = load_config(CONFIG_PATH)
config.lob = LOB
print(f"Models to evaluate : {config.forecast.forecasters}")
print(f"Backtest folds     : {config.backtest.n_folds}")
print(f"Validation weeks   : {config.backtest.val_weeks}")

# %% [markdown]
# ## 3 — Read actuals from Lakehouse

# %%
from src.fabric.config import FabricConfig
from src.fabric.lakehouse import FabricLakehouse

fabric_cfg = FabricConfig(workspace=WORKSPACE, lakehouse=LAKEHOUSE, environment=ENVIRONMENT)
lh = FabricLakehouse(spark, fabric_cfg)

actuals_sdf = lh.read_table("actuals")
print(f"Actuals loaded: {actuals_sdf.count():,} rows")
actuals_sdf.printSchema()

# %% [markdown]
# ## 4 — Prepare series-level data
#
# The SparkForecastPipeline expects:
#   - A ``series_id`` column (Store + product key).
#   - A ``week`` column (weekly period date).
#   - A ``quantity`` column (target).

# %%
from pyspark.sql import functions as F

# For Rossmann: series_id = Store, week = truncated Date, quantity = Sales
actuals_series_sdf = (
    actuals_sdf
    .filter(F.col("Open") == 1)             # exclude closed days
    .withColumn("series_id", F.col("Store").cast("string"))
    .withColumn("week", F.date_trunc("week", F.col("Date")))
    .groupby("series_id", "week")
    .agg(F.sum("Sales").alias("quantity"))
    .orderBy("series_id", "week")
)

print(f"Series rows: {actuals_series_sdf.count():,}")
print(f"Series count: {actuals_series_sdf.select('series_id').distinct().count():,}")
actuals_series_sdf.show(5)

# %% [markdown]
# ## 5 — Run distributed backtest

# %%
from src.spark.pipeline import SparkForecastPipeline

pipeline = SparkForecastPipeline(spark, config)

backtest_results_sdf = pipeline.run_backtest(
    actuals_sdf=actuals_series_sdf,
    model_names=config.forecast.forecasters,
)
backtest_results_sdf.cache()
print(f"Backtest result rows: {backtest_results_sdf.count():,}")
backtest_results_sdf.show(20)

# %% [markdown]
# ## 6 — Select champion model

# %%
leaderboard_sdf = pipeline.select_champion(
    backtest_results_sdf,
    primary_metric=config.backtest.primary_metric,
)
leaderboard_sdf.show(truncate=False)

champion_model = leaderboard_sdf.filter(F.col("rank") == 1).select("model").collect()[0][0]
print(f"\nChampion model: {champion_model}")

# %% [markdown]
# ## 7 — Write results to Lakehouse

# %%
from src.fabric.delta_writer import DeltaWriter
from pyspark.sql import functions as F as _F

writer = DeltaWriter(spark, fabric_cfg)

# Backtest results
from datetime import date
run_date = date.today().isoformat()

backtest_results_sdf_out = (
    backtest_results_sdf
    .withColumn("lob", _F.lit(LOB))
    .withColumn("run_date", _F.lit(run_date))
)
writer.append(
    backtest_results_sdf_out,
    table_name="backtest_results",
    partition_by=["lob", "run_date"],
)
print("Backtest results written.")

# Leaderboard
leaderboard_out = (
    leaderboard_sdf
    .withColumn("lob", _F.lit(LOB))
    .withColumn("run_date", _F.lit(run_date))
    .withColumn("champion_model", _F.lit(champion_model))
)
writer.upsert(
    leaderboard_out,
    table_name="leaderboard",
    merge_keys=["lob", "run_date", "model"],
)
print("Leaderboard written.")

# %% [markdown]
# ## 8 — Summary

# %%
print("=" * 60)
print(f"LOB            : {LOB}")
print(f"Run date       : {run_date}")
print(f"Champion model : {champion_model}")
leaderboard_sdf.show(truncate=False)
print("Backtest notebook complete.")
