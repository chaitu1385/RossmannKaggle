# Fabric Notebook: 01 — Data Ingestion & Feature Engineering
# ============================================================
# Run this notebook first to:
#   1. Read raw actuals from the Lakehouse Files area (CSV / Parquet).
#   2. Apply distributed feature engineering with SparkFeatureEngineer.
#   3. Write the processed actuals Delta table to the Lakehouse.
#
# Fabric usage:
#   - Open in a Microsoft Fabric Lakehouse Notebook.
#   - Attach to a Spark pool (Medium or larger recommended).
#   - Set FABRIC_WORKSPACE and FABRIC_LAKEHOUSE env-vars or fill in below.

# %% [markdown]
# ## 0 — Configuration

# %%
import os
import sys

# ── Add platform src to path ─────────────────────────────────────────────────
# In Fabric, mount the repo via the Lakehouse Git integration or upload the
# forecasting-platform/ folder.  Adjust the path below as needed.
PLATFORM_ROOT = "/lakehouse/default/Files/forecasting-platform"
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)

# ── Fabric / Lakehouse identifiers ───────────────────────────────────────────
WORKSPACE  = os.environ.get("FABRIC_WORKSPACE", "my-workspace")
LAKEHOUSE  = os.environ.get("FABRIC_LAKEHOUSE", "my-lakehouse")
LOB        = os.environ.get("FORECAST_LOB", "rossmann")
ENVIRONMENT = os.environ.get("FABRIC_ENVIRONMENT", "development")

print(f"Workspace : {WORKSPACE}")
print(f"Lakehouse : {LAKEHOUSE}")
print(f"LOB       : {LOB}")
print(f"Environment: {ENVIRONMENT}")

# %% [markdown]
# ## 1 — Spark Session

# %%
from src.spark.session import get_or_create_spark

spark = get_or_create_spark(app_name=f"ForecastingPlatform-DataPrep-{LOB}")
spark.sparkContext.setLogLevel("WARN")
print(f"Spark version: {spark.version}")

# %% [markdown]
# ## 2 — Load raw data

# %%
from src.spark.loader import SparkDataLoader
from src.spark.utils import abfss_uri

# Build the ABFSS base path for the Lakehouse
base_path = abfss_uri(WORKSPACE, LAKEHOUSE)
loader = SparkDataLoader(spark, base_path=base_path)

# For local / dev mode, read from Files/raw/
# For Rossmann Kaggle data specifically:
train_sdf, test_sdf, store_sdf = loader.read_rossmann_all()

print(f"Train rows : {train_sdf.count():,}")
print(f"Test rows  : {test_sdf.count():,}")
print(f"Store rows : {store_sdf.count():,}")

# %% [markdown]
# ## 3 — Merge with store metadata

# %%
actuals_sdf = train_sdf.join(store_sdf, on="Store", how="left")
print(f"Actuals after store join: {actuals_sdf.count():,} rows")
actuals_sdf.printSchema()

# %% [markdown]
# ## 4 — Distributed feature engineering

# %%
from src.spark.feature_engineering import SparkFeatureEngineer

eng = SparkFeatureEngineer(
    lag_periods=[1, 7, 14, 30],
    rolling_windows=[7, 14, 30],
)

actuals_features_sdf = eng.fit_transform(
    actuals_sdf,
    date_col="Date",
    target_col="Sales",
    group_col="Store",
)

print(f"Feature columns ({len(actuals_features_sdf.columns)}): {actuals_features_sdf.columns}")
actuals_features_sdf.show(5, truncate=False)

# %% [markdown]
# ## 5 — Quality checks

# %%
from pyspark.sql import functions as F

# Missing values summary
null_counts = actuals_features_sdf.select(
    [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in actuals_features_sdf.columns]
)
print("Null counts per column:")
null_counts.show(truncate=False)

# Date range
actuals_features_sdf.select(
    F.min("Date").alias("min_date"),
    F.max("Date").alias("max_date"),
    F.countDistinct("Store").alias("n_stores"),
).show()

# %% [markdown]
# ## 6 — Write processed actuals to Lakehouse (Delta)

# %%
from src.fabric.config import FabricConfig
from src.fabric.lakehouse import FabricLakehouse

fabric_cfg = FabricConfig(
    workspace=WORKSPACE,
    lakehouse=LAKEHOUSE,
    environment=ENVIRONMENT,
)
lh = FabricLakehouse(spark, fabric_cfg)

lh.write_table(
    actuals_features_sdf,
    table_name="actuals",
    mode="overwrite",
    partition_by=["Store"],
    merge_schema=True,
)

print(f"Actuals Delta table written to: {fabric_cfg.table_path('actuals')}")

# %% [markdown]
# ## 7 — (Optional) Optimize the Delta table

# %%
lh.optimize("actuals", z_order_by=["Date"])
print("OPTIMIZE complete.")
