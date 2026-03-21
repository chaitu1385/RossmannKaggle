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
import yaml

# ── Add platform src to path ─────────────────────────────────────────────────
# In Fabric, mount the repo via the Lakehouse Git integration or upload the
# forecasting-product/ folder.  Adjust the path below as needed.
PLATFORM_ROOT = "/lakehouse/default/Files/forecasting-product"
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)

# ── Fabric / Lakehouse identifiers ───────────────────────────────────────────
WORKSPACE       = os.environ.get("FABRIC_WORKSPACE", "my-workspace")
LAKEHOUSE       = os.environ.get("FABRIC_LAKEHOUSE", "my-lakehouse")
LOB             = os.environ.get("FORECAST_LOB", "rossmann")
ENVIRONMENT     = os.environ.get("FABRIC_ENVIRONMENT", "development")
FABRIC_CFG_PATH = os.environ.get("FABRIC_CONFIG", "configs/fabric_config.yaml")

with open(FABRIC_CFG_PATH) as _f:
    fabric_yaml = yaml.safe_load(_f)

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

base_path = abfss_uri(WORKSPACE, LAKEHOUSE)
loader = SparkDataLoader(spark, base_path=base_path)

# Rossmann Kaggle data
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
#
# Column names are read from fabric_config.yaml → series_builder, so this
# cell works without changes for any LOB.

# %%
from src.spark.feature_engineering import SparkFeatureEngineer

sb_cfg = fabric_yaml["series_builder"]
feat_cfg = fabric_yaml.get("features", {})

eng = SparkFeatureEngineer(
    lag_periods=feat_cfg.get("lag_periods", [1, 7, 14, 30]),
    rolling_windows=feat_cfg.get("rolling_windows", [7, 14, 30]),
)

actuals_features_sdf = eng.fit_transform(
    actuals_sdf,
    date_col=sb_cfg["date_col"],
    target_col=sb_cfg["target_col"],
    group_col=sb_cfg["source_id_cols"][0],   # primary grouping key for rolling/lag
)

print(f"Feature columns ({len(actuals_features_sdf.columns)}): {actuals_features_sdf.columns}")
actuals_features_sdf.show(5, truncate=False)

# %% [markdown]
# ## 5 — Quality checks

# %%
from pyspark.sql import functions as F

date_col   = sb_cfg["date_col"]
target_col = sb_cfg["target_col"]
id_col     = sb_cfg["source_id_cols"][0]

# Missing values summary
null_counts = actuals_features_sdf.select(
    [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in actuals_features_sdf.columns]
)
print("Null counts per column:")
null_counts.show(truncate=False)

# Date range and series count
actuals_features_sdf.select(
    F.min(date_col).alias("min_date"),
    F.max(date_col).alias("max_date"),
    F.countDistinct(id_col).alias("n_series"),
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
    partition_by=[id_col],
    merge_schema=True,
)

print(f"Actuals Delta table written to: {fabric_cfg.table_path('actuals')}")

# %% [markdown]
# ## 7 — (Optional) Optimize the Delta table

# %%
lh.optimize("actuals", z_order_by=[date_col])
print("OPTIMIZE complete.")
