"""
Spark / Microsoft Fabric layer for the forecasting platform.

This sub-package provides distributed execution of the forecasting
pipeline on Apache Spark (including Microsoft Fabric Spark).

Modules
-------
session             SparkSession factory with Fabric auto-detection.
loader              SparkDataLoader — reads CSV / Parquet / Delta Lake.
feature_engineering SparkFeatureEngineer — PySpark-native transformations.
pipeline            SparkForecastPipeline — pandas_udf distributed runner.
utils               Schema helpers and Polars ↔ Spark conversion utilities.
"""

from .session import get_or_create_spark  # noqa: F401
from .loader import SparkDataLoader  # noqa: F401
from .feature_engineering import SparkFeatureEngineer  # noqa: F401
from .pipeline import SparkForecastPipeline  # noqa: F401
