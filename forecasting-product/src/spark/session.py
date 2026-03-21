"""
SparkSession factory with Microsoft Fabric auto-detection.

Usage
-----
>>> from src.spark.session import get_or_create_spark
>>> spark = get_or_create_spark()          # local (dev/test)
>>> spark = get_or_create_spark("fabric")  # Fabric / Synapse
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── Fabric environment marker ────────────────────────────────────────────────
# Microsoft Fabric injects this env-var in every Spark session.
_FABRIC_ENV_VAR = "MSSPARKUTILS_VERSION"
_SYNAPSE_ENV_VAR = "SYNAPSE_SPARKPOOL_NAME"


def _is_fabric() -> bool:
    """Return True when running inside a Fabric / Synapse notebook."""
    return (
        os.environ.get(_FABRIC_ENV_VAR) is not None
        or os.environ.get(_SYNAPSE_ENV_VAR) is not None
    )


def _is_notebookutils_available() -> bool:
    try:
        import notebookutils  # noqa: F401 — Fabric runtime package
        return True
    except ImportError:
        return False


def get_or_create_spark(
    mode: Optional[str] = None,
    app_name: str = "ForecastingPlatform",
    executor_memory: str = "4g",
    executor_cores: int = 2,
    shuffle_partitions: int = 200,
):
    """
    Return an active SparkSession, creating one if necessary.

    Parameters
    ----------
    mode:
        ``"local"``  — forces a local[*] session (unit tests / dev).
        ``"fabric"`` — assumes a running Fabric / Synapse session already
                       exists and returns ``SparkSession.getActiveSession()``.
        ``None``     — auto-detect: Fabric if env-vars present, else local.
    app_name:
        Application name embedded in the Spark UI.
    executor_memory:
        Memory per executor for local mode (ignored in Fabric).
    executor_cores:
        Cores per executor for local mode (ignored in Fabric).
    shuffle_partitions:
        ``spark.sql.shuffle.partitions`` (tune for cluster size).

    Returns
    -------
    pyspark.sql.SparkSession
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise ImportError(
            "PySpark is required for the Spark layer.  "
            "Install it with: pip install pyspark"
        ) from exc

    resolved_mode = mode or ("fabric" if (_is_fabric() or _is_notebookutils_available()) else "local")

    if resolved_mode == "fabric":
        # In Fabric, the session is pre-created by the runtime.
        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError(
                "No active SparkSession found in Fabric environment.  "
                "Ensure you are running inside a Fabric notebook or Spark job."
            )
        logger.info("Using existing Fabric SparkSession: %s", spark.sparkContext.appName)
        return spark

    # ── local mode ────────────────────────────────────────────────────────────
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.executor.memory", executor_memory)
        .config("spark.executor.cores", str(executor_cores))
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        # Delta Lake support (open-source)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # Parquet / date handling
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("Created local SparkSession: %s", spark.sparkContext.appName)
    return spark
