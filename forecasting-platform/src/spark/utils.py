"""
Schema helpers and Polars ↔ Spark conversion utilities.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


# ── Polars → Spark type mapping ───────────────────────────────────────────────

def polars_to_spark_schema(polars_df):
    """
    Convert a Polars DataFrame schema to a PySpark StructType.

    Parameters
    ----------
    polars_df:
        polars.DataFrame whose schema to convert.

    Returns
    -------
    pyspark.sql.types.StructType
    """
    import polars as pl
    from pyspark.sql import types as T

    _TYPE_MAP = {
        pl.Int8:    T.ByteType(),
        pl.Int16:   T.ShortType(),
        pl.Int32:   T.IntegerType(),
        pl.Int64:   T.LongType(),
        pl.UInt8:   T.ShortType(),
        pl.UInt16:  T.IntegerType(),
        pl.UInt32:  T.LongType(),
        pl.UInt64:  T.LongType(),
        pl.Float32: T.FloatType(),
        pl.Float64: T.DoubleType(),
        pl.Boolean: T.BooleanType(),
        pl.Utf8:    T.StringType(),
        pl.String:  T.StringType(),
        pl.Date:    T.DateType(),
        pl.Datetime: T.TimestampType(),
    }

    fields = []
    for col_name, dtype in zip(polars_df.columns, polars_df.dtypes):
        spark_type = _TYPE_MAP.get(type(dtype), T.StringType())
        fields.append(T.StructField(col_name, spark_type, nullable=True))

    return T.StructType(fields)


def polars_to_spark(polars_df, spark):
    """
    Convert a Polars DataFrame to a PySpark DataFrame.

    Uses an intermediate pandas conversion for correctness across all types.
    """
    return spark.createDataFrame(polars_df.to_pandas())


def spark_to_polars(spark_df):
    """
    Convert a PySpark DataFrame to a Polars DataFrame.
    """
    import polars as pl
    return pl.from_pandas(spark_df.toPandas())


# ── Partition utilities ────────────────────────────────────────────────────────

def repartition_by_series(df, series_id_col: str = "series_id", num_partitions: Optional[int] = None):
    """
    Repartition a Spark DataFrame so that each series lands on one partition.

    This is important for the ``pandas_udf`` pattern: each task processes
    exactly one series, avoiding cross-partition shuffles in the UDF.

    Parameters
    ----------
    df:
        Spark DataFrame containing ``series_id_col``.
    series_id_col:
        Column that identifies each time series.
    num_partitions:
        Explicit partition count.  If None, defaults to the number of
        distinct series (capped at 2 000 to avoid excessive overhead).

    Returns
    -------
    Repartitioned Spark DataFrame.
    """
    from pyspark.sql import functions as F

    if num_partitions is None:
        n_series = df.select(F.countDistinct(series_id_col)).collect()[0][0]
        num_partitions = min(int(n_series), 2000)

    logger.info(
        "Repartitioning to %d partitions on '%s'", num_partitions, series_id_col
    )
    return df.repartition(num_partitions, series_id_col)


# ── ABFSS URI builder ─────────────────────────────────────────────────────────

def abfss_uri(
    workspace: str,
    lakehouse: str,
    subpath: str = "",
    host: str = "onelake.dfs.fabric.microsoft.com",
) -> str:
    """
    Build an ABFSS URI for a Microsoft Fabric Lakehouse path.

    Parameters
    ----------
    workspace:
        Fabric workspace name (or GUID).
    lakehouse:
        Lakehouse name (or GUID).
    subpath:
        Optional sub-path within the Lakehouse (e.g. "Tables/forecasts").
    host:
        OneLake DFS endpoint.

    Returns
    -------
    ``abfss://<workspace>@<host>/<lakehouse>/<subpath>``

    Example
    -------
    >>> abfss_uri("myws", "myLH", "Tables/actuals")
    'abfss://myws@onelake.dfs.fabric.microsoft.com/myLH/Tables/actuals'
    """
    base = f"abfss://{workspace}@{host}/{lakehouse}"
    if subpath:
        return f"{base}/{subpath.lstrip('/')}"
    return base
