"""
DeltaWriter — upsert (MERGE) and overwrite helpers for Delta Lake tables.

Provides a higher-level API on top of ``FabricLakehouse.write_table`` for
common write patterns used by the forecasting product:

  * **upsert**    — MERGE INTO for idempotent re-runs.
  * **overwrite** — full partition overwrite (replaces one forecast cycle).
  * **append**    — streaming / incremental writes.

Usage
-----
>>> from src.fabric.delta_writer import DeltaWriter
>>> writer = DeltaWriter(spark, config)
>>> writer.upsert(
...     df=forecast_sdf,
...     table_name="forecasts",
...     merge_keys=["series_id", "week", "lob"],
... )
"""

import logging
from typing import List, Optional

from .config import FabricConfig

logger = logging.getLogger(__name__)


class DeltaWriter:
    """
    Opinionated Delta table writer for the forecasting product.

    Parameters
    ----------
    spark:
        Active SparkSession.
    config:
        ``FabricConfig`` with Lakehouse path settings.
    """

    def __init__(self, spark, config: FabricConfig):
        self.spark = spark
        self.config = config

    # ── upsert ────────────────────────────────────────────────────────────────

    def upsert(
        self,
        df,
        table_name: str,
        merge_keys: List[str],
        update_columns: Optional[List[str]] = None,
        partition_by: Optional[List[str]] = None,
    ) -> None:
        """
        MERGE (upsert) new data into an existing Delta table.

        If the target table does not yet exist, it is created via an
        initial ``write`` with ``mode="overwrite"``.

        Parameters
        ----------
        df:
            Source Spark DataFrame.
        table_name:
            Target Delta table name (under ``Tables/``).
        merge_keys:
            Columns used to match source rows to target rows.
        update_columns:
            Columns to update on match.  Defaults to all non-key columns.
        partition_by:
            Partition columns for the initial table creation.
        """
        try:
            from delta.tables import DeltaTable  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "delta-spark is required for upsert operations.  "
                "Install with: pip install delta-spark"
            ) from exc

        path = self.config.table_path(table_name)

        if not DeltaTable.isDeltaTable(self.spark, path):
            logger.info(
                "Target table '%s' not found — creating via initial write.", table_name
            )
            writer = df.write.format("delta").mode("overwrite")
            if partition_by:
                writer = writer.partitionBy(*partition_by)
            writer.save(path)
            return

        target = DeltaTable.forPath(self.spark, path)
        all_cols = df.columns
        update_cols = update_columns or [c for c in all_cols if c not in merge_keys]

        merge_condition = " AND ".join(
            [f"target.{k} = source.{k}" for k in merge_keys]
        )
        update_expr = {f"target.{c}": f"source.{c}" for c in update_cols}
        insert_expr = {c: f"source.{c}" for c in all_cols}

        logger.info(
            "Upserting into '%s' on keys=%s (%d update cols)",
            table_name, merge_keys, len(update_cols),
        )

        (
            target.alias("target")
            .merge(df.alias("source"), merge_condition)
            .whenMatchedUpdate(set=update_expr)
            .whenNotMatchedInsert(values=insert_expr)
            .execute()
        )
        logger.info("Upsert complete for table '%s'", table_name)

    # ── partition overwrite ───────────────────────────────────────────────────

    def overwrite_partition(
        self,
        df,
        table_name: str,
        partition_by: List[str],
    ) -> str:
        """
        Replace specific partitions in a Delta table with new data.

        Uses ``replaceWhere`` so only the partitions present in ``df``
        are overwritten — other partitions are untouched.

        Parameters
        ----------
        df:
            Source Spark DataFrame containing the new partition data.
        table_name:
            Target Delta table.
        partition_by:
            Partition columns.  The distinct values in ``df`` for these
            columns define which partitions are replaced.

        Returns
        -------
        str — ABFSS path of the target table.
        """

        path = self.config.table_path(table_name)

        # Build a replaceWhere predicate from the distinct partition values in df
        replace_conditions = []
        for col in partition_by:
            distinct_vals = [r[0] for r in df.select(col).distinct().collect()]
            if not distinct_vals:
                continue
            # Format as SQL IN clause
            if isinstance(distinct_vals[0], str):
                vals_str = ", ".join(f"'{v}'" for v in distinct_vals)
            else:
                vals_str = ", ".join(str(v) for v in distinct_vals)
            replace_conditions.append(f"{col} IN ({vals_str})")

        replace_where = " AND ".join(replace_conditions) if replace_conditions else None

        writer = (
            df.write
            .format("delta")
            .mode("overwrite")
            .option("mergeSchema", "true")
            .partitionBy(*partition_by)
        )
        if replace_where:
            writer = writer.option("replaceWhere", replace_where)

        logger.info(
            "Overwriting partition(s) in '%s' where: %s", table_name, replace_where
        )
        writer.save(path)
        logger.info("Partition overwrite complete for '%s'", table_name)
        return path

    # ── append ────────────────────────────────────────────────────────────────

    def append(
        self,
        df,
        table_name: str,
        partition_by: Optional[List[str]] = None,
        merge_schema: bool = True,
    ) -> str:
        """
        Append rows to a Delta table (creates it if it does not exist).

        Parameters
        ----------
        df:
            Source Spark DataFrame.
        table_name:
            Target Delta table.
        partition_by:
            Partition columns for initial creation.
        merge_schema:
            Allow schema evolution.

        Returns
        -------
        str — ABFSS path of the target table.
        """
        path = self.config.table_path(table_name)
        writer = (
            df.write
            .format("delta")
            .mode("append")
            .option("mergeSchema", str(merge_schema).lower())
        )
        if partition_by:
            writer = writer.partitionBy(*partition_by)

        logger.info("Appending to Delta table '%s'", table_name)
        writer.save(path)
        logger.info("Append complete for '%s'", table_name)
        return path

    # ── write forecast output ─────────────────────────────────────────────────

    def write_forecasts(
        self,
        df,
        lob: str,
        forecast_origin: str,
        mode: str = "upsert",
    ) -> None:
        """
        Write forecast results with standard partitioning and merge keys.

        The forecasts table is partitioned by ``(lob, forecast_origin)``
        and upserted / overwritten on ``(series_id, week)``.

        Parameters
        ----------
        df:
            Forecast DataFrame with at minimum [series_id, week, forecast].
        lob:
            Line-of-business identifier (added as a literal column if absent).
        forecast_origin:
            ISO-8601 date string for the forecast run (added if absent).
        mode:
            ``"upsert"`` | ``"overwrite_partition"`` | ``"append"``.
        """
        from pyspark.sql import functions as F

        if "lob" not in df.columns:
            df = df.withColumn("lob", F.lit(lob))
        if "forecast_origin" not in df.columns:
            df = df.withColumn("forecast_origin", F.lit(forecast_origin))

        table_name = "forecasts"

        if mode == "upsert":
            self.upsert(
                df,
                table_name,
                merge_keys=["series_id", "week", "lob", "forecast_origin"],
                partition_by=["lob", "forecast_origin"],
            )
        elif mode == "overwrite_partition":
            self.overwrite_partition(df, table_name, partition_by=["lob", "forecast_origin"])
        else:
            self.append(df, table_name, partition_by=["lob", "forecast_origin"])
