"""
FabricLakehouse — read / write Delta tables in a Microsoft Fabric Lakehouse.

This class wraps the Spark + Delta Lake APIs and provides a clean interface
that is aware of Fabric's OneLake path conventions.

Usage
-----
>>> from src.fabric.config import FabricConfig
>>> from src.fabric.lakehouse import FabricLakehouse
>>>
>>> cfg = FabricConfig.from_env()
>>> lh  = FabricLakehouse(spark, cfg)
>>>
>>> actuals_sdf  = lh.read_table("actuals")
>>> lh.write_table(forecasts_sdf, "forecasts", partition_by=["lob", "forecast_date"])
"""

import logging
import re
from typing import List, Optional

from .config import FabricConfig

logger = logging.getLogger(__name__)

# Characters allowed in Spark SQL partition filters to prevent injection.
# Permits column names, operators, literals, and common SQL keywords.
_SAFE_FILTER_RE = re.compile(
    r"^[\w\s=<>!.'\",()\-]+(?:(?:AND|OR|NOT|IN|IS|NULL|LIKE|BETWEEN)\s+[\w\s=<>!.'\",()\-]+)*$",
    re.IGNORECASE,
)


class FabricLakehouse:
    """
    High-level interface for reading and writing Delta tables in a
    Fabric Lakehouse via ABFSS / OneLake paths.

    Parameters
    ----------
    spark:
        Active SparkSession (Fabric Spark runtime or local Delta session).
    config:
        ``FabricConfig`` with workspace and lakehouse identifiers.
    """

    def __init__(self, spark, config: FabricConfig):
        self.spark = spark
        self.config = config

    # ── read ──────────────────────────────────────────────────────────────────

    def read_table(
        self,
        table_name: str,
        version: Optional[int] = None,
        timestamp: Optional[str] = None,
        date_col: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        partition_filter: Optional[str] = None,
    ):
        """
        Read a Delta table from the Lakehouse with optional partition pruning.

        Partition filters are pushed down to the Delta scan — only the
        matching files are read, which dramatically reduces I/O for large
        tables with years of history.

        Parameters
        ----------
        table_name:
            Name of the Delta table (sub-path under ``Tables/``).
        version:
            Time-travel to a specific Delta version number.
        timestamp:
            Time-travel to a specific timestamp string (ISO-8601).
        date_col:
            Name of the date/timestamp partition column.  Required when
            ``start_date`` or ``end_date`` is supplied.
        start_date:
            Inclusive lower bound (ISO-8601 string, e.g. ``"2022-01-01"``).
            Generates a ``date_col >= 'start_date'`` predicate.
        end_date:
            Exclusive upper bound (ISO-8601 string, e.g. ``"2024-01-01"``).
            Generates a ``date_col < 'end_date'`` predicate.
        partition_filter:
            Arbitrary Spark SQL WHERE predicate for additional partition
            pruning (e.g. ``"lob = 'surface' AND region = 'EMEA'"``).
            Applied in addition to any date bounds above.

        Returns
        -------
        pyspark.sql.DataFrame

        Examples
        --------
        Read only the last two years of actuals:

        >>> lh.read_table("actuals", date_col="Date",
        ...               start_date="2022-01-01", end_date="2024-01-01")

        Read a specific LOB partition:

        >>> lh.read_table("forecasts", partition_filter="lob = 'surface'")
        """
        from pyspark.sql import functions as F

        path = self.config.table_path(table_name)
        reader = self.spark.read.format("delta")

        if version is not None:
            reader = reader.option("versionAsOf", str(version))
            logger.info("Reading Delta table '%s' at version %d", table_name, version)
        elif timestamp is not None:
            reader = reader.option("timestampAsOf", timestamp)
            logger.info("Reading Delta table '%s' at timestamp %s", table_name, timestamp)
        else:
            logger.info("Reading Delta table '%s' (latest)", table_name)

        df = reader.load(path)

        # ── apply partition / date filters ────────────────────────────────────
        filters_applied = []

        if date_col and start_date:
            df = df.filter(F.col(date_col) >= start_date)
            filters_applied.append(f"{date_col} >= {start_date!r}")

        if date_col and end_date:
            df = df.filter(F.col(date_col) < end_date)
            filters_applied.append(f"{date_col} < {end_date!r}")

        if partition_filter:
            if not _SAFE_FILTER_RE.match(partition_filter):
                raise ValueError(
                    f"Rejected potentially unsafe partition_filter: {partition_filter!r}. "
                    "Only simple column comparisons with AND/OR/IN/LIKE are allowed."
                )
            df = df.filter(partition_filter)
            filters_applied.append(partition_filter)

        if filters_applied:
            logger.info(
                "Partition filters applied to '%s': %s",
                table_name, " AND ".join(filters_applied),
            )

        return df

    def read_file(self, subpath: str, format: str = "parquet", **options):
        """
        Read an unmanaged file from the Lakehouse Files root.

        Parameters
        ----------
        subpath:
            Path relative to the ``Files/`` root.
        format:
            ``"parquet"`` | ``"csv"`` | ``"json"`` | ``"delta"``.
        """
        path = self.config.file_path(subpath)
        logger.info("Reading file '%s' (format=%s)", path, format)
        return (
            self.spark.read
            .options(**options)
            .format(format)
            .load(path)
        )

    # ── write ─────────────────────────────────────────────────────────────────

    def write_table(
        self,
        df,
        table_name: str,
        mode: str = "overwrite",
        partition_by: Optional[List[str]] = None,
        merge_schema: bool = True,
    ) -> str:
        """
        Write a Spark DataFrame to a Delta table in the Lakehouse.

        Parameters
        ----------
        df:
            Spark DataFrame to write.
        table_name:
            Target Delta table name (sub-path under ``Tables/``).
        mode:
            ``"overwrite"`` | ``"append"`` | ``"error"`` | ``"ignore"``.
        partition_by:
            List of column names to partition the Delta table by.
        merge_schema:
            Allow schema evolution on existing Delta tables.

        Returns
        -------
        str — the full ABFSS path where the table was written.
        """
        path = self.config.table_path(table_name)
        writer = (
            df.write
            .format("delta")
            .mode(mode)
            .option("mergeSchema", str(merge_schema).lower())
        )

        if partition_by:
            writer = writer.partitionBy(*partition_by)

        if self.config.enable_delta_log_retention:
            writer = (
                writer
                .option("delta.logRetentionDuration",
                        f"interval {self.config.delta_log_retention_days} days")
                .option("delta.deletedFileRetentionDuration",
                        f"interval {self.config.delta_data_retention_days} days")
            )

        logger.info(
            "Writing Delta table '%s' (mode=%s, partitions=%s)", table_name, mode, partition_by
        )
        writer.save(path)
        logger.info("Delta table written to %s", path)
        return path

    def write_file(
        self,
        df,
        subpath: str,
        format: str = "parquet",
        mode: str = "overwrite",
        **options,
    ) -> str:
        """Write a DataFrame to the Lakehouse Files area."""
        path = self.config.file_path(subpath)
        logger.info("Writing file '%s' (format=%s, mode=%s)", path, format, mode)
        df.write.mode(mode).options(**options).format(format).save(path)
        return path

    # ── DDL helpers ───────────────────────────────────────────────────────────

    def table_exists(self, table_name: str) -> bool:
        """Return True if the Delta table directory exists and is a valid Delta table."""
        try:
            from delta.tables import DeltaTable  # type: ignore
            path = self.config.table_path(table_name)
            return DeltaTable.isDeltaTable(self.spark, path)
        except ImportError:
            # Fall back to checking if the path has _delta_log
            try:
                path = self.config.table_path(table_name) + "/_delta_log"
                self.spark.read.format("delta").load(
                    self.config.table_path(table_name)
                ).limit(0)
                return True
            except Exception:
                return False

    def vacuum(self, table_name: str, retention_hours: int = 168) -> None:
        """
        Run VACUUM on a Delta table to remove stale data files.

        Parameters
        ----------
        table_name:
            Target table.
        retention_hours:
            Minimum file age to delete (default: 7 days = 168 h).
        """
        path = self.config.table_path(table_name)
        logger.info("Running VACUUM on '%s' (retention=%d h)", table_name, retention_hours)
        self.spark.sql(
            f"VACUUM delta.`{path}` RETAIN {retention_hours} HOURS"
        )

    def optimize(self, table_name: str, z_order_by: Optional[List[str]] = None) -> None:
        """
        Run OPTIMIZE (and optionally Z-ORDER) on a Delta table.

        Parameters
        ----------
        table_name:
            Target table.
        z_order_by:
            Columns to Z-ORDER by for data skipping.
        """
        path = self.config.table_path(table_name)
        z_order_clause = ""
        if z_order_by:
            cols = ", ".join(z_order_by)
            z_order_clause = f" ZORDER BY ({cols})"
        sql = f"OPTIMIZE delta.`{path}`{z_order_clause}"
        logger.info("Running OPTIMIZE on '%s'%s", table_name, z_order_clause)
        self.spark.sql(sql)

    # ── history ───────────────────────────────────────────────────────────────

    def history(self, table_name: str, limit: int = 10):
        """Return the Delta transaction log history for a table."""
        path = self.config.table_path(table_name)
        return self.spark.sql(f"DESCRIBE HISTORY delta.`{path}` LIMIT {limit}")
