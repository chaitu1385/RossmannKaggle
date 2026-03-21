"""
SparkDataLoader — distributed data ingestion for the forecasting product.

Supports reading from:
  - Local CSV / Parquet files (dev / unit-test mode)
  - ABFSS paths on OneLake / ADLS Gen2 (Microsoft Fabric / Azure)
  - Delta Lake tables (Fabric Lakehouse or standalone Delta)

Usage
-----
>>> from src.spark.loader import SparkDataLoader
>>> loader = SparkDataLoader(spark, base_path="abfss://my-ws@onelake.dfs.fabric.microsoft.com/lh")
>>> actuals   = loader.read_actuals()
>>> forecasts = loader.read_delta("forecasts/surface_2024-01-01")
"""

import logging

logger = logging.getLogger(__name__)


class SparkDataLoader:
    """
    Reads raw and processed data into Spark DataFrames.

    Parameters
    ----------
    spark:
        Active SparkSession.
    base_path:
        Root path for all datasets.  Can be a local directory
        (``/data/…``) or an ABFSS URI
        (``abfss://<workspace>@onelake.dfs.fabric.microsoft.com/<lakehouse>``).
    """

    def __init__(self, spark, base_path: str):
        self.spark = spark
        self.base_path = base_path.rstrip("/")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _path(self, *parts: str) -> str:
        return "/".join([self.base_path] + list(parts))

    def _read_csv(self, path: str, **options):
        return (
            self.spark.read
            .options(header=True, inferSchema=True, **options)
            .csv(path)
        )

    def _read_parquet(self, path: str):
        return self.spark.read.parquet(path)

    def _read_delta(self, path: str):
        return self.spark.read.format("delta").load(path)

    # ── public API ────────────────────────────────────────────────────────────

    def read_actuals(
        self,
        table_or_subpath: str = "actuals",
        format: str = "delta",
    ):
        """
        Read the canonical actuals table.

        Parameters
        ----------
        table_or_subpath:
            Sub-path under ``base_path`` or a fully-qualified Delta table name.
        format:
            ``"delta"`` | ``"parquet"`` | ``"csv"``.

        Returns
        -------
        pyspark.sql.DataFrame
        """
        path = self._path(table_or_subpath)
        logger.info("Reading actuals from %s (format=%s)", path, format)

        if format == "delta":
            return self._read_delta(path)
        elif format == "parquet":
            return self._read_parquet(path)
        else:
            return self._read_csv(path)

    def read_product_master(
        self,
        table_or_subpath: str = "product_master",
        format: str = "delta",
    ):
        """Read product/SKU master data."""
        path = self._path(table_or_subpath)
        logger.info("Reading product master from %s", path)
        if format == "delta":
            return self._read_delta(path)
        elif format == "parquet":
            return self._read_parquet(path)
        return self._read_csv(path)

    def read_delta(self, subpath: str):
        """Generic Delta table reader relative to ``base_path``."""
        path = self._path(subpath)
        logger.info("Reading Delta table: %s", path)
        return self._read_delta(path)

    def read_parquet(self, subpath: str):
        """Generic Parquet reader relative to ``base_path``."""
        path = self._path(subpath)
        logger.info("Reading Parquet: %s", path)
        return self._read_parquet(path)

    def read_csv(self, subpath: str, **options):
        """Generic CSV reader relative to ``base_path``."""
        path = self._path(subpath)
        logger.info("Reading CSV: %s", path)
        return self._read_csv(path, **options)

    # ── Rossmann-specific helpers (dev / Kaggle data) ─────────────────────────

    def read_rossmann_train(self):
        """Read Rossmann train.csv (local dev)."""
        return self._read_csv(self._path("train.csv"))

    def read_rossmann_test(self):
        """Read Rossmann test.csv (local dev)."""
        return self._read_csv(self._path("test.csv"))

    def read_rossmann_store(self):
        """Read Rossmann store.csv (local dev)."""
        return self._read_csv(self._path("store.csv"))

    def read_rossmann_all(self):
        """
        Load all three Rossmann datasets.

        Returns
        -------
        (train_df, test_df, store_df)
        """
        train = self.read_rossmann_train()
        test = self.read_rossmann_test()
        store = self.read_rossmann_store()
        return train, test, store
