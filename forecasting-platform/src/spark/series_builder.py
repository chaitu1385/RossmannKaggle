"""
SparkSeriesBuilder — configurable mapping from raw actuals to the canonical
series schema expected by SparkForecastPipeline.

The canonical schema has three columns:
  series_id  (string)  — unique identifier per time series
  week       (date)    — period start date, truncated to the configured frequency
  quantity   (double)  — forecast target, summed over the period

All LOB-specific column names are driven by ``fabric_config.yaml``
(the ``series_builder`` section), so notebooks and scripts contain zero
hard-coded source column references.

Usage
-----
>>> import yaml
>>> raw = yaml.safe_load(open("configs/fabric_config.yaml"))
>>> builder = SparkSeriesBuilder.from_config(raw["series_builder"])
>>> series_sdf = builder.build(actuals_sdf)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SeriesBuilderConfig:
    """
    Mapping from raw actuals columns to the canonical series schema.

    Attributes
    ----------
    source_id_cols:
        One or more columns whose values are concatenated (with ``_``) to
        form the ``series_id`` string.
        Examples:
          - Rossmann: ``["Store"]``
          - Surface:  ``["product_unit", "country", "channel"]``
    date_col:
        Name of the date column in the raw actuals table.
    target_col:
        Numeric column that is summed to produce ``quantity``.
    filter_expr:
        Optional Spark SQL WHERE predicate applied before aggregation.
        Example: ``"Open = 1"`` (Rossmann) or ``"status = 'shipped'"``.
    frequency:
        ``date_trunc`` argument for the time period.
        ``"week"`` → Monday-aligned ISO weeks.
        ``"month"`` → first day of month.
    """
    source_id_cols: List[str] = field(default_factory=lambda: ["Store"])
    date_col: str = "Date"
    target_col: str = "Sales"
    filter_expr: Optional[str] = None
    frequency: str = "week"

    @classmethod
    def from_dict(cls, d: dict) -> "SeriesBuilderConfig":
        return cls(
            source_id_cols=d.get("source_id_cols", ["Store"]),
            date_col=d.get("date_col", "Date"),
            target_col=d.get("target_col", "Sales"),
            filter_expr=d.get("filter_expr") or None,
            frequency=d.get("frequency", "week"),
        )


class SparkSeriesBuilder:
    """
    Transforms a raw actuals Spark DataFrame into the canonical
    [series_id, week, quantity] panel format.

    Parameters
    ----------
    config:
        ``SeriesBuilderConfig`` describing the column mapping.
    """

    def __init__(self, config: SeriesBuilderConfig):
        self.config = config

    @classmethod
    def from_config(cls, cfg_dict: dict) -> "SparkSeriesBuilder":
        """Build from the ``series_builder`` dict in fabric_config.yaml."""
        return cls(SeriesBuilderConfig.from_dict(cfg_dict))

    def build(self, df):
        """
        Apply filter, build ``series_id``, truncate to period, and aggregate.

        Parameters
        ----------
        df:
            Raw actuals Spark DataFrame.

        Returns
        -------
        Spark DataFrame with columns [series_id, <frequency>, quantity],
        where the time column name matches ``platform_config.forecast.time_column``
        (default: ``"week"``).
        """
        from pyspark.sql import functions as F

        cfg = self.config

        # ── 1. Optional row filter ─────────────────────────────────────────
        if cfg.filter_expr:
            df = df.filter(cfg.filter_expr)
            logger.debug("Applied filter: %s", cfg.filter_expr)

        # ── 2. Build series_id from one or more source columns ─────────────
        if len(cfg.source_id_cols) == 1:
            df = df.withColumn("series_id", F.col(cfg.source_id_cols[0]).cast("string"))
        else:
            # Concatenate multiple columns with "_" separator
            concat_expr = F.concat_ws(
                "_", *[F.col(c).cast("string") for c in cfg.source_id_cols]
            )
            df = df.withColumn("series_id", concat_expr)

        logger.debug("series_id built from: %s", cfg.source_id_cols)

        # ── 3. Truncate date to configured frequency ───────────────────────
        time_col = cfg.frequency  # "week" or "month" → becomes the output column name
        df = df.withColumn(time_col, F.date_trunc(cfg.frequency, F.col(cfg.date_col)))

        # ── 4. Aggregate target to period grain ────────────────────────────
        series_sdf = (
            df
            .groupby("series_id", time_col)
            .agg(F.sum(F.col(cfg.target_col)).alias("quantity"))
            .orderBy("series_id", time_col)
        )

        n_series = series_sdf.select("series_id").distinct().count()
        n_rows   = series_sdf.count()
        logger.info(
            "SparkSeriesBuilder.build: %d series, %d rows (filter=%r, id_cols=%s)",
            n_series, n_rows, cfg.filter_expr, cfg.source_id_cols,
        )
        return series_sdf
