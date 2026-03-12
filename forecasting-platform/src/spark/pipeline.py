"""
SparkForecastPipeline — distributed forecast and backtest via pandas_udf.

Design
------
The existing forecasters (``src.forecasting.*``) are Polars/pandas-based
single-node models.  This pipeline fans them out across the Spark cluster
using ``applyInPandas`` (GROUPED_MAP pattern):

  1. Partition the actuals DataFrame by ``series_id``.
  2. Each Spark task picks up one partition (one series) and calls the
     forecaster's ``fit → predict`` cycle locally.
  3. Results are collected back into a single Spark DataFrame.

This avoids large model broadcasts and lets Spark schedule tasks purely
based on data locality.

Usage
-----
>>> from src.spark.pipeline import SparkForecastPipeline
>>> from src.config import load_config
>>> config   = load_config("configs/platform_config.yaml")
>>> pipeline = SparkForecastPipeline(spark, config)
>>> forecasts_sdf = pipeline.run_forecast(actuals_sdf, champion_model="lgbm_direct")
"""

import logging
import pickle
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SparkForecastPipeline:
    """
    Distributed forecast / backtest pipeline built on PySpark.

    Parameters
    ----------
    spark:
        Active SparkSession.
    config:
        ``PlatformConfig`` instance (loaded from YAML).
    """

    def __init__(self, spark, config):
        self.spark = spark
        self.config = config

    # ── forecast ──────────────────────────────────────────────────────────────

    def run_forecast(
        self,
        actuals_sdf,
        champion_model: str = "naive_seasonal",
        horizon: Optional[int] = None,
        num_partitions: Optional[int] = None,
    ):
        """
        Fit a champion model per series and generate forecasts.

        Parameters
        ----------
        actuals_sdf:
            Spark DataFrame with columns
            [series_id_col, time_col, target_col, …].
        champion_model:
            Registered forecaster name (see ``src.forecasting.registry``).
        horizon:
            Forecast horizon in weeks.  Defaults to ``config.forecast.horizon_weeks``.
        num_partitions:
            Number of Spark partitions (≈ parallelism).  Defaults to
            number of distinct series (capped at 2 000).

        Returns
        -------
        Spark DataFrame: [series_id, week, forecast].
        """
        from pyspark.sql import functions as F, types as T
        from .utils import repartition_by_series

        fc = self.config.forecast
        horizon = horizon or fc.horizon_weeks
        id_col = fc.series_id_column
        time_col = fc.time_column
        target_col = fc.target_column

        # Serialise config & model name for broadcast
        config_bytes = self.spark.sparkContext.broadcast(
            pickle.dumps((self.config, champion_model, horizon, id_col, time_col, target_col))
        )

        # Output schema: series_id (string), week (date), forecast (double)
        output_schema = T.StructType([
            T.StructField(id_col,    T.StringType(),  nullable=False),
            T.StructField(time_col,  T.DateType(),    nullable=False),
            T.StructField("forecast", T.DoubleType(), nullable=True),
        ])

        def _forecast_partition(pdf):
            """
            Called once per series partition.  Runs entirely on the executor.
            """
            import polars as pl
            from src.forecasting.registry import registry

            cfg, model_name, h, id_c, time_c, tgt_c = pickle.loads(config_bytes.value)

            if pdf.empty:
                import pandas as pd
                return pd.DataFrame(columns=[id_c, time_c, "forecast"])

            series = pl.from_pandas(pdf)

            forecaster = registry.build(model_name)
            forecaster.fit(series, target_col=tgt_c, time_col=time_c, id_col=id_c)
            result = forecaster.predict(horizon=h, id_col=id_c, time_col=time_c)
            return result.to_pandas()

        actuals_sdf = repartition_by_series(actuals_sdf, id_col, num_partitions)
        forecasts_sdf = actuals_sdf.groupby(id_col).applyInPandas(
            _forecast_partition, schema=output_schema
        )
        logger.info("SparkForecastPipeline.run_forecast launched (model=%s, horizon=%d)", champion_model, horizon)
        return forecasts_sdf

    # ── backtest ──────────────────────────────────────────────────────────────

    def run_backtest(
        self,
        actuals_sdf,
        model_names: Optional[List[str]] = None,
        num_partitions: Optional[int] = None,
    ):
        """
        Run walk-forward backtest for multiple models across all series.

        Each (series × model × fold) is evaluated independently on an
        executor, producing per-series metrics.

        Parameters
        ----------
        actuals_sdf:
            Spark DataFrame with actuals.
        model_names:
            List of model names to evaluate.  Defaults to
            ``config.forecast.forecasters``.
        num_partitions:
            Spark parallelism.

        Returns
        -------
        Spark DataFrame: [series_id, model, fold, wmape, normalized_bias, …].
        """
        from pyspark.sql import functions as F, types as T
        from .utils import repartition_by_series

        fc = self.config.forecast
        bt = self.config.backtest
        id_col = fc.series_id_column
        time_col = fc.time_column
        target_col = fc.target_column
        model_names = model_names or fc.forecasters

        config_bytes = self.spark.sparkContext.broadcast(
            pickle.dumps((self.config, model_names, id_col, time_col, target_col))
        )

        output_schema = T.StructType([
            T.StructField(id_col,            T.StringType(),  nullable=False),
            T.StructField("model",           T.StringType(),  nullable=False),
            T.StructField("fold",            T.IntegerType(), nullable=False),
            T.StructField("wmape",           T.DoubleType(),  nullable=True),
            T.StructField("normalized_bias", T.DoubleType(),  nullable=True),
            T.StructField("mae",             T.DoubleType(),  nullable=True),
        ])

        def _backtest_partition(pdf):
            """Per-series backtest.  Runs on executor."""
            import pandas as pd
            import polars as pl
            import numpy as np
            from src.forecasting.registry import registry

            cfg, models, id_c, time_c, tgt_c = pickle.loads(config_bytes.value)
            bt_cfg = cfg.backtest

            if pdf.empty:
                return pd.DataFrame(columns=[id_c, "model", "fold", "wmape", "normalized_bias", "mae"])

            series = pl.from_pandas(pdf).sort(time_c)
            n = len(series)
            rows = []

            for model_name in models:
                for fold in range(bt_cfg.n_folds):
                    val_end   = n - fold * bt_cfg.val_weeks
                    val_start = val_end - bt_cfg.val_weeks
                    train_end = val_start - bt_cfg.gap_weeks

                    if train_end < 10 or val_start < 0:
                        continue

                    train = series[:train_end]
                    val   = series[val_start:val_end]

                    try:
                        forecaster = registry.build(model_name)
                        forecaster.fit(train, target_col=tgt_c, time_col=time_c, id_col=id_c)
                        preds = forecaster.predict(
                            horizon=bt_cfg.val_weeks,
                            id_col=id_c,
                            time_col=time_c,
                        )

                        actual_vals = val[tgt_c].to_numpy().astype(float)
                        pred_vals   = preds["forecast"].to_numpy().astype(float)
                        min_len     = min(len(actual_vals), len(pred_vals))
                        actual_vals = actual_vals[:min_len]
                        pred_vals   = pred_vals[:min_len]

                        denom = np.sum(np.abs(actual_vals))
                        wmape = float(np.sum(np.abs(actual_vals - pred_vals)) / denom) if denom else None
                        norm_bias = float(np.sum(actual_vals - pred_vals) / denom) if denom else None
                        mae = float(np.mean(np.abs(actual_vals - pred_vals)))

                        series_id_val = pdf[id_c].iloc[0] if id_c in pdf.columns else "unknown"
                        rows.append({
                            id_c:              series_id_val,
                            "model":           model_name,
                            "fold":            fold,
                            "wmape":           wmape,
                            "normalized_bias": norm_bias,
                            "mae":             mae,
                        })
                    except Exception as exc:
                        logger.warning("Backtest failed for series %s model %s fold %d: %s",
                                       pdf[id_c].iloc[0] if id_c in pdf.columns else "?",
                                       model_name, fold, exc)

            return pd.DataFrame(rows) if rows else pd.DataFrame(
                columns=[id_c, "model", "fold", "wmape", "normalized_bias", "mae"]
            )

        actuals_sdf = repartition_by_series(actuals_sdf, id_col, num_partitions)
        results_sdf = actuals_sdf.groupby(id_col).applyInPandas(
            _backtest_partition, schema=output_schema
        )
        logger.info(
            "SparkForecastPipeline.run_backtest launched (models=%s, folds=%d)",
            model_names, self.config.backtest.n_folds,
        )
        return results_sdf

    # ── champion selection ────────────────────────────────────────────────────

    def select_champion(self, backtest_results_sdf, primary_metric: str = "wmape"):
        """
        Aggregate backtest results and select the champion model per LOB.

        Parameters
        ----------
        backtest_results_sdf:
            Output of ``run_backtest``.
        primary_metric:
            Metric column to minimise for champion selection.

        Returns
        -------
        Spark DataFrame: [model, mean_wmape, mean_normalized_bias, mean_mae, rank].
        """
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        leaderboard = (
            backtest_results_sdf
            .groupby("model")
            .agg(
                F.mean("wmape").alias("mean_wmape"),
                F.mean("normalized_bias").alias("mean_normalized_bias"),
                F.mean("mae").alias("mean_mae"),
                F.count("*").alias("n_evals"),
            )
            .withColumn(
                "rank",
                F.rank().over(Window.orderBy(F.col(f"mean_{primary_metric}").asc_nulls_last())),
            )
            .orderBy("rank")
        )
        return leaderboard
