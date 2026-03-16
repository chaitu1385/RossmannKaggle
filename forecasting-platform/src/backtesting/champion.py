"""
Champion model selection.

After backtesting, selects the best model based on the primary metric
(WMAPE by default) at the configured granularity (per-LOB to start).

Supports multi-horizon selection: different champion models per forecast
horizon bucket (e.g. short/medium/long term).

The champion table is consumed by the inference pipeline to decide
which model generates the final forecast.
"""

import logging
from typing import Dict, List, Optional

import polars as pl

from ..config.schema import BacktestConfig, HorizonBucket

logger = logging.getLogger(__name__)


class ChampionSelector:
    """
    Select the best-performing model from backtesting results.

    Supports selection at multiple granularities:
      - "lob": one champion for the entire LOB (default)
      - "product_group": one champion per product group
      - "series": one champion per individual series (most granular)
    """

    def __init__(self, config: BacktestConfig):
        self.primary_metric = config.primary_metric
        self.secondary_metric = config.secondary_metric
        self.granularity = config.champion_granularity

    def select(
        self,
        backtest_results: pl.DataFrame,
        granularity_col: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Select champion model(s) from backtest results.

        Parameters
        ----------
        backtest_results:
            Output from BacktestEngine.run().
        granularity_col:
            Column to group by for champion selection.
            Overrides the config-level granularity.

        Returns
        -------
        DataFrame with columns: [group_key, model_id, primary_metric,
        secondary_metric].  One row per group.
        """
        if backtest_results.is_empty():
            logger.warning("Empty backtest results; no champion to select.")
            return pl.DataFrame(schema={
                "group_key": pl.Utf8,
                "model_id": pl.Utf8,
                self.primary_metric: pl.Float64,
                self.secondary_metric: pl.Float64,
            })

        # Determine grouping
        if granularity_col and granularity_col in backtest_results.columns:
            group_col = granularity_col
        elif self.granularity == "lob" and "lob" in backtest_results.columns:
            group_col = "lob"
        elif self.granularity == "series" and "series_id" in backtest_results.columns:
            group_col = "series_id"
        else:
            # Global champion
            group_col = None

        if group_col:
            leaderboard = (
                backtest_results
                .group_by([group_col, "model_id"])
                .agg([
                    pl.col(self.primary_metric).mean().alias(self.primary_metric),
                    pl.col(self.secondary_metric).mean().alias(self.secondary_metric),
                ])
            )

            # For WMAPE: lower is better.  For bias: closer to 0 is better.
            # Select by primary metric (ascending), break ties by abs(bias).
            champions = (
                leaderboard
                .with_columns(
                    pl.col(self.secondary_metric).abs().alias("_abs_bias")
                )
                .sort([group_col, self.primary_metric, "_abs_bias"])
                .group_by(group_col)
                .first()
                .drop("_abs_bias")
                .rename({group_col: "group_key"})
            )
        else:
            # Global: one champion across everything
            leaderboard = (
                backtest_results
                .group_by("model_id")
                .agg([
                    pl.col(self.primary_metric).mean().alias(self.primary_metric),
                    pl.col(self.secondary_metric).mean().alias(self.secondary_metric),
                ])
                .sort(self.primary_metric)
            )
            best = leaderboard.head(1)
            champions = best.with_columns(
                pl.lit("global").alias("group_key")
            )

        logger.info(
            "Champion selection (%s): %d group(s)",
            self.granularity,
            len(champions),
        )
        for row in champions.iter_rows(named=True):
            logger.info(
                "  %s → %s (WMAPE=%.4f, Bias=%.4f)",
                row["group_key"],
                row["model_id"],
                row.get(self.primary_metric, 0),
                row.get(self.secondary_metric, 0),
            )

        return champions

    def select_by_horizon(
        self,
        backtest_results: pl.DataFrame,
        horizon_buckets: List[HorizonBucket],
        granularity_col: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Select champion model per horizon bucket.

        Each horizon bucket (e.g. short=1-4, medium=5-13, long=14-39)
        gets its own champion model based on the primary metric computed
        only over the forecast steps within that bucket.

        Parameters
        ----------
        backtest_results :
            Output from BacktestEngine.run(), must include ``forecast_step``.
        horizon_buckets :
            List of HorizonBucket defining step ranges.

        Returns
        -------
        DataFrame with columns:
          [group_key, horizon_bucket, start_step, end_step, model_id,
           primary_metric, secondary_metric]
        """
        if backtest_results.is_empty() or not horizon_buckets:
            return self.select(backtest_results, granularity_col)

        if "forecast_step" not in backtest_results.columns:
            logger.warning(
                "forecast_step column missing; falling back to single champion"
            )
            return self.select(backtest_results, granularity_col)

        # Assign horizon bucket to each row
        bucket_expr = pl.lit("unassigned")
        for bucket in horizon_buckets:
            bucket_expr = (
                pl.when(
                    pl.col("forecast_step").is_between(
                        bucket.start_step, bucket.end_step
                    )
                )
                .then(pl.lit(bucket.name))
                .otherwise(bucket_expr)
            )

        enriched = backtest_results.with_columns(
            bucket_expr.alias("horizon_bucket")
        ).filter(pl.col("horizon_bucket") != "unassigned")

        if enriched.is_empty():
            logger.warning("No rows matched any horizon bucket; falling back")
            return self.select(backtest_results, granularity_col)

        # Determine group column
        if granularity_col and granularity_col in backtest_results.columns:
            group_col = granularity_col
        elif self.granularity == "lob" and "lob" in backtest_results.columns:
            group_col = "lob"
        elif self.granularity == "series" and "series_id" in backtest_results.columns:
            group_col = "series_id"
        else:
            group_col = None

        # Build group keys list
        group_keys = ["horizon_bucket", "model_id"]
        if group_col:
            group_keys = [group_col] + group_keys

        # Aggregate metrics per (group, bucket, model)
        leaderboard = (
            enriched
            .group_by(group_keys)
            .agg([
                pl.col(self.primary_metric).mean().alias(self.primary_metric),
                pl.col(self.secondary_metric).mean().alias(self.secondary_metric),
            ])
        )

        # Select best model per (group, bucket)
        sort_keys = (
            ([group_col] if group_col else [])
            + ["horizon_bucket", self.primary_metric]
        )
        rank_keys = ([group_col] if group_col else []) + ["horizon_bucket"]

        champions = (
            leaderboard
            .with_columns(
                pl.col(self.secondary_metric).abs().alias("_abs_bias")
            )
            .sort(sort_keys + ["_abs_bias"])
            .group_by(rank_keys)
            .first()
            .drop("_abs_bias")
        )

        # Add start_step / end_step from bucket config
        bucket_map = {b.name: b for b in horizon_buckets}
        champions = champions.with_columns([
            pl.col("horizon_bucket").replace_strict(
                {b.name: b.start_step for b in horizon_buckets}
            ).alias("start_step"),
            pl.col("horizon_bucket").replace_strict(
                {b.name: b.end_step for b in horizon_buckets}
            ).alias("end_step"),
        ])

        # Add group_key column
        if group_col:
            champions = champions.rename({group_col: "group_key"})
        else:
            champions = champions.with_columns(
                pl.lit("global").alias("group_key")
            )

        logger.info(
            "Multi-horizon champion selection: %d bucket(s), %d champion(s)",
            len(horizon_buckets), len(champions),
        )
        for row in champions.iter_rows(named=True):
            logger.info(
                "  %s / %s → %s (%s=%.4f)",
                row["group_key"],
                row["horizon_bucket"],
                row["model_id"],
                self.primary_metric,
                row.get(self.primary_metric, 0),
            )

        return champions

    def compute_ensemble_weights(
        self, backtest_results: pl.DataFrame
    ) -> Dict[str, float]:
        """
        Compute inverse-WMAPE blend weights for a weighted ensemble.

        Each model receives a weight proportional to ``1 / mean_WMAPE`` across
        all series and folds.  This gives higher influence to better-performing
        models while every model retains some contribution.

        Parameters
        ----------
        backtest_results:
            Output from ``BacktestEngine.run()``.

        Returns
        -------
        Dict mapping model name → normalised weight (values sum to 1.0).
        Returns uniform weights if all models have zero WMAPE.
        """
        if backtest_results.is_empty():
            return {}

        # Aggregate mean WMAPE per model
        model_stats = (
            backtest_results
            .group_by("model_id")
            .agg(pl.col(self.primary_metric).mean().alias("mean_metric"))
            .sort("mean_metric")
        )

        rows = model_stats.iter_rows(named=True)
        # Inverse-WMAPE weights; protect against zero WMAPE
        raw: Dict[str, float] = {}
        for row in rows:
            wmape = row["mean_metric"]
            raw[row["model_id"]] = 1.0 / wmape if wmape > 1e-9 else 1e9

        total = sum(raw.values())
        weights = {k: v / total for k, v in raw.items()}

        logger.info("Ensemble weights computed:")
        for model, w in sorted(weights.items(), key=lambda x: -x[1]):
            logger.info("  %s → %.4f", model, w)

        return weights
