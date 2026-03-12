"""
Champion model selection.

After backtesting, selects the best model based on the primary metric
(WMAPE by default) at the configured granularity (per-LOB to start).

The champion table is consumed by the inference pipeline to decide
which model generates the final forecast.
"""

import logging
from typing import Optional

import polars as pl

from ..config.schema import BacktestConfig

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
