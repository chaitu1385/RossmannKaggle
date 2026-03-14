"""
FVA aggregation and reporting.

Takes per-series, per-fold FVA detail data and produces summary tables
for BI consumption and stakeholder reporting.
"""

import logging
from typing import List, Optional

import polars as pl

from ..metrics.fva import classify_fva

logger = logging.getLogger(__name__)


class FVAAnalyzer:
    """
    Aggregate FVA results across series, folds, and LOBs.

    Usage
    -----
    >>> analyzer = FVAAnalyzer()
    >>> detail = analyzer.compute_fva_detail(backtest_results)
    >>> summary = analyzer.summarize(detail)
    >>> leaderboard = analyzer.layer_leaderboard(detail)
    """

    LAYER_NAMES = {
        "naive_seasonal": "naive",
        "auto_arima": "statistical",
        "auto_ets": "statistical",
        "croston": "statistical",
        "croston_sba": "statistical",
        "tsb": "statistical",
        "lgbm_direct": "ml",
        "xgboost_direct": "ml",
    }

    def classify_model_layer(self, model_id: str) -> str:
        """Map a model_id to its FVA layer name."""
        return self.LAYER_NAMES.get(model_id, "ml")

    def compute_fva_detail(
        self,
        backtest_results: pl.DataFrame,
        id_col: str = "series_id",
        time_col: str = "target_week",
    ) -> pl.DataFrame:
        """
        Compute per-series, per-fold FVA from backtest results.

        The backtest_results DataFrame must have columns:
        model_id, fold, series_id, target_week, actual, forecast, wmape

        Returns a DataFrame with FVA metrics per (series, fold, layer).
        """
        if backtest_results.is_empty():
            return pl.DataFrame()

        required_cols = {"model_id", "fold", id_col, time_col, "actual", "forecast"}
        missing = required_cols - set(backtest_results.columns)
        if missing:
            raise ValueError(f"Missing columns for FVA: {missing}")

        # Add layer classification
        df = backtest_results.with_columns(
            pl.col("model_id")
            .map_elements(self.classify_model_layer, return_dtype=pl.Utf8)
            .alias("forecast_layer")
        )

        # For each layer, pick the best model (lowest WMAPE) per series per fold
        best_per_layer = (
            df.group_by([id_col, "fold", "forecast_layer"])
            .agg([
                pl.col("actual").mean().alias("actual_mean"),
                # Pick model with lowest WMAPE within layer
                pl.col("wmape").min().alias("layer_wmape"),
                pl.col("model_id").first().alias("layer_model"),
                pl.col("forecast").mean().alias("layer_forecast"),
                pl.col("actual").sum().alias("total_volume"),
            ])
        )

        # Pivot: for each (series, fold) get wmape per layer
        results = []
        for (sid, fold), group in best_per_layer.group_by([id_col, "fold"]):
            layer_data = {}
            for row in group.iter_rows(named=True):
                layer = row["forecast_layer"]
                layer_data[layer] = {
                    "wmape": row["layer_wmape"],
                    "model": row["layer_model"],
                    "volume": row["total_volume"],
                }

            # Compute FVA cascade
            layer_order = ["naive", "statistical", "ml"]
            available = [lyr for lyr in layer_order if lyr in layer_data]

            for i, layer in enumerate(available):
                parent_layer = available[i - 1] if i > 0 else None
                parent_wmape = layer_data[parent_layer]["wmape"] if parent_layer else None
                layer_wmape = layer_data[layer]["wmape"]
                fva = (parent_wmape - layer_wmape) if parent_wmape is not None else 0.0

                results.append({
                    id_col: sid,
                    "fold": fold,
                    "forecast_layer": layer,
                    "parent_layer": parent_layer,
                    "model_id": layer_data[layer]["model"],
                    "wmape": layer_wmape,
                    "parent_wmape": parent_wmape,
                    "fva_wmape": fva,
                    "fva_class": "BASELINE" if parent_layer is None else classify_fva(fva),
                    "total_volume": layer_data[layer]["volume"],
                })

        if not results:
            return pl.DataFrame()

        return pl.DataFrame(results)

    def summarize(
        self,
        fva_detail: pl.DataFrame,
        group_by: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Aggregate FVA detail into a summary table.

        Default grouping is by forecast_layer. Optionally group by
        additional columns (e.g. LOB, sparse_class).

        Returns one row per (group x layer) with:
        - mean_wmape, mean_fva_wmape
        - pct_adds_value, pct_neutral, pct_destroys_value
        - total_volume
        """
        if fva_detail.is_empty():
            return pl.DataFrame()

        group_cols = ["forecast_layer"]
        if group_by:
            group_cols = group_by + group_cols

        summary = (
            fva_detail
            .group_by(group_cols)
            .agg([
                pl.len().alias("n_series"),
                pl.col("wmape").mean().alias("mean_wmape"),
                pl.col("fva_wmape").mean().alias("mean_fva_wmape"),
                pl.col("fva_wmape").sum().alias("total_fva_wmape"),
                (pl.col("fva_class") == "ADDS_VALUE")
                    .sum().alias("n_adds_value"),
                (pl.col("fva_class") == "NEUTRAL")
                    .sum().alias("n_neutral"),
                (pl.col("fva_class") == "DESTROYS_VALUE")
                    .sum().alias("n_destroys_value"),
                pl.col("total_volume").sum().alias("total_volume"),
            ])
        )

        # Compute percentages
        summary = summary.with_columns([
            (pl.col("n_adds_value") / pl.col("n_series") * 100)
                .round(1).alias("pct_adds_value"),
            (pl.col("n_neutral") / pl.col("n_series") * 100)
                .round(1).alias("pct_neutral"),
            (pl.col("n_destroys_value") / pl.col("n_series") * 100)
                .round(1).alias("pct_destroys_value"),
        ])

        return summary.sort("forecast_layer")

    def layer_leaderboard(
        self, fva_detail: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Rank layers by aggregate FVA contribution.

        Returns one row per layer with:
        - rank (1 = most value added)
        - cumulative WMAPE reduction
        - robustness score (% of series improved)
        - recommendation (Keep / Review / Remove)
        """
        if fva_detail.is_empty():
            return pl.DataFrame()

        # Skip baseline layer for ranking
        ranked_data = fva_detail.filter(pl.col("fva_class") != "BASELINE")

        if ranked_data.is_empty():
            return pl.DataFrame()

        board = (
            ranked_data
            .group_by("forecast_layer")
            .agg([
                pl.col("fva_wmape").mean().alias("mean_fva_wmape"),
                pl.col("fva_wmape").sum().alias("cumulative_wmape_reduction"),
                pl.len().alias("n_series"),
                (pl.col("fva_class") == "ADDS_VALUE")
                    .sum().alias("n_improved"),
            ])
            .with_columns([
                (pl.col("n_improved") / pl.col("n_series") * 100)
                    .round(1).alias("robustness_score"),
            ])
            .sort("mean_fva_wmape", descending=True)
        )

        # Add rank and recommendation
        board = board.with_row_index("rank", offset=1)

        board = board.with_columns(
            pl.when(pl.col("robustness_score") >= 60)
            .then(pl.lit("Keep"))
            .when(pl.col("robustness_score") >= 30)
            .then(pl.lit("Review"))
            .otherwise(pl.lit("Remove"))
            .alias("recommendation")
        )

        return board
