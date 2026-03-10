"""
Backtesting engine — runs all forecasters across all CV folds.

For each (model, fold) combination:
  1. Fit on training data.
  2. Predict over the validation horizon.
  3. Compute metrics (WMAPE, Normalized Bias, etc.) per series.
  4. Write results to the metric store.
"""

import logging
import uuid
from datetime import date
from typing import Dict, List, Optional

import polars as pl

from ..config.schema import BacktestConfig, PlatformConfig
from ..forecasting.base import BaseForecaster
from ..metrics.definitions import compute_all_metrics
from ..metrics.store import MetricStore
from .cross_validator import WalkForwardCV

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Run walk-forward backtesting for multiple models.

    Produces a detailed metric table and writes to the metric store.
    """

    def __init__(
        self,
        config: PlatformConfig,
        metric_store: Optional[MetricStore] = None,
    ):
        self.config = config
        self.bt_config = config.backtest
        self.metric_store = metric_store or MetricStore(
            config.output.metrics_path
        )
        self._cv = WalkForwardCV(
            n_folds=self.bt_config.n_folds,
            val_weeks=self.bt_config.val_weeks,
            gap_weeks=self.bt_config.gap_weeks,
        )

    def run(
        self,
        series: pl.DataFrame,
        forecasters: List[BaseForecaster],
        target_col: Optional[str] = None,
        time_col: Optional[str] = None,
        id_col: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Run full backtesting.

        Parameters
        ----------
        series:
            Model-ready panel data (from SeriesBuilder).
        forecasters:
            List of forecaster instances to evaluate.

        Returns
        -------
        DataFrame of per-(model, fold, series, week) metrics.
        """
        fc = self.config.forecast
        target_col = target_col or fc.target_column
        time_col = time_col or fc.time_column
        id_col = id_col or fc.series_id_column

        splits = self._cv.split_data(series, time_col)
        if not splits:
            logger.warning("No valid CV folds. Check data date range.")
            return pl.DataFrame()

        run_id = f"backtest-{uuid.uuid4().hex[:8]}"
        run_date = date.today()
        all_results: List[pl.DataFrame] = []

        for fold, train, val in splits:
            logger.info(
                "Fold %d: train %s→%s, val %s→%s (%d/%d train/val rows)",
                fold.fold_index,
                fold.train_start, fold.train_end,
                fold.val_start, fold.val_end,
                len(train), len(val),
            )

            for forecaster in forecasters:
                logger.info(
                    "  Model: %s (fold %d)", forecaster.name, fold.fold_index
                )
                try:
                    fold_result = self._run_one(
                        forecaster=forecaster,
                        train=train,
                        val=val,
                        fold_index=fold.fold_index,
                        run_id=run_id,
                        run_date=run_date,
                        target_col=target_col,
                        time_col=time_col,
                        id_col=id_col,
                    )
                    all_results.append(fold_result)
                except Exception as e:
                    logger.error(
                        "  Model %s fold %d failed: %s",
                        forecaster.name, fold.fold_index, e,
                    )

        if not all_results:
            return pl.DataFrame()

        combined = pl.concat(all_results, how="vertical_relaxed")

        # Write to metric store
        self.metric_store.write(
            combined,
            run_type="backtest",
            lob=self.config.lob,
        )
        logger.info(
            "Backtest complete: %d results written to metric store",
            len(combined),
        )

        return combined

    def _run_one(
        self,
        forecaster: BaseForecaster,
        train: pl.DataFrame,
        val: pl.DataFrame,
        fold_index: int,
        run_id: str,
        run_date: date,
        target_col: str,
        time_col: str,
        id_col: str,
    ) -> pl.DataFrame:
        """Run one (model, fold) combination."""
        # Fit
        forecaster.fit(train, target_col=target_col, time_col=time_col, id_col=id_col)

        # Predict
        horizon = self.bt_config.val_weeks
        predictions = forecaster.predict(horizon, id_col=id_col, time_col=time_col)

        # Join predictions with actuals
        merged = val.join(
            predictions,
            on=[id_col, time_col],
            how="inner",
        )

        if merged.is_empty():
            return pl.DataFrame()

        # Compute per-series metrics
        results = []
        for sid in merged[id_col].unique().to_list():
            s = merged.filter(pl.col(id_col) == sid)
            actual = s[target_col]
            forecast = s["forecast"]

            metrics = compute_all_metrics(
                actual, forecast,
                metric_names=self.config.metrics,
            )

            for _, row in enumerate(s.iter_rows(named=True)):
                record = {
                    "run_id": run_id,
                    "run_type": "backtest",
                    "run_date": run_date,
                    "lob": self.config.lob,
                    "model_id": forecaster.name,
                    "fold": fold_index,
                    "grain_level": "series",
                    "series_id": sid,
                    "channel": "",
                    "target_week": row[time_col],
                    "actual": float(row[target_col]),
                    "forecast": float(row["forecast"]),
                }
                record.update(metrics)
                results.append(record)

        return pl.DataFrame(results)
