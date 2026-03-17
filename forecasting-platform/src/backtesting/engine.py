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
from typing import List, Optional

import polars as pl

from ..config.schema import PlatformConfig, get_frequency_profile
from ..forecasting.base import BaseForecaster
from ..metrics.definitions import compute_all_metrics
from ..metrics.store import METRIC_SCHEMA, MetricStore
from .cross_validator import WalkForwardCV

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Run walk-forward backtesting for multiple models.

    Produces a detailed metric table and writes to the metric store.
    Model failures are tracked and returned alongside results.
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
            frequency=config.forecast.frequency,
        )
        self._failures: List[dict] = []

    def run(
        self,
        series: pl.DataFrame,
        forecasters: List[BaseForecaster],
        sparse_forecasters: Optional[List[BaseForecaster]] = None,
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
            List of forecaster instances to evaluate on dense (regular) series.
        sparse_forecasters:
            Optional list of forecasters for intermittent/sparse series.
            When provided, series are split by the SparseDetector and each
            partition is evaluated with its dedicated model list.

        Returns
        -------
        DataFrame of per-(model, fold, series, week) metrics.
        """
        fc = self.config.forecast
        target_col = target_col or fc.target_column
        time_col = time_col or fc.time_column
        id_col = id_col or fc.series_id_column

        # Sparse routing: split series into dense and sparse partitions
        if sparse_forecasters:
            from ..series.sparse_detector import SparseDetector
            detector = SparseDetector(
                adi_threshold=fc.sparse_adi_threshold,
                cv2_threshold=fc.sparse_cv2_threshold,
            )
            dense_series, sparse_series = detector.split(
                series, target_col=target_col, id_col=id_col
            )
            n_sparse = sparse_series[id_col].n_unique() if not sparse_series.is_empty() else 0
            n_dense = dense_series[id_col].n_unique() if not dense_series.is_empty() else 0
            logger.info(
                "Sparse routing: %d dense series → %s, %d sparse series → %s",
                n_dense, [f.name for f in forecasters],
                n_sparse, [f.name for f in sparse_forecasters],
            )
        else:
            dense_series = series
            sparse_series = pl.DataFrame()

        splits = self._cv.split_data(series, time_col)
        if not splits:
            logger.warning("No valid CV folds. Check data date range.")
            return pl.DataFrame()

        run_id = f"backtest-{uuid.uuid4().hex[:8]}"
        run_date = date.today()
        all_results: List[pl.DataFrame] = []
        self._failures = []

        for fold, _train_full, _val_full in splits:
            logger.info(
                "Fold %d: train %s→%s, val %s→%s (%d/%d train/val rows)",
                fold.fold_index,
                fold.train_start, fold.train_end,
                fold.val_start, fold.val_end,
                len(_train_full), len(_val_full),
            )

            # Build per-partition train/val splits
            partitions: List[tuple] = []
            if not dense_series.is_empty():
                dense_ids = dense_series[id_col].unique().to_list()
                train_d = _train_full.filter(pl.col(id_col).is_in(dense_ids))
                val_d = _val_full.filter(pl.col(id_col).is_in(dense_ids))
                partitions.append((train_d, val_d, forecasters))
            if sparse_forecasters and not sparse_series.is_empty():
                sparse_ids = sparse_series[id_col].unique().to_list()
                train_s = _train_full.filter(pl.col(id_col).is_in(sparse_ids))
                val_s = _val_full.filter(pl.col(id_col).is_in(sparse_ids))
                partitions.append((train_s, val_s, sparse_forecasters))
            if not partitions:
                partitions = [(_train_full, _val_full, forecasters)]

            for train, val, models in partitions:
                if train.is_empty() or val.is_empty():
                    continue
                for forecaster in models:
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
                        self._failures.append({
                            "model_id": forecaster.name,
                            "fold": fold.fold_index,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        })

        if self._failures:
            logger.warning(
                "Backtest had %d model-fold failure(s): %s",
                len(self._failures),
                ", ".join(
                    f"{f['model_id']}(fold {f['fold']}): {f['error_type']}"
                    for f in self._failures
                ),
            )

        # Log neural model training notes (CPU-vs-GPU advisory)
        self._neural_notes = []
        for forecaster in forecasters:
            if hasattr(forecaster, "training_notes"):
                notes = forecaster.training_notes()
                self._neural_notes.append(notes)
                if not notes.get("is_production_quality", True):
                    logger.warning(
                        "NEURAL MODEL ADVISORY: %s", notes["recommendation"]
                    )

        if not all_results:
            return pl.DataFrame()

        all_results = [self._normalize_result_schema(r) for r in all_results]
        combined = pl.concat(all_results, how="diagonal")

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

    @property
    def failures(self) -> List[dict]:
        """Return list of model-fold failures from the most recent run."""
        return list(self._failures)

    @property
    def neural_training_notes(self) -> List[dict]:
        """Return neural model training advisories from the most recent run.

        Each entry contains current settings, whether the configuration is
        production-quality, and recommended GPU settings.  Useful for
        backtest reports and model cards.
        """
        return list(getattr(self, "_neural_notes", []))

    def get_failure_summary(self) -> pl.DataFrame:
        """Return failures as a DataFrame for reporting."""
        if not self._failures:
            return pl.DataFrame(schema={
                "model_id": pl.Utf8,
                "fold": pl.Int32,
                "error_type": pl.Utf8,
                "error_message": pl.Utf8,
            })
        return pl.DataFrame(self._failures)

    @staticmethod
    def _normalize_result_schema(df: pl.DataFrame) -> pl.DataFrame:
        """Ensure result DataFrame conforms to METRIC_SCHEMA before concat."""
        if df.is_empty():
            return df
        for col, dtype in METRIC_SCHEMA.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
        return df

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

        # Quantile predictions (if calibration enabled)
        quantiles = self.config.forecast.quantiles
        if quantiles and self.config.forecast.calibration.enabled:
            qdf = forecaster.predict_quantiles(
                horizon=horizon, quantiles=quantiles,
                id_col=id_col, time_col=time_col,
            )
            # Drop point forecast column from quantile df to avoid join conflict
            q_cols = [c for c in qdf.columns if c.startswith("forecast_p")]
            predictions = predictions.join(
                qdf.select([id_col, time_col] + q_cols),
                on=[id_col, time_col],
                how="left",
            )

        # Join predictions with actuals
        merged = val.join(
            predictions,
            on=[id_col, time_col],
            how="inner",
        )

        if merged.is_empty():
            return pl.DataFrame()

        # Compute per-series metrics
        val_start = val[time_col].min()
        results = []
        for sid in merged[id_col].unique().to_list():
            s = merged.filter(pl.col(id_col) == sid)
            actual = s[target_col]
            forecast = s["forecast"]

            # Build context for context-dependent metrics (e.g. MASE)
            insample = train.filter(pl.col(id_col) == sid)[target_col]
            context = {"insample": insample}

            metrics = compute_all_metrics(
                actual, forecast,
                metric_names=self.config.metrics,
                context=context,
            )

            # Detect quantile columns present in merged data
            q_cols = [c for c in s.columns if c.startswith("forecast_p")]

            for row in s.iter_rows(named=True):
                # forecast_step: 1-indexed week offset from validation start
                step_days = get_frequency_profile(self.config.forecast.frequency)["timedelta_kwargs"]
                period_days = sum(
                    v * (7 if k == "weeks" else 1) for k, v in step_days.items()
                )
                forecast_step = (row[time_col] - val_start).days // period_days + 1

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
                    "forecast_step": forecast_step,
                    "actual": float(row[target_col]),
                    "forecast": float(row["forecast"]),
                }
                record.update(metrics)
                for qc in q_cols:
                    if row[qc] is not None:
                        record[qc] = float(row[qc])
                results.append(record)

        return pl.DataFrame(results)
