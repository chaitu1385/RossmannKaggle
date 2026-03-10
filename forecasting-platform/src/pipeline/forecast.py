"""
Forecast pipeline — production inference workflow.

Steps:
  1. Load config.
  2. Build series (with transitions).
  3. Load champion model(s) from backtest results.
  4. Fit champion on all available data.
  5. Forecast the full horizon (39 weeks).
  6. (Optionally) reconcile across hierarchy.
  7. Write frozen forecasts to output store.
"""

import logging
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import polars as pl

from ..config.schema import PlatformConfig
from ..forecasting.registry import registry
from ..series.builder import SeriesBuilder

logger = logging.getLogger(__name__)


class ForecastPipeline:
    """
    End-to-end production forecast pipeline.

    Usage
    -----
    >>> config = load_config("configs/platform_config.yaml")
    >>> pipeline = ForecastPipeline(config)
    >>> forecast_df = pipeline.run(actuals_df, champion_model="lgbm_direct")
    """

    def __init__(self, config: PlatformConfig):
        self.config = config
        self._series_builder = SeriesBuilder(config)

    def run(
        self,
        actuals: pl.DataFrame,
        champion_model: str = "naive_seasonal",
        product_master: Optional[pl.DataFrame] = None,
        mapping_table: Optional[pl.DataFrame] = None,
        forecast_origin: Optional[date] = None,
        overrides: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """
        Generate production forecasts.

        Parameters
        ----------
        actuals:
            Historical data.
        champion_model:
            Name of the model to use (from registry).
        product_master:
            Product metadata for transitions.
        mapping_table:
            SKU mappings for transitions.
        forecast_origin:
            Forecast start date.  Defaults to the last date in actuals.
        overrides:
            Planner transition overrides.

        Returns
        -------
        Forecast DataFrame: [series_id, week, forecast].
        """
        fc = self.config.forecast
        horizon = fc.horizon_weeks

        # Step 1: Build series
        logger.info("Building model-ready series...")
        series = self._series_builder.build(
            actuals=actuals,
            product_master=product_master,
            mapping_table=mapping_table,
            forecast_origin=forecast_origin,
            overrides=overrides,
        )

        # Step 2: Instantiate champion
        forecaster = registry.build(champion_model)
        logger.info("Champion model: %s", forecaster.name)

        # Step 3: Fit on all data
        logger.info("Fitting on %d rows...", len(series))
        forecaster.fit(
            series,
            target_col=fc.target_column,
            time_col=fc.time_column,
            id_col=fc.series_id_column,
        )

        # Step 4: Predict
        logger.info("Forecasting %d weeks...", horizon)
        forecast = forecaster.predict(
            horizon=horizon,
            id_col=fc.series_id_column,
            time_col=fc.time_column,
        )

        # Step 5: Write to output
        output_path = Path(self.config.output.forecast_path)
        output_path.mkdir(parents=True, exist_ok=True)

        origin_str = (
            forecast_origin.isoformat()
            if forecast_origin
            else date.today().isoformat()
        )
        filename = f"forecast_{self.config.lob}_{origin_str}.parquet"
        filepath = output_path / filename

        forecast.write_parquet(str(filepath))
        logger.info("Forecast written to %s (%d rows)", filepath, len(forecast))

        return forecast
