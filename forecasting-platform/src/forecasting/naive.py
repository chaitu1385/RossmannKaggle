"""
Seasonal Naive forecaster — the mandatory baseline.

Repeats the last observed seasonal cycle forward.  For weekly data with
yearly seasonality, it copies weeks from the same period one year ago.

This serves as the "beat this" benchmark for all other models.
"""

from datetime import timedelta
from typing import Any, Dict

import polars as pl

from .base import BaseForecaster
from .registry import registry


@registry.register("naive_seasonal")
class SeasonalNaiveForecaster(BaseForecaster):
    """Seasonal Naive: repeat last year's values."""

    name = "naive_seasonal"

    def __init__(self, season_length: int = 52):
        """
        Parameters
        ----------
        season_length:
            Number of periods in one seasonal cycle (52 for weekly-yearly).
        """
        self.season_length = season_length
        self._fitted_data: pl.DataFrame = pl.DataFrame()
        self._target_col: str = "quantity"
        self._time_col: str = "week"
        self._id_col: str = "series_id"

    def fit(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> None:
        self._fitted_data = df.select([id_col, time_col, target_col])
        self._target_col = target_col
        self._time_col = time_col
        self._id_col = id_col

    def predict(
        self,
        horizon: int,
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        results = []

        for series_id in self._fitted_data[self._id_col].unique().to_list():
            series = (
                self._fitted_data
                .filter(pl.col(self._id_col) == series_id)
                .sort(self._time_col)
            )

            values = series[self._target_col].to_list()
            max_date = series[self._time_col].max()

            if len(values) == 0 or max_date is None:
                continue

            # Build forecasts by cycling through seasonal history
            n = len(values)
            forecasts = []
            for h in range(1, horizon + 1):
                # Index into the seasonal cycle
                idx = n - self.season_length + ((h - 1) % self.season_length)
                if idx < 0:
                    idx = (h - 1) % n  # fallback for short series
                val = values[idx] if 0 <= idx < n else 0.0
                forecast_date = max_date + timedelta(weeks=h)
                forecasts.append({
                    id_col: series_id,
                    time_col: forecast_date,
                    "forecast": float(val),
                })

            results.extend(forecasts)

        if not results:
            return pl.DataFrame(schema={
                id_col: pl.Utf8, time_col: pl.Date, "forecast": pl.Float64
            })

        return pl.DataFrame(results)

    def get_params(self) -> Dict[str, Any]:
        return {"season_length": self.season_length}
