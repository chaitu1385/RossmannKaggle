"""
Statistical forecasters via statsforecast (Nixtla).

Wraps AutoARIMA and AutoETS to provide the BaseForecaster interface.
Falls back to a simple seasonal repeat if statsforecast is not installed.

These models output the full horizon vector at once (direct multi-step),
which aligns with the platform's seq2seq requirement.
"""

from typing import Any, Dict, List, Optional

import polars as pl

from .base import BaseForecaster
from .registry import registry

# Attempt to import statsforecast; fall back gracefully
try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA as _AutoARIMA
    from statsforecast.models import AutoETS as _AutoETS
    _HAS_STATSFORECAST = True
except ImportError:
    _HAS_STATSFORECAST = False


class _StatsforecastBase(BaseForecaster):
    """
    Shared logic for statsforecast-backed models.

    Handles the Polars ↔ pandas conversion that statsforecast requires,
    and maps columns to the expected ``unique_id / ds / y`` schema.
    """

    def __init__(self, season_length: int = 52):
        self.season_length = season_length
        self._sf: Optional[Any] = None
        self._id_col: str = "series_id"
        self._time_col: str = "week"
        self._target_col: str = "quantity"

    def _get_model(self):
        raise NotImplementedError

    def fit(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> None:
        if not _HAS_STATSFORECAST:
            raise ImportError(
                f"{self.name} requires 'statsforecast'. "
                "Install with: pip install statsforecast"
            )

        self._id_col = id_col
        self._time_col = time_col
        self._target_col = target_col

        # statsforecast expects pandas with columns: unique_id, ds, y
        pdf = (
            df.select([id_col, time_col, target_col])
            .rename({id_col: "unique_id", time_col: "ds", target_col: "y"})
            .to_pandas()
        )
        pdf["ds"] = pdf["ds"].astype("datetime64[ns]")

        model = self._get_model()
        self._sf = StatsForecast(
            models=[model],
            freq="W",
            n_jobs=1,
        )
        self._sf.fit(pdf)

    def predict(
        self,
        horizon: int,
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        if self._sf is None:
            raise RuntimeError(f"{self.name}: call fit() before predict()")

        result_pdf = self._sf.predict(h=horizon)
        result_pdf = result_pdf.reset_index()

        # Map back to original column names
        result = pl.from_pandas(result_pdf)

        # statsforecast names the prediction column after the model class
        pred_cols = [c for c in result.columns if c not in ("unique_id", "ds")]
        if pred_cols:
            result = result.rename({pred_cols[0]: "forecast"})

        result = result.rename({"unique_id": id_col, "ds": time_col})
        result = result.select([id_col, time_col, "forecast"])

        # Ensure date type
        if result[time_col].dtype != pl.Date:
            result = result.with_columns(pl.col(time_col).cast(pl.Date))

        return result

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float],
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """
        Native statsforecast prediction intervals.

        statsforecast returns symmetric PI at ``level`` % coverage.
        A ``level=80`` interval covers [P10, P90]; ``level=90`` covers [P5, P95].
        For each lower quantile q < 0.5 the required coverage level is
        ``(1 - 2*q) * 100``.
        """
        if self._sf is None:
            raise RuntimeError(f"{self.name}: call fit() before predict_quantiles()")

        # Determine the coverage levels we need
        lower_qs = sorted(q for q in quantiles if q < 0.5 - 1e-6)
        levels = sorted({int(round((1 - 2 * q) * 100)) for q in lower_qs}) if lower_qs else [80]

        result_pdf = self._sf.predict(h=horizon, level=levels)
        result_pdf = result_pdf.reset_index()

        result = pl.from_pandas(result_pdf)
        result = result.rename({"unique_id": id_col, "ds": time_col})
        if result[time_col].dtype != pl.Date:
            result = result.with_columns(pl.col(time_col).cast(pl.Date))

        # Identify point forecast column (no "-lo-" / "-hi-" in name)
        point_col = next(
            c for c in result.columns
            if c not in (id_col, time_col) and "-lo-" not in c and "-hi-" not in c
        )

        output = result.select([id_col, time_col])
        for q in quantiles:
            col = f"forecast_p{int(round(q * 100))}"
            if abs(q - 0.5) < 1e-6:
                output = output.with_columns(result[point_col].alias(col))
            elif q < 0.5:
                level = int(round((1 - 2 * q) * 100))
                lo = next((c for c in result.columns if f"-lo-{level}" in c), None)
                src = result[lo] if lo else result[point_col]
                output = output.with_columns(src.alias(col))
            else:  # q > 0.5 — mirror lower bound
                mirror_q = 1.0 - q
                level = int(round((1 - 2 * mirror_q) * 100))
                hi = next((c for c in result.columns if f"-hi-{level}" in c), None)
                src = result[hi] if hi else result[point_col]
                output = output.with_columns(src.alias(col))

        return output


@registry.register("auto_arima")
class AutoARIMAForecaster(_StatsforecastBase):
    """AutoARIMA via statsforecast."""

    name = "auto_arima"

    def _get_model(self):
        return _AutoARIMA(season_length=self.season_length)

    def get_params(self) -> Dict[str, Any]:
        return {"model": "AutoARIMA", "season_length": self.season_length}


@registry.register("auto_ets")
class AutoETSForecaster(_StatsforecastBase):
    """AutoETS via statsforecast."""

    name = "auto_ets"

    def _get_model(self):
        return _AutoETS(season_length=self.season_length)

    def get_params(self) -> Dict[str, Any]:
        return {"model": "AutoETS", "season_length": self.season_length}
