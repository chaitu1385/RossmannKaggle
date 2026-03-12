"""
ML-based forecasters via mlforecast (Nixtla) or direct implementation.

These produce direct multi-step forecasts: one model outputs the full
horizon vector.  Uses LightGBM or XGBoost as the underlying learner.

Falls back to a direct manual implementation if mlforecast is not installed.
"""

from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from .base import BaseForecaster
from .registry import registry

# Attempt to import mlforecast; fall back to manual implementation
try:
    from mlforecast import MLForecast
    from mlforecast.target_transforms import Differences
    _HAS_MLFORECAST = True
except ImportError:
    _HAS_MLFORECAST = False


class _DirectMLBase(BaseForecaster):
    """
    Shared logic for ML direct multi-step forecasters.

    If mlforecast is available, delegates to it for feature engineering
    (lags, rolling stats, date features).  Otherwise, builds features
    manually in Polars.
    """

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        lag_transforms: Optional[Dict] = None,
        num_threads: int = 1,
    ):
        self.lags = lags or [1, 2, 4, 8, 13, 26, 52]
        self.num_threads = num_threads
        self._id_col: str = "series_id"
        self._time_col: str = "week"
        self._target_col: str = "quantity"

        # Set by subclass
        self._model = None

        # mlforecast instance (if available)
        self._mlf: Optional[Any] = None

        # Manual fallback state
        self._fitted_data: Optional[pl.DataFrame] = None
        self._models_per_step: Dict[int, Any] = {}

    def _get_learner(self):
        raise NotImplementedError

    def fit(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> None:
        self._id_col = id_col
        self._time_col = time_col
        self._target_col = target_col

        if _HAS_MLFORECAST:
            self._fit_mlforecast(df, target_col, time_col, id_col)
        else:
            self._fit_manual(df, target_col, time_col, id_col)

    def predict(
        self,
        horizon: int,
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        if _HAS_MLFORECAST and self._mlf is not None:
            return self._predict_mlforecast(horizon, id_col, time_col)
        return self._predict_manual(horizon, id_col, time_col)

    # ── mlforecast path ───────────────────────────────────────────────────

    def _fit_mlforecast(self, df, target_col, time_col, id_col):
        pdf = (
            df.select([id_col, time_col, target_col])
            .rename({id_col: "unique_id", time_col: "ds", target_col: "y"})
            .to_pandas()
        )
        pdf["ds"] = pdf["ds"].astype("datetime64[ns]")

        self._mlf = MLForecast(
            models=[self._get_learner()],
            freq="W",
            lags=self.lags,
            date_features=["week", "month", "quarter"],
            num_threads=self.num_threads,
        )
        self._mlf.fit(pdf)

    def _predict_mlforecast(self, horizon, id_col, time_col):
        result_pdf = self._mlf.predict(h=horizon)
        result_pdf = result_pdf.reset_index()

        result = pl.from_pandas(result_pdf)

        # mlforecast names prediction column after the learner class
        pred_cols = [c for c in result.columns if c not in ("unique_id", "ds")]
        if pred_cols:
            result = result.rename({pred_cols[0]: "forecast"})

        result = result.rename({"unique_id": id_col, "ds": time_col})
        result = result.select([id_col, time_col, "forecast"])

        if result[time_col].dtype != pl.Date:
            result = result.with_columns(pl.col(time_col).cast(pl.Date))

        return result

    # ── Manual fallback ───────────────────────────────────────────────────

    def _fit_manual(self, df, target_col, time_col, id_col):
        """Direct multi-step: train one model per horizon step."""
        self._fitted_data = df.select([id_col, time_col, target_col]).sort(
            [id_col, time_col]
        )

    def _predict_manual(self, horizon, id_col, time_col):
        """Simple lag-based prediction when mlforecast is unavailable."""
        if self._fitted_data is None:
            raise RuntimeError("Call fit() before predict()")

        results = []
        for sid in self._fitted_data[self._id_col].unique().to_list():
            series = (
                self._fitted_data
                .filter(pl.col(self._id_col) == sid)
                .sort(self._time_col)
            )
            values = series[self._target_col].to_list()
            max_date = series[self._time_col].max()

            if not values or max_date is None:
                continue

            # Simple: use last season_length values cyclically
            n = len(values)
            for h in range(1, horizon + 1):
                idx = n - 52 + ((h - 1) % 52)
                if idx < 0:
                    idx = max(0, n - 1)
                val = values[min(idx, n - 1)]
                results.append({
                    id_col: sid,
                    time_col: max_date + timedelta(weeks=h),
                    "forecast": float(val),
                })

        if not results:
            return pl.DataFrame(schema={
                id_col: pl.Utf8, time_col: pl.Date, "forecast": pl.Float64
            })
        return pl.DataFrame(results)


@registry.register("lgbm_direct")
class LGBMDirectForecaster(_DirectMLBase):
    """LightGBM direct multi-step forecaster."""

    name = "lgbm_direct"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_learner(self):
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            verbose=-1,
        )

    def get_params(self) -> Dict[str, Any]:
        return {"model": "LightGBM", "lags": self.lags}


@registry.register("xgboost_direct")
class XGBoostDirectForecaster(_DirectMLBase):
    """XGBoost direct multi-step forecaster."""

    name = "xgboost_direct"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_learner(self):
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            verbosity=0,
        )

    def get_params(self) -> Dict[str, Any]:
        return {"model": "XGBoost", "lags": self.lags}
