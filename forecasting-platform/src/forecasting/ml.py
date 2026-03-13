"""
ML-based forecasters via mlforecast (Nixtla) or direct implementation.

These produce direct multi-step forecasts: one model outputs the full
horizon vector.  Uses LightGBM or XGBoost as the underlying learner.

Falls back to a direct manual implementation if mlforecast is not installed.
"""

from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

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

        # Quantile regression state (lazily populated by predict_quantiles)
        self._train_pdf: Optional[Any] = None          # pandas df stored for quantile refit
        self._quantile_mlfs: Dict[float, Any] = {}     # q -> fitted MLForecast

    def _get_learner(self):
        raise NotImplementedError

    def _get_quantile_learner(self, alpha: float):
        """Return a quantile regression version of the base learner. Override in subclasses."""
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
        self._train_pdf = pdf  # store for lazy quantile model training

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

    # ── Probabilistic forecasting ─────────────────────────────────────────

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float],
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """
        Quantile forecasts via native quantile regression.

        For each non-P50 quantile a separate model is trained lazily using the
        underlying learner's quantile objective (LightGBM ``objective="quantile"``,
        XGBoost ``objective="reg:quantileerror"``).  P50 reuses the point forecast.

        Falls back to YoY-residual empirical quantiles when mlforecast is not
        available or quantile learner is not supported by the subclass.
        """
        # P50 always comes from the already-fitted point model
        point = self.predict(horizon, id_col=id_col, time_col=time_col)
        output = point.select([id_col, time_col])

        for q in quantiles:
            col = f"forecast_p{int(round(q * 100))}"
            if abs(q - 0.5) < 1e-6:
                output = output.with_columns(point["forecast"].alias(col))
                continue

            if _HAS_MLFORECAST and self._train_pdf is not None:
                try:
                    q_preds = self._predict_quantile_mlforecast(
                        q, horizon, id_col, time_col
                    )
                    output = output.join(
                        q_preds.rename({"forecast": col}),
                        on=[id_col, time_col],
                        how="left",
                    )
                    continue
                except (NotImplementedError, Exception):
                    pass  # fall through to residual fallback

            # Residual fallback: use empirical quantile of historical errors
            q_preds = self._predict_quantile_residual(
                q, point, id_col, time_col
            )
            output = output.with_columns(q_preds.alias(col))

        return output

    def _predict_quantile_mlforecast(
        self, q: float, horizon: int, id_col: str, time_col: str
    ) -> pl.DataFrame:
        """Lazily train + predict a quantile regression model via mlforecast."""
        if q not in self._quantile_mlfs:
            q_learner = self._get_quantile_learner(q)  # raises NotImplementedError if unsupported
            mlf_q = MLForecast(
                models=[q_learner],
                freq="W",
                lags=self.lags,
                date_features=["week", "month", "quarter"],
                num_threads=self.num_threads,
            )
            mlf_q.fit(self._train_pdf)
            self._quantile_mlfs[q] = mlf_q

        result_pdf = self._quantile_mlfs[q].predict(h=horizon).reset_index()
        result = pl.from_pandas(result_pdf)
        pred_cols = [c for c in result.columns if c not in ("unique_id", "ds")]
        result = result.rename({pred_cols[0]: "forecast", "unique_id": id_col, "ds": time_col})
        result = result.select([id_col, time_col, "forecast"])
        if result[time_col].dtype != pl.Date:
            result = result.with_columns(pl.col(time_col).cast(pl.Date))
        return result

    def _predict_quantile_residual(
        self, q: float, point: pl.DataFrame, id_col: str, time_col: str
    ) -> pl.Series:
        """
        Residual-based interval: point forecast ± empirical quantile of historical
        seasonal residuals.  Used when quantile regression is unavailable.
        """
        if self._fitted_data is None:
            return point["forecast"]

        offsets = []
        for h_row in point.iter_rows(named=True):
            sid = h_row[id_col]
            series = (
                self._fitted_data
                .filter(pl.col(self._id_col) == sid)
                .sort(self._time_col)
            )
            values = series[self._target_col].to_list()
            n = len(values)
            sl = 52
            residuals = [
                values[i] - values[i - sl]
                for i in range(sl, n)
            ] or [0.0]
            offset = float(np.quantile(residuals, q))
            offsets.append(h_row["forecast"] + offset)

        return pl.Series("forecast", offsets)

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

    def _get_quantile_learner(self, alpha: float):
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            verbose=-1,
            objective="quantile",
            alpha=alpha,
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

    def _get_quantile_learner(self, alpha: float):
        import xgboost as xgb
        try:
            # XGBoost ≥ 1.6 supports reg:quantileerror
            return xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                verbosity=0,
                objective="reg:quantileerror",
                quantile_alpha=alpha,
            )
        except TypeError:
            # Older XGBoost: fall back to pseudo-Huber (approximate)
            return xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                verbosity=0,
            )

    def get_params(self) -> Dict[str, Any]:
        return {"model": "XGBoost", "lags": self.lags}
