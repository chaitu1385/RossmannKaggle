"""XGBoost forecasting model."""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

import xgboost as xgb

from .base import BaseForecaster


class XGBoostForecaster(BaseForecaster):
    """XGBoost-based time series forecaster."""

    DEFAULT_PARAMS = {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "early_stopping_rounds": 50,
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(
        self,
        name: str = "xgboost",
        params: Optional[Dict[str, Any]] = None,
        feature_cols: Optional[List[str]] = None,
    ):
        merged_params = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(name=name, params=merged_params)
        self.feature_cols = feature_cols
        self.model = xgb.XGBRegressor(**self._get_model_params())

    def _get_model_params(self) -> Dict[str, Any]:
        """Extract XGBoost-specific params (excluding training params)."""
        exclude = {"early_stopping_rounds"}
        return {k: v for k, v in self.params.items() if k not in exclude}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "XGBoostForecaster":
        """Train the XGBoost model."""
        if self.feature_cols:
            X = X[self.feature_cols]
            if X_val is not None:
                X_val = X_val[self.feature_cols]

        self.feature_names = list(X.columns)

        eval_set = [(X, y)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            verbose=100,
            early_stopping_rounds=self.params.get("early_stopping_rounds", 50),
        )
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        if self.feature_cols:
            X = X[self.feature_cols]
        else:
            X = X[self.feature_names]

        return self.model.predict(X)

    def get_feature_importance(self) -> pd.Series:
        """Return feature importances."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")

        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
