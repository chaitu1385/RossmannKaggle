"""LightGBM forecasting model."""

from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import polars as pl

from .base import BaseForecaster


class LightGBMForecaster(BaseForecaster):
    """LightGBM-based time series forecaster."""

    DEFAULT_PARAMS = {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "objective": "regression",
        "metric": "rmse",
        "early_stopping_rounds": 50,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    def __init__(
        self,
        name: str = "lightgbm",
        params: Optional[Dict[str, Any]] = None,
        feature_cols: Optional[List[str]] = None,
    ):
        merged_params = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(name=name, params=merged_params)
        self.feature_cols = feature_cols
        self.model = lgb.LGBMRegressor(**self._get_model_params())

    def _get_model_params(self) -> Dict[str, Any]:
        """Extract LightGBM-specific params."""
        exclude = {"early_stopping_rounds"}
        return {k: v for k, v in self.params.items() if k not in exclude}

    def fit(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        X_val: Optional[pl.DataFrame] = None,
        y_val: Optional[pl.Series] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> "LightGBMForecaster":
        """Train the LightGBM model."""
        if self.feature_cols:
            X = X.select(self.feature_cols)
            if X_val is not None:
                X_val = X_val.select(self.feature_cols)

        self.feature_names = X.columns

        # External library requires pandas DataFrame
        X_pd = X.to_pandas()
        y_pd = y.to_pandas()

        callbacks = [lgb.log_evaluation(100)]
        if X_val is not None and y_val is not None:
            # External library requires pandas DataFrame
            X_val_pd = X_val.to_pandas()
            y_val_pd = y_val.to_pandas()
            callbacks.append(
                lgb.early_stopping(self.params.get("early_stopping_rounds", 50))
            )
            eval_set = [(X_val_pd, y_val_pd)]
        else:
            eval_set = None

        self.model.fit(
            X_pd,
            y_pd,
            eval_set=eval_set,
            callbacks=callbacks,
            categorical_feature=categorical_features or "auto",
        )
        self.is_fitted = True
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        if self.feature_cols:
            X = X.select(self.feature_cols)
        else:
            X = X.select(self.feature_names)

        # External library requires pandas DataFrame
        return self.model.predict(X.to_pandas())

    def get_feature_importance(self) -> pl.DataFrame:
        """Return feature importances as a Polars DataFrame with columns [feature, importance]."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")

        importance = self.model.feature_importances_
        return (
            pl.DataFrame({
                "feature": self.feature_names,
                "importance": importance.tolist(),
            })
            .sort("importance", descending=True)
        )
