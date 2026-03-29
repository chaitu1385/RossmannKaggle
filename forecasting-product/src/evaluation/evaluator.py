"""Model evaluation utilities."""

from typing import Dict, List

import numpy as np
import polars as pl

from ..models.base import BaseForecaster
from .metrics import mae, mape, rmse, rmspe


class ModelEvaluator:
    """Evaluates forecasting models across multiple metrics."""

    METRICS = {
        "rmspe": rmspe,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }

    def evaluate(
        self,
        model: BaseForecaster,
        X: pl.DataFrame,
        y: pl.Series,
        metrics: List[str] = None,
    ) -> Dict[str, float]:
        """Evaluate a model and return metric scores."""
        if metrics is None:
            metrics = list(self.METRICS.keys())

        y_pred = model.predict(X)
        y_true = y.to_numpy()

        results = {}
        for metric_name in metrics:
            if metric_name in self.METRICS:
                results[metric_name] = self.METRICS[metric_name](y_true, y_pred)

        return results

    def compare_models(
        self,
        models: List[BaseForecaster],
        X: pl.DataFrame,
        y: pl.Series,
    ) -> pl.DataFrame:
        """Compare multiple models and return a summary DataFrame."""
        rows = []
        for model in models:
            scores = self.evaluate(model, X, y)
            scores["model"] = model.name
            rows.append(scores)

        df = pl.DataFrame(rows)
        df = df.sort("rmspe")
        # Move 'model' to be a usable index-like column (first position)
        cols = ["model"] + [c for c in df.columns if c != "model"]
        return df.select(cols)
