"""Model evaluation utilities."""

import numpy as np
import pandas as pd
from typing import Dict, List

from .metrics import rmspe, rmse, mae, mape
from ..models.base import BaseForecaster


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
        X: pd.DataFrame,
        y: pd.Series,
        metrics: List[str] = None,
    ) -> Dict[str, float]:
        """Evaluate a model and return metric scores."""
        if metrics is None:
            metrics = list(self.METRICS.keys())

        y_pred = model.predict(X)
        y_true = y.values

        results = {}
        for metric_name in metrics:
            if metric_name in self.METRICS:
                results[metric_name] = self.METRICS[metric_name](y_true, y_pred)

        return results

    def compare_models(
        self,
        models: List[BaseForecaster],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """Compare multiple models and return a summary DataFrame."""
        rows = []
        for model in models:
            scores = self.evaluate(model, X, y)
            scores["model"] = model.name
            rows.append(scores)

        return pd.DataFrame(rows).set_index("model").sort_values("rmspe")
