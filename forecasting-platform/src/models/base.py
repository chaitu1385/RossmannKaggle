"""Base forecaster class for all models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.feature_names: List[str] = []
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseForecaster":
        """Train the model."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""

    def save(self, path: str) -> None:
        """Save model to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path / f"{self.name}.pkl")

    @classmethod
    def load(cls, path: str, name: str) -> "BaseForecaster":
        """Load model from disk."""
        return joblib.load(Path(path) / f"{name}.pkl")

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Return feature importances if available."""
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, fitted={self.is_fitted})"
