"""
Base forecaster interface.

Every forecaster — statistical, ML, or ensemble — implements this ABC.
The interface is designed to work with Polars DataFrames in a multi-series
(panel) setting where multiple time series are stacked vertically.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import polars as pl


class BaseForecaster(ABC):
    """
    Abstract base for all forecasters.

    Subclasses must implement ``fit`` and ``predict``.  The registry
    discovers forecasters by their ``name`` attribute.
    """

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> None:
        """
        Fit the model on training data.

        Parameters
        ----------
        df:
            Panel DataFrame with columns [id_col, time_col, target_col].
            May contain additional feature columns.
        target_col:
            Column name for the target variable.
        time_col:
            Column name for the time dimension (weekly dates).
        id_col:
            Column name for the series identifier.
        """
        ...

    @abstractmethod
    def predict(
        self,
        horizon: int,
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """
        Generate forecasts for all fitted series.

        Parameters
        ----------
        horizon:
            Number of future periods (weeks) to forecast.
        id_col:
            Column name for series identifier in output.
        time_col:
            Column name for time dimension in output.

        Returns
        -------
        DataFrame with columns: [id_col, time_col, "forecast"].
        One row per (series, future week).
        """
        ...

    def get_params(self) -> Dict[str, Any]:
        """Return model parameters for logging/reproducibility."""
        return {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
