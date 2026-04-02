"""
Base forecaster interface.

Every forecaster — statistical, ML, or ensemble — implements this ABC.
The interface is designed to work with Polars DataFrames in a multi-series
(panel) setting where multiple time series are stacked vertically.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import polars as pl

from ..utils.gap_fill import fill_gaps as _shared_fill_gaps


class BaseForecaster(ABC):
    """
    Abstract base for all forecasters.

    Subclasses must implement ``fit`` and ``predict``.  The registry
    discovers forecasters by their ``name`` attribute.

    Subclasses may override ``validate_and_prepare`` to add model-specific
    data preprocessing (e.g. gap-filling for backends that require
    contiguous weekly dates).
    """

    name: str = "base"

    def validate_and_prepare(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> pl.DataFrame:
        """
        Pre-fit validation and preparation hook.

        Called automatically before ``fit``.  The default implementation
        returns the data unchanged.  Override in subclasses to add
        model-specific requirements (e.g. filling missing weeks).

        Parameters
        ----------
        df:
            Panel DataFrame with columns [id_col, time_col, target_col].
        target_col, time_col, id_col:
            Column names.

        Returns
        -------
        Cleaned / validated DataFrame ready for fitting.
        """
        return df

    @staticmethod
    def fill_gaps(
        df: pl.DataFrame,
        time_col: str = "week",
        id_col: str = "series_id",
        target_col: str = "quantity",
        strategy: str = "zero",
        freq: str = "W",
    ) -> pl.DataFrame:
        """
        Fill missing periods for each series to ensure contiguous dates.

        Delegates to :func:`src.utils.gap_fill.fill_gaps`.

        Parameters
        ----------
        strategy:
            ``"zero"`` — fill gaps with 0.0 (good for statistical models
            where zero demand is meaningful).
            ``"forward_fill"`` — propagate last known value forward, then
            back-fill any leading nulls with the first known value.  Better
            for tree-based models where artificial zeros contaminate splits.
        freq:
            Frequency code: ``"D"`` | ``"W"`` | ``"M"`` | ``"Q"``.
        """
        return _shared_fill_gaps(
            df,
            time_col=time_col,
            id_col=id_col,
            target_col=target_col,
            strategy=strategy,  # type: ignore[arg-type]
            freq=freq,
        )

    @staticmethod
    def fill_weekly_gaps(
        df: pl.DataFrame,
        time_col: str = "week",
        id_col: str = "series_id",
        target_col: str = "quantity",
        strategy: str = "zero",
    ) -> pl.DataFrame:
        """
        Fill missing weeks for each series to ensure contiguous weekly dates.

        .. deprecated::
            Use :meth:`fill_gaps` with ``freq="W"`` instead.
        """
        warnings.warn(
            "fill_weekly_gaps is deprecated; use fill_gaps(freq=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _shared_fill_gaps(
            df,
            time_col=time_col,
            id_col=id_col,
            target_col=target_col,
            strategy=strategy,  # type: ignore[arg-type]
            freq="W",
        )

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

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float],
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """
        Generate probabilistic forecasts at the requested quantile levels.

        Default implementation returns the point forecast for every quantile
        (degenerate / zero-width intervals).  Subclasses should override
        for proper interval estimation.

        Parameters
        ----------
        horizon:
            Number of future periods to forecast.
        quantiles:
            Ordered list of quantile levels, e.g. [0.1, 0.5, 0.9].
        id_col, time_col:
            Column names in output.

        Returns
        -------
        DataFrame with columns: [id_col, time_col, "forecast_p{q}"] for each q.
        E.g. for quantiles [0.1, 0.5, 0.9]: columns forecast_p10, forecast_p50, forecast_p90.
        """
        point = self.predict(horizon, id_col=id_col, time_col=time_col)
        for q in quantiles:
            col = f"forecast_p{int(round(q * 100))}"
            point = point.with_columns(pl.col("forecast").alias(col))
        return point.drop("forecast")

    def get_params(self) -> Dict[str, Any]:
        """Return model parameters for logging/reproducibility."""
        return {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
