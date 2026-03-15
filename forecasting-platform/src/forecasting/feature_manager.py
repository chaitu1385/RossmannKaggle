"""
External feature lifecycle management for mlforecast-based models.

Handles:
  1. Detection of external feature columns during fit.
  2. Merging features into the training DataFrame.
  3. Generating future feature values for prediction (forward-fill fallback).
  4. Accepting user-provided future features via ``set_future_features``.
"""

from typing import Any, Dict, List, Optional

import polars as pl

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


class MLForecastFeatureManager:
    """
    Manages external features for mlforecast models.

    Separates feature handling from model fitting/prediction logic
    so that ``_DirectMLBase`` doesn't need inline feature plumbing.
    """

    def __init__(self) -> None:
        self._feature_cols: List[str] = []
        self._train_pdf: Optional[Any] = None  # pandas DataFrame
        self._future_features: Optional[Any] = None  # pandas DataFrame

    @property
    def feature_cols(self) -> List[str]:
        """External feature column names detected during fit."""
        return list(self._feature_cols)

    @property
    def has_features(self) -> bool:
        return len(self._feature_cols) > 0

    def detect_features(
        self, df: pl.DataFrame, id_col: str, time_col: str, target_col: str
    ) -> List[str]:
        """Detect external feature columns (anything beyond id, time, target)."""
        core_cols = {id_col, time_col, target_col}
        self._feature_cols = [c for c in df.columns if c not in core_cols]
        return self._feature_cols

    def prepare_fit(
        self,
        df: pl.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
    ) -> "pd.DataFrame":
        """
        Build the pandas training DataFrame for mlforecast.

        Returns the full DataFrame (core + features) renamed to
        mlforecast conventions (unique_id, ds, y).
        """
        self.detect_features(df, id_col, time_col, target_col)

        # Core columns
        pdf = (
            df.select([id_col, time_col, target_col])
            .rename({id_col: "unique_id", time_col: "ds", target_col: "y"})
            .to_pandas()
        )
        pdf["ds"] = pdf["ds"].astype("datetime64[ns]")
        self._train_pdf = pdf

        if not self._feature_cols:
            return pdf

        # Merge external features
        exog_pdf = (
            df.select([id_col, time_col] + self._feature_cols)
            .rename({id_col: "unique_id", time_col: "ds"})
            .to_pandas()
        )
        exog_pdf["ds"] = exog_pdf["ds"].astype("datetime64[ns]")
        pdf_with_features = pdf.merge(exog_pdf, on=["unique_id", "ds"], how="left")
        pdf_with_features[self._feature_cols] = (
            pdf_with_features[self._feature_cols].fillna(0)
        )
        return pdf_with_features

    def prepare_predict(self, horizon: int) -> Optional["pd.DataFrame"]:
        """
        Build future feature DataFrame for mlforecast predict.

        Returns None if no external features are present.
        Uses user-provided future features if available, otherwise
        forward-fills the last known values per series.
        """
        if not self._feature_cols:
            return None

        # User-provided future features take priority
        if self._future_features is not None:
            return self._future_features

        # Forward-fill fallback
        if self._train_pdf is None:
            return None

        last_ds = self._train_pdf["ds"].max()
        uids = self._train_pdf["unique_id"].unique()
        future_rows = []
        for uid in uids:
            uid_data = self._train_pdf[self._train_pdf["unique_id"] == uid]
            last_row = uid_data.iloc[-1]
            for h in range(1, horizon + 1):
                row: Dict[str, Any] = {
                    "unique_id": uid,
                    "ds": last_ds + pd.Timedelta(weeks=h),
                }
                for col in self._feature_cols:
                    row[col] = (
                        float(last_row.get(col, 0))
                        if col in uid_data.columns
                        else 0.0
                    )
                future_rows.append(row)

        return pd.DataFrame(future_rows)

    def set_future_features(
        self,
        future_features: pl.DataFrame,
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> None:
        """Set user-provided external feature values for the forecast horizon."""
        if future_features is not None and not future_features.is_empty():
            self._future_features = (
                future_features
                .rename({id_col: "unique_id", time_col: "ds"})
                .to_pandas()
            )
            self._future_features["ds"] = self._future_features["ds"].astype(
                "datetime64[ns]"
            )

    @property
    def train_pdf(self) -> Optional["pd.DataFrame"]:
        """Stored training pandas DataFrame (for quantile model retraining)."""
        return self._train_pdf
