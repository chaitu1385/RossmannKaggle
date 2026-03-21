"""
Sparse / intermittent demand detection.

Classifies time series using the Syntetos-Boylan-Croston (SBC) demand
classification matrix:

              CV² < 0.49    CV² >= 0.49
  ADI < 1.32    Smooth        Erratic
  ADI >= 1.32   Intermittent  Lumpy

  ADI = Average Demand Interval = T / (number of non-zero periods)
  CV² = (std of non-zero demands / mean of non-zero demands)²

Smooth and erratic series → regular forecasting models
Intermittent and lumpy series → intermittent demand models
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl

# SBC recommended thresholds
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49


class SparseDetector:
    """
    Classify time series as sparse (intermittent/lumpy) or dense (smooth/erratic).

    Parameters
    ----------
    adi_threshold:
        ADI value above which a series is considered intermittent.
        Default 1.32 follows the Syntetos-Boylan-Croston recommendation.
    cv2_threshold:
        CV² value above which demand variation is considered erratic.
        Default 0.49 follows the Syntetos-Boylan-Croston recommendation.
    min_periods:
        Minimum number of total periods required to classify a series.
        Series below this threshold are labelled "insufficient_data" and
        treated as dense (regular models).
    min_nonzero:
        Minimum number of non-zero observations required for CV² estimation.
    """

    def __init__(
        self,
        adi_threshold: float = ADI_THRESHOLD,
        cv2_threshold: float = CV2_THRESHOLD,
        min_periods: int = 10,
        min_nonzero: int = 2,
    ):
        self.adi_threshold = adi_threshold
        self.cv2_threshold = cv2_threshold
        self.min_periods = min_periods
        self.min_nonzero = min_nonzero

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def classify(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        id_col: str = "series_id",
    ) -> pl.DataFrame:
        """
        Classify each series in the panel DataFrame.

        Returns
        -------
        DataFrame with columns:
          [id_col, "adi", "cv2", "demand_class", "is_sparse"]

        demand_class values:
          "smooth"            — ADI < threshold, CV² < threshold
          "intermittent"      — ADI >= threshold, CV² < threshold
          "erratic"           — ADI < threshold, CV² >= threshold
          "lumpy"             — ADI >= threshold, CV² >= threshold
          "insufficient_data" — too few periods or non-zero observations
        """
        records: List[Dict[str, Any]] = []
        for series_id in df[id_col].unique().to_list():
            series = df.filter(pl.col(id_col) == series_id)
            values = series[target_col].to_list()
            record = self._classify_single(series_id, values)
            record[id_col] = record.pop("_sid")
            records.append(record)

        if not records:
            return pl.DataFrame(
                schema={
                    id_col: pl.Utf8,
                    "adi": pl.Float64,
                    "cv2": pl.Float64,
                    "demand_class": pl.Utf8,
                    "is_sparse": pl.Boolean,
                }
            )

        return pl.DataFrame(records)

    def split(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        id_col: str = "series_id",
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split a panel DataFrame into dense and sparse subsets.

        Returns
        -------
        (dense_df, sparse_df)
        """
        classification = self.classify(df, target_col=target_col, id_col=id_col)
        sparse_ids = (
            classification
            .filter(pl.col("is_sparse"))
            [id_col]
            .to_list()
        )

        sparse_df = df.filter(pl.col(id_col).is_in(sparse_ids))
        dense_df = df.filter(~pl.col(id_col).is_in(sparse_ids))
        return dense_df, sparse_df

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _classify_single(self, series_id: Any, values: List[float]) -> Dict[str, Any]:
        """Return classification dict for one series."""
        base: Dict[str, Any] = {
            "_sid": series_id,
            "adi": None,
            "cv2": None,
            "demand_class": "insufficient_data",
            "is_sparse": False,
        }

        T = len(values)
        nonzero = [v for v in values if v > 0]
        n_nonzero = len(nonzero)

        if T < self.min_periods or n_nonzero < self.min_nonzero:
            return base

        # ADI: average inter-demand interval
        adi = float(T) / n_nonzero

        # CV²: squared coefficient of variation of non-zero demands
        nz_arr = np.array(nonzero, dtype=float)
        mean_nz = nz_arr.mean()
        if mean_nz <= 0:
            return base
        cv2 = float((nz_arr.std() / mean_nz) ** 2) if n_nonzero > 1 else 0.0

        # SBC classification matrix
        high_adi = adi >= self.adi_threshold
        high_cv2 = cv2 >= self.cv2_threshold

        if not high_adi and not high_cv2:
            demand_class = "smooth"
        elif high_adi and not high_cv2:
            demand_class = "intermittent"
        elif not high_adi and high_cv2:
            demand_class = "erratic"
        else:
            demand_class = "lumpy"

        return {
            "_sid": series_id,
            "adi": adi,
            "cv2": cv2,
            "demand_class": demand_class,
            "is_sparse": high_adi,
        }
