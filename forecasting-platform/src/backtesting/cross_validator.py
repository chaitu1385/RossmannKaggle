"""
Walk-forward cross-validation splitter.

Creates expanding-window training sets and fixed-size validation windows
that move forward in time.  This is the standard temporal CV scheme for
time series — no future data leaks into training.

Example with n_folds=3, val_weeks=13::

    ──────────────────────────────────────────────────────────────→ time
    [═══════ TRAIN ═══════][── VAL 1 ──]
    [════════════ TRAIN ════════════][── VAL 2 ──]
    [═══════════════════ TRAIN ═══════════════════][── VAL 3 ──]
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Tuple

import polars as pl


@dataclass
class CVFold:
    """A single cross-validation fold."""
    fold_index: int
    train_start: date
    train_end: date
    val_start: date
    val_end: date


class WalkForwardCV:
    """
    Walk-forward expanding-window cross-validation.

    The last ``n_folds * val_weeks`` of data are reserved for validation.
    Each fold's training set expands to include all prior data.
    """

    def __init__(
        self,
        n_folds: int = 3,
        val_weeks: int = 13,
        gap_weeks: int = 0,
    ):
        """
        Parameters
        ----------
        n_folds:
            Number of validation folds.
        val_weeks:
            Weeks in each validation window.
        gap_weeks:
            Gap between train end and val start (to simulate production lag).
        """
        self.n_folds = n_folds
        self.val_weeks = val_weeks
        self.gap_weeks = gap_weeks

    def split(
        self,
        df: pl.DataFrame,
        time_col: str = "week",
    ) -> List[CVFold]:
        """
        Compute fold boundaries from the data's date range.

        Returns a list of ``CVFold`` objects (no data copying).
        The caller uses ``fold.train_end`` / ``fold.val_start`` to filter.
        """
        min_date = df[time_col].min()
        max_date = df[time_col].max()

        if min_date is None or max_date is None:
            return []

        # Work backwards from max_date
        folds: List[CVFold] = []
        for i in range(self.n_folds - 1, -1, -1):
            val_end = max_date - timedelta(weeks=i * self.val_weeks)
            val_start = val_end - timedelta(weeks=self.val_weeks - 1)
            train_end = val_start - timedelta(weeks=self.gap_weeks + 1)

            if train_end < min_date:
                continue

            folds.append(CVFold(
                fold_index=self.n_folds - 1 - i,
                train_start=min_date,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
            ))

        return folds

    def split_data(
        self,
        df: pl.DataFrame,
        time_col: str = "week",
    ) -> List[Tuple[CVFold, pl.DataFrame, pl.DataFrame]]:
        """
        Split the data into (fold, train_df, val_df) tuples.

        This is the convenience method most callers will use.
        """
        folds = self.split(df, time_col)
        results = []

        for fold in folds:
            train = df.filter(
                (pl.col(time_col) >= fold.train_start)
                & (pl.col(time_col) <= fold.train_end)
            )
            val = df.filter(
                (pl.col(time_col) >= fold.val_start)
                & (pl.col(time_col) <= fold.val_end)
            )
            results.append((fold, train, val))

        return results
