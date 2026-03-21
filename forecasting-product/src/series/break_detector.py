"""
Structural break detection for time series.

Detects level shifts, trend breaks, and regime changes that can make
historical data misleading for model training.  Supports two methods:

  - **PELT** (Pruned Exact Linear Time): via the ``ruptures`` library.
    Most accurate but requires an optional dependency.
  - **CUSUM** (Cumulative Sum): simple rolling-mean change-point detection.
    Zero-dependency fallback.

Follows the same per-series iteration pattern as ``SparseDetector``.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from ..config.schema import StructuralBreakConfig

logger = logging.getLogger(__name__)


@dataclass
class BreakReport:
    """Summary of structural breaks detected across all series."""

    total_series: int = 0
    series_with_breaks: int = 0
    total_breaks: int = 0
    per_series: Optional[pl.DataFrame] = None
    warnings: List[str] = field(default_factory=list)


class StructuralBreakDetector:
    """
    Detect structural breaks in panel time series data.

    Parameters
    ----------
    config : StructuralBreakConfig
        Detection settings (method, penalty, segment length, etc.).
    """

    def __init__(self, config: StructuralBreakConfig):
        self.config = config

    def detect(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> BreakReport:
        """
        Detect structural breaks in all series.

        Parameters
        ----------
        df : pl.DataFrame
            Panel data with time, target, and id columns.

        Returns
        -------
        BreakReport with per-series break information.
        """
        records: List[Dict[str, Any]] = []
        for sid in df[id_col].unique().sort().to_list():
            series = df.filter(pl.col(id_col) == sid).sort(time_col)
            record = self._detect_single(sid, series, target_col, time_col)
            records.append(record)

        return self._build_report(records, id_col)

    def truncate(
        self,
        df: pl.DataFrame,
        report: BreakReport,
        time_col: str,
        id_col: str,
    ) -> pl.DataFrame:
        """
        Truncate each series to data after its last detected break.

        Series without breaks are left unchanged.
        """
        if report.per_series is None or report.per_series.is_empty():
            return df

        # Get last break date per series
        break_info = report.per_series.filter(pl.col("n_breaks") > 0)
        if break_info.is_empty():
            return df

        parts = []
        series_with_breaks = set(break_info[id_col].to_list())

        for sid in df[id_col].unique().to_list():
            s = df.filter(pl.col(id_col) == sid)
            if sid not in series_with_breaks:
                parts.append(s)
                continue

            row = break_info.filter(pl.col(id_col) == sid)
            if row.is_empty():
                parts.append(s)
                continue

            last_break = row["last_break_date"][0]
            if last_break is not None:
                truncated = s.filter(pl.col(time_col) > last_break)
                if not truncated.is_empty():
                    parts.append(truncated)
                else:
                    # Keep at least the original data if truncation would empty it
                    parts.append(s)
            else:
                parts.append(s)

        if not parts:
            return df
        return pl.concat(parts)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _detect_single(
        self,
        sid: Any,
        series_df: pl.DataFrame,
        target_col: str,
        time_col: str,
    ) -> Dict[str, Any]:
        """Detect breaks in a single series."""
        values = series_df[target_col].to_numpy().astype(float)
        dates = series_df[time_col].to_list()

        if len(values) < 2 * self.config.min_segment_length:
            return {
                "_sid": sid,
                "n_breaks": 0,
                "break_indices": [],
                "break_dates": [],
                "last_break_date": None,
            }

        if self.config.method == "pelt":
            breakpoints = self._run_pelt(values)
        else:
            breakpoints = self._run_cusum(values)

        # Map indices to dates
        break_dates = [dates[i] for i in breakpoints if i < len(dates)]
        last_break = break_dates[-1] if break_dates else None

        return {
            "_sid": sid,
            "n_breaks": len(break_dates),
            "break_indices": breakpoints,
            "break_dates": break_dates,
            "last_break_date": last_break,
        }

    def _run_pelt(self, values: np.ndarray) -> List[int]:
        """Run PELT algorithm via ruptures library, falling back to CUSUM."""
        try:
            import ruptures as rpt
        except ImportError:
            logger.info("ruptures not installed; falling back to CUSUM method")
            return self._run_cusum(values)

        algo = rpt.Pelt(
            model=self.config.cost_model,
            min_size=self.config.min_segment_length,
        )
        algo.fit(values)
        breakpoints = algo.predict(pen=self.config.penalty)

        # ruptures returns breakpoints including the last index; remove it
        breakpoints = [b for b in breakpoints if b < len(values)]
        return breakpoints[: self.config.max_breakpoints]

    def _run_cusum(self, values: np.ndarray) -> List[int]:
        """
        CUSUM-based change-point detection.

        Uses a binary-segmentation approach: finds the split point that
        maximises the mean-shift statistic, checks against a threshold
        based on the *local* (within-segment) standard deviation, and
        recurses into each segment.  This avoids the problem where the
        overall std is inflated by the very level shifts we are trying
        to detect.
        """
        n = len(values)
        min_seg = self.config.min_segment_length

        if n < 2 * min_seg:
            return []

        breakpoints: List[int] = []
        self._cusum_recurse(values, 0, n, min_seg, breakpoints)
        breakpoints.sort()
        return breakpoints[: self.config.max_breakpoints]

    def _cusum_recurse(
        self,
        values: np.ndarray,
        start: int,
        end: int,
        min_seg: int,
        breakpoints: List[int],
    ) -> None:
        """Recursively find change points in values[start:end]."""
        n = end - start
        if n < 2 * min_seg:
            return
        if len(breakpoints) >= self.config.max_breakpoints:
            return

        segment = values[start:end]
        cumsum = np.cumsum(segment)
        total = cumsum[-1]

        # Find the split that maximises |left_mean - right_mean|
        best_score = 0.0
        best_idx = -1

        for i in range(min_seg, n - min_seg + 1):
            left_mean = cumsum[i - 1] / i
            right_mean = (total - cumsum[i - 1]) / (n - i)
            score = abs(left_mean - right_mean)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx < 0:
            return

        # Use within-segment std as the baseline for thresholding
        left_std = float(np.std(segment[:best_idx]))
        right_std = float(np.std(segment[best_idx:]))
        pooled_std = max(
            (left_std * best_idx + right_std * (n - best_idx)) / n,
            1e-9,
        )

        if best_score > self.config.penalty * pooled_std:
            abs_idx = start + best_idx
            breakpoints.append(abs_idx)
            # Recurse into left and right segments
            self._cusum_recurse(values, start, abs_idx, min_seg, breakpoints)
            self._cusum_recurse(values, abs_idx, end, min_seg, breakpoints)

    def _build_report(
        self, records: List[Dict[str, Any]], id_col: str
    ) -> BreakReport:
        """Build a BreakReport from per-series detection records."""
        if not records:
            return BreakReport()

        total_series = len(records)
        series_with_breaks = sum(1 for r in records if r["n_breaks"] > 0)
        total_breaks = sum(r["n_breaks"] for r in records)

        # Build per-series DataFrame
        per_series_records = []
        for r in records:
            per_series_records.append({
                id_col: r["_sid"],
                "n_breaks": r["n_breaks"],
                "last_break_date": r["last_break_date"],
            })
        per_series = pl.DataFrame(per_series_records)

        # Warnings
        warnings: List[str] = []
        if series_with_breaks > 0:
            warnings.append(
                f"{series_with_breaks} series have structural breaks "
                f"({total_breaks} total break points)"
            )

        return BreakReport(
            total_series=total_series,
            series_with_breaks=series_with_breaks,
            total_breaks=total_breaks,
            per_series=per_series,
            warnings=warnings,
        )
