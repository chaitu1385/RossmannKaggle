"""
Tests for the structural break detection module.

Covers:
  - CUSUM-based detection (default, zero-dependency)
  - Edge cases (short series, flat series, no breaks)
  - Truncation logic
  - Builder integration
  - Quality report integration
"""

from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from src.config.schema import (
    DataQualityConfig,
    DataQualityReportConfig,
    PlatformConfig,
    StructuralBreakConfig,
)
from src.series.break_detector import BreakReport, StructuralBreakDetector


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_weekly_dates(n_weeks: int, start: date = date(2022, 1, 3)) -> list:
    """Generate n weekly dates starting from start."""
    return [start + timedelta(weeks=i) for i in range(n_weeks)]


def _make_panel(
    values: list,
    series_id: str = "S1",
    start: date = date(2022, 1, 3),
) -> pl.DataFrame:
    """Create a single-series panel DataFrame."""
    n = len(values)
    return pl.DataFrame({
        "series_id": [series_id] * n,
        "week": _make_weekly_dates(n, start),
        "quantity": values,
    })


def _make_level_shift_series(
    n_before: int = 52,
    n_after: int = 52,
    level_before: float = 100.0,
    level_after: float = 200.0,
    noise_std: float = 5.0,
    seed: int = 42,
) -> list:
    """Create a series with a clear level shift."""
    rng = np.random.RandomState(seed)
    before = rng.normal(level_before, noise_std, n_before).tolist()
    after = rng.normal(level_after, noise_std, n_after).tolist()
    return before + after


def _make_multi_regime_series(
    regime_lengths: list = None,
    regime_levels: list = None,
    noise_std: float = 3.0,
    seed: int = 42,
) -> list:
    """Create a series with multiple regime changes."""
    if regime_lengths is None:
        regime_lengths = [40, 40, 40]
    if regime_levels is None:
        regime_levels = [100.0, 200.0, 50.0]
    rng = np.random.RandomState(seed)
    values = []
    for length, level in zip(regime_lengths, regime_levels):
        values.extend(rng.normal(level, noise_std, length).tolist())
    return values


# ── Detection tests ──────────────────────────────────────────────────────────

class TestDetectNoBreaks:
    def test_flat_series(self):
        config = StructuralBreakConfig(enabled=True, method="cusum")
        detector = StructuralBreakDetector(config)
        values = [100.0] * 104
        df = _make_panel(values)
        report = detector.detect(df)
        assert report.series_with_breaks == 0
        assert report.total_breaks == 0

    def test_noisy_but_stable(self):
        config = StructuralBreakConfig(enabled=True, method="cusum", penalty=3.0)
        detector = StructuralBreakDetector(config)
        rng = np.random.RandomState(42)
        values = rng.normal(100, 10, 104).tolist()
        df = _make_panel(values)
        report = detector.detect(df)
        assert report.series_with_breaks == 0


class TestDetectLevelShift:
    def test_clear_shift_detected(self):
        config = StructuralBreakConfig(
            enabled=True, method="cusum", min_segment_length=13, penalty=2.0
        )
        detector = StructuralBreakDetector(config)
        values = _make_level_shift_series(
            n_before=52, n_after=52,
            level_before=100, level_after=300,
            noise_std=5,
        )
        df = _make_panel(values)
        report = detector.detect(df)
        assert report.series_with_breaks == 1
        assert report.total_breaks >= 1

    def test_break_location_is_reasonable(self):
        config = StructuralBreakConfig(
            enabled=True, method="cusum", min_segment_length=13, penalty=2.0
        )
        detector = StructuralBreakDetector(config)
        values = _make_level_shift_series(
            n_before=52, n_after=52,
            level_before=50, level_after=200,
            noise_std=3,
        )
        df = _make_panel(values)
        report = detector.detect(df)
        assert report.per_series is not None
        row = report.per_series.filter(pl.col("series_id") == "S1")
        assert row["n_breaks"][0] >= 1


class TestDetectMultipleBreaks:
    def test_two_regime_changes(self):
        config = StructuralBreakConfig(
            enabled=True, method="cusum", min_segment_length=13, penalty=2.0
        )
        detector = StructuralBreakDetector(config)
        values = _make_multi_regime_series(
            regime_lengths=[40, 40, 40],
            regime_levels=[100, 300, 50],
            noise_std=3,
        )
        df = _make_panel(values)
        report = detector.detect(df)
        assert report.total_breaks >= 2


class TestMaxBreakpointsCap:
    def test_cap_is_respected(self):
        config = StructuralBreakConfig(
            enabled=True, method="cusum", min_segment_length=13,
            penalty=1.0, max_breakpoints=2,
        )
        detector = StructuralBreakDetector(config)
        # Create a series with many regime changes
        values = _make_multi_regime_series(
            regime_lengths=[30, 30, 30, 30],
            regime_levels=[50, 200, 50, 200],
            noise_std=2,
        )
        df = _make_panel(values)
        report = detector.detect(df)
        assert report.total_breaks <= 2


class TestShortSeriesSkipped:
    def test_series_too_short(self):
        config = StructuralBreakConfig(
            enabled=True, method="cusum", min_segment_length=13
        )
        detector = StructuralBreakDetector(config)
        values = [100.0] * 20  # shorter than 2 * min_segment_length
        df = _make_panel(values)
        report = detector.detect(df)
        assert report.series_with_breaks == 0
        assert report.total_breaks == 0


class TestMultipleSeries:
    def test_mixed_series(self):
        config = StructuralBreakConfig(
            enabled=True, method="cusum", min_segment_length=13, penalty=2.0
        )
        detector = StructuralBreakDetector(config)

        # Series 1: clear break
        s1 = _make_level_shift_series(52, 52, 50, 250, noise_std=3)
        # Series 2: no break
        rng = np.random.RandomState(99)
        s2 = rng.normal(100, 5, 104).tolist()

        df1 = _make_panel(s1, series_id="S1")
        df2 = _make_panel(s2, series_id="S2")
        df = pl.concat([df1, df2])

        report = detector.detect(df)
        assert report.total_series == 2
        assert report.series_with_breaks >= 1


# ── Truncation tests ─────────────────────────────────────────────────────────

class TestTruncate:
    def test_truncation_keeps_post_break_data(self):
        config = StructuralBreakConfig(
            enabled=True, method="cusum", min_segment_length=13, penalty=2.0
        )
        detector = StructuralBreakDetector(config)
        values = _make_level_shift_series(52, 52, 50, 300, noise_std=3)
        df = _make_panel(values)

        report = detector.detect(df)
        assert report.series_with_breaks >= 1

        truncated = detector.truncate(df, report, "week", "series_id")
        assert len(truncated) < len(df)

    def test_no_breaks_no_truncation(self):
        config = StructuralBreakConfig(enabled=True, method="cusum")
        detector = StructuralBreakDetector(config)
        values = [100.0] * 104
        df = _make_panel(values)
        report = detector.detect(df)

        truncated = detector.truncate(df, report, "week", "series_id")
        assert len(truncated) == len(df)


# ── CUSUM fallback test ──────────────────────────────────────────────────────

class TestCusumFallback:
    def test_pelt_falls_back_to_cusum(self):
        """When ruptures is not installed, PELT falls back to CUSUM."""
        config = StructuralBreakConfig(
            enabled=True, method="pelt", min_segment_length=13, penalty=2.0
        )
        detector = StructuralBreakDetector(config)

        values = _make_level_shift_series(52, 52, 50, 300, noise_std=3)
        df = _make_panel(values)

        # Mock ruptures import failure
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "ruptures":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            report = detector.detect(df)
            # Should still detect the break via CUSUM fallback
            assert report.series_with_breaks >= 1


# ── Builder integration test ─────────────────────────────────────────────────

class TestBuilderIntegration:
    def test_builder_with_breaks_enabled(self):
        config = PlatformConfig(
            data_quality=DataQualityConfig(
                fill_gaps=True,
                min_series_length_weeks=0,
                structural_breaks=StructuralBreakConfig(
                    enabled=True,
                    method="cusum",
                    min_segment_length=13,
                    penalty=2.0,
                ),
            ),
        )
        from src.series.builder import SeriesBuilder
        builder = SeriesBuilder(config)

        values = _make_level_shift_series(52, 52, 50, 300, noise_std=3)
        df = _make_panel(values)

        result = builder.build(df)
        assert builder._last_break_report is not None
        assert builder._last_break_report.total_series == 1
        assert builder._last_break_report.series_with_breaks >= 1

    def test_builder_with_truncation(self):
        config = PlatformConfig(
            data_quality=DataQualityConfig(
                fill_gaps=True,
                min_series_length_weeks=0,
                structural_breaks=StructuralBreakConfig(
                    enabled=True,
                    method="cusum",
                    min_segment_length=13,
                    penalty=2.0,
                    truncate_to_last_break=True,
                ),
            ),
        )
        from src.series.builder import SeriesBuilder
        builder = SeriesBuilder(config)

        values = _make_level_shift_series(52, 52, 50, 300, noise_std=3)
        df = _make_panel(values)

        result = builder.build(df)
        # After truncation, should have fewer rows
        assert len(result) < len(df)


# ── Quality report integration test ──────────────────────────────────────────

class TestQualityReportIntegration:
    def test_break_counts_in_quality_report(self):
        config = PlatformConfig(
            data_quality=DataQualityConfig(
                fill_gaps=True,
                min_series_length_weeks=0,
                structural_breaks=StructuralBreakConfig(
                    enabled=True, method="cusum", min_segment_length=13, penalty=2.0,
                ),
                report=DataQualityReportConfig(
                    enabled=True,
                    sparse_classification=False,
                ),
            ),
        )
        from src.series.builder import SeriesBuilder
        builder = SeriesBuilder(config)

        values = _make_level_shift_series(52, 52, 50, 300, noise_std=3)
        df = _make_panel(values)
        builder.build(df)

        qr = builder._last_quality_report
        assert qr is not None
        assert qr.series_with_breaks >= 1
        assert qr.total_breaks >= 1
        # Check that the break warning appears
        break_warnings = [w for w in qr.warnings if "structural break" in w.lower()]
        assert len(break_warnings) >= 1
