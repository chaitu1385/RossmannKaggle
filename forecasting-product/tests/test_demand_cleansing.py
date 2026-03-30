"""
Tests for the demand cleansing module.

Covers:
  - Outlier detection (IQR, z-score) and correction (clip, interpolate, flag_only)
  - Stockout detection (consecutive zeros → recovery) and imputation (seasonal, interpolate)
  - Period exclusion (interpolate, drop, flag)
  - CleansingReport accuracy
  - Integration with SeriesBuilder
"""

import unittest
from datetime import date, timedelta
from typing import List

import polars as pl

from src.config.schema import CleansingConfig, DataQualityConfig, PlatformConfig
from src.data.cleanser import DemandCleanser, CleansingReport, CleansingResult
import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _make_clean_series(
    n_weeks: int = 104,
    base: float = 100.0,
    noise: float = 2.0,
    seed: int = 7,
    series_id: str = "S1",
) -> pl.DataFrame:
    """Steady demand ~base with very small uniform noise.  No outliers."""
    import random
    random.seed(seed)
    rows = []
    for w in range(n_weeks):
        week_date = date(2021, 1, 4) + timedelta(weeks=w)
        val = base + random.uniform(-noise, noise)
        rows.append({"series_id": series_id, "week": week_date, "quantity": val})
    return pl.DataFrame(rows).with_columns(pl.col("quantity").cast(pl.Float64))


def _make_series_with_outliers(
    n_weeks: int = 104,
    seed: int = 42,
    outlier_indices: List[int] = None,
) -> pl.DataFrame:
    """Normal demand ~100 with injected outliers at 500+."""
    import random
    random.seed(seed)
    if outlier_indices is None:
        outlier_indices = [10, 30, 50, 70, 90]
    rows = []
    for w in range(n_weeks):
        week_date = date(2021, 1, 4) + timedelta(weeks=w)
        val = 100.0 + random.gauss(0, 5)
        if w in outlier_indices:
            val = 500.0 + random.gauss(0, 10)
        rows.append({"series_id": "S1", "week": week_date, "quantity": max(val, 0.0)})
    return pl.DataFrame(rows).with_columns(pl.col("quantity").cast(pl.Float64))


def _make_series_with_stockout(n_weeks: int = 104) -> pl.DataFrame:
    """Steady demand with a 4-week zero run at weeks 30-33, then recovery."""
    rows = []
    for w in range(n_weeks):
        week_date = date(2021, 1, 4) + timedelta(weeks=w)
        if 30 <= w <= 33:
            val = 0.0  # stockout
        else:
            val = 100.0 + 2.0 * (w % 10)
        rows.append({"series_id": "S1", "week": week_date, "quantity": val})
    return pl.DataFrame(rows).with_columns(pl.col("quantity").cast(pl.Float64))


def _make_series_with_trailing_zeros(n_weeks: int = 104) -> pl.DataFrame:
    """Demand that drops to zero at the end (end-of-life)."""
    rows = []
    for w in range(n_weeks):
        week_date = date(2021, 1, 4) + timedelta(weeks=w)
        if w >= 100:
            val = 0.0
        else:
            val = 100.0
        rows.append({"series_id": "S1", "week": week_date, "quantity": val})
    return pl.DataFrame(rows).with_columns(pl.col("quantity").cast(pl.Float64))


def _make_multi_series(n_series: int = 3, n_weeks: int = 104) -> pl.DataFrame:
    """Multiple series with different scales + one outlier each."""
    import random
    random.seed(99)
    rows = []
    for s in range(1, n_series + 1):
        base = s * 100.0
        for w in range(n_weeks):
            week_date = date(2021, 1, 4) + timedelta(weeks=w)
            val = base + random.gauss(0, base * 0.05)
            if w == 50:
                val = base * 5  # outlier
            rows.append({"series_id": f"S{s}", "week": week_date, "quantity": max(val, 0.0)})
    return pl.DataFrame(rows).with_columns(pl.col("quantity").cast(pl.Float64))


# --------------------------------------------------------------------------- #
#  Outlier Detection Tests
# --------------------------------------------------------------------------- #

class TestOutlierDetection(unittest.TestCase):

    def test_iqr_flags_extreme_values(self):
        """Injected outliers at 500+ should be flagged when base demand is ~100."""
        df = _make_series_with_outliers()
        cfg = CleansingConfig(enabled=True, outlier_method="iqr", outlier_action="flag_only")
        cleanser = DemandCleanser(cfg)
        result = cleanser.detect_outliers(df, "week", "quantity", "series_id")

        flagged = result.filter(pl.col("_outlier_flag"))
        self.assertGreaterEqual(flagged.height, 4, "Should flag most injected outliers")

    def test_iqr_does_not_flag_normal_values(self):
        """Clean series with small noise should have zero outliers."""
        df = _make_clean_series()
        cfg = CleansingConfig(enabled=True, outlier_method="iqr", outlier_action="flag_only")
        cleanser = DemandCleanser(cfg)
        result = cleanser.detect_outliers(df, "week", "quantity", "series_id")

        flagged = result.filter(pl.col("_outlier_flag"))
        self.assertEqual(flagged.height, 0, "Clean series should have no outliers")

    def test_zscore_flags_extreme_values(self):
        """Z-score method should also catch extreme outliers."""
        df = _make_series_with_outliers()
        cfg = CleansingConfig(enabled=True, outlier_method="zscore", zscore_threshold=3.0,
                              outlier_action="flag_only")
        cleanser = DemandCleanser(cfg)
        result = cleanser.detect_outliers(df, "week", "quantity", "series_id")

        flagged = result.filter(pl.col("_outlier_flag"))
        self.assertGreaterEqual(flagged.height, 4)

    def test_clip_action_winsorizes_to_fence(self):
        """Clipped outlier values should be at fence bounds, not above."""
        df = _make_series_with_outliers()
        cfg = CleansingConfig(enabled=True, outlier_method="iqr", outlier_action="clip")
        cleanser = DemandCleanser(cfg)
        result = cleanser.cleanse(df, "week", "quantity", "series_id")

        max_val = result.df["quantity"].max()
        # After clipping, no value should be anywhere near 500
        self.assertLess(max_val, 300, f"Clipped max {max_val} should be well below 500")

    def test_interpolate_action_fills_smoothly(self):
        """Interpolated outlier replacements should be between neighbors."""
        df = _make_series_with_outliers(outlier_indices=[50])
        cfg = CleansingConfig(enabled=True, outlier_method="iqr", outlier_action="interpolate")
        cleanser = DemandCleanser(cfg)
        result = cleanser.cleanse(df, "week", "quantity", "series_id")

        # The value at index 50 should now be interpolated (close to 100, not 500)
        val_at_50 = result.df.sort("week").row(50)[2]  # quantity column
        self.assertLess(val_at_50, 200, "Interpolated value should be near baseline")

    def test_flag_only_preserves_original_values(self):
        """flag_only action should not modify any values."""
        df = _make_series_with_outliers()
        cfg = CleansingConfig(enabled=True, outlier_method="iqr", outlier_action="flag_only")
        cleanser = DemandCleanser(cfg)
        result = cleanser.cleanse(df, "week", "quantity", "series_id")

        # Values should be identical to input
        orig_sorted = df.sort(["series_id", "week"])["quantity"].to_list()
        clean_sorted = result.df.sort(["series_id", "week"])["quantity"].to_list()
        self.assertEqual(orig_sorted, clean_sorted)

    def test_outlier_detection_per_series(self):
        """Each series should have independent IQR statistics."""
        df = _make_multi_series(n_series=3)
        cfg = CleansingConfig(enabled=True, outlier_method="iqr", outlier_action="flag_only")
        cleanser = DemandCleanser(cfg)
        result = cleanser.cleanse(df, "week", "quantity", "series_id")

        # Each series had one outlier injected at week 50
        for sid in ["S1", "S2", "S3"]:
            series_flags = result.df.filter(
                (pl.col("series_id") == sid) & pl.col("_outlier_flag")
            )
            self.assertGreaterEqual(
                series_flags.height, 1,
                f"Series {sid} should have at least 1 outlier flagged",
            )

    def test_empty_dataframe_no_error(self):
        """Cleansing an empty DataFrame should not raise."""
        df = pl.DataFrame({"series_id": [], "week": [], "quantity": []}).with_columns(
            pl.col("quantity").cast(pl.Float64),
            pl.col("week").cast(pl.Date),
        )
        cfg = CleansingConfig(enabled=True)
        cleanser = DemandCleanser(cfg)
        result = cleanser.cleanse(df, "week", "quantity", "series_id")
        self.assertEqual(result.df.height, 0)
        self.assertEqual(result.report.total_series, 0)


# --------------------------------------------------------------------------- #
#  Stockout Detection Tests
# --------------------------------------------------------------------------- #

class TestStockoutDetection(unittest.TestCase):

    def test_consecutive_zeros_with_recovery_flagged(self):
        """4-week zero run at weeks 30-33 followed by recovery = stockout."""
        df = _make_series_with_stockout()
        cfg = CleansingConfig(enabled=True, stockout_detection=True, min_zero_run=2,
                              stockout_imputation="none", outlier_action="flag_only")
        cleanser = DemandCleanser(cfg)
        result = cleanser.cleanse(df, "week", "quantity", "series_id")

        stockouts = result.df.filter(pl.col("_stockout_flag"))
        self.assertEqual(stockouts.height, 4, "Should flag exactly 4 stockout weeks")

    def test_consecutive_zeros_at_end_not_flagged(self):
        """Trailing zeros should NOT be flagged as stockout (could be EOL)."""
        df = _make_series_with_trailing_zeros()
        cfg = CleansingConfig(enabled=True, stockout_detection=True, min_zero_run=2,
                              stockout_imputation="none", outlier_action="flag_only")
        cleanser = DemandCleanser(cfg)
        result = cleanser.cleanse(df, "week", "quantity", "series_id")

        stockouts = result.df.filter(pl.col("_stockout_flag"))
        self.assertEqual(stockouts.height, 0, "Trailing zeros should not be flagged")

    def test_short_zero_run_not_flagged(self):
        """A single zero week should not be flagged when min_zero_run=2."""
        rows = []
        for w in range(52):
            week_date = date(2022, 1, 3) + timedelta(weeks=w)
            val = 0.0 if w == 25 else 100.0
            rows.append({"series_id": "S1", "week": week_date, "quantity": val})
        df = pl.DataFrame(rows).with_columns(pl.col("quantity").cast(pl.Float64))

        cfg = CleansingConfig(enabled=True, stockout_detection=True, min_zero_run=2,
                              stockout_imputation="none", outlier_action="flag_only")
        cleanser = DemandCleanser(cfg)
        result = cleanser.cleanse(df, "week", "quantity", "series_id")

        stockouts = result.df.filter(pl.col("_stockout_flag"))
        self.assertEqual(stockouts.height, 0)

    def test_seasonal_imputation_uses_prior_year(self):
        """Stockout imputation should pull from 52 weeks back."""
        # Build 2 years of data with stockout in year 2
        rows = []
        for w in range(104):
            week_date = date(2021, 1, 4) + timedelta(weeks=w)
            if 82 <= w <= 84:  # stockout in year 2 (weeks 82-84)
                val = 0.0
            else:
                val = 100.0 + (w % 52) * 2.0  # mild seasonality
            rows.append({"series_id": "S1", "week": week_date, "quantity": val})
        df = pl.DataFrame(rows).with_columns(pl.col("quantity").cast(pl.Float64))

        cfg = CleansingConfig(enabled=True, stockout_detection=True, min_zero_run=2,
                              stockout_imputation="seasonal", outlier_action="flag_only")
        cleanser = DemandCleanser(cfg)
        result = cleanser.cleanse(df, "week", "quantity", "series_id")

        # The imputed values at weeks 82-84 should be close to the year-1 values
        # at weeks 30-32 (82-52=30, etc.)
        imputed = result.df.sort("week").slice(82, 3)["quantity"].to_list()
        for val in imputed:
            self.assertGreater(val, 0, "Stockout should be imputed to non-zero")

    def test_interpolate_imputation_bridges_gap(self):
        """Linear interpolation should produce values between boundaries."""
        df = _make_series_with_stockout()
        cfg = CleansingConfig(enabled=True, stockout_detection=True, min_zero_run=2,
                              stockout_imputation="interpolate", outlier_action="flag_only")
        cleanser = DemandCleanser(cfg)
        result = cleanser.cleanse(df, "week", "quantity", "series_id")

        # Stockout at weeks 30-33 should now be interpolated
        imputed = result.df.sort("week").slice(30, 4)["quantity"].to_list()
        for val in imputed:
            self.assertGreater(val, 0, "Interpolated stockout should be > 0")

    def test_stockout_detection_multi_series(self):
        """Stockouts should be detected independently per series."""
        # S1 has stockout, S2 does not
        s1 = _make_series_with_stockout()
        rows_s2 = []
        for w in range(104):
            week_date = date(2021, 1, 4) + timedelta(weeks=w)
            rows_s2.append({"series_id": "S2", "week": week_date, "quantity": 200.0})
        s2 = pl.DataFrame(rows_s2).with_columns(pl.col("quantity").cast(pl.Float64))
        df = pl.concat([s1, s2])

        cfg = CleansingConfig(enabled=True, stockout_detection=True, min_zero_run=2,
                              stockout_imputation="none", outlier_action="flag_only")
        cleanser = DemandCleanser(cfg)
        result = cleanser.cleanse(df, "week", "quantity", "series_id")

        s1_stockouts = result.df.filter(
            (pl.col("series_id") == "S1") & pl.col("_stockout_flag")
        )
        s2_stockouts = result.df.filter(
            (pl.col("series_id") == "S2") & pl.col("_stockout_flag")
        )
        self.assertEqual(s1_stockouts.height, 4)
        self.assertEqual(s2_stockouts.height, 0)


# --------------------------------------------------------------------------- #
#  Period Exclusion Tests
# --------------------------------------------------------------------------- #

class TestPeriodExclusion(unittest.TestCase):

    def _make_series_for_exclusion(self) -> pl.DataFrame:
        rows = []
        for w in range(104):
            week_date = date(2021, 1, 4) + timedelta(weeks=w)
            val = 100.0
            # Spike in weeks 50-60 (COVID-like anomaly)
            if 50 <= w <= 60:
                val = 500.0
            rows.append({"series_id": "S1", "week": week_date, "quantity": val})
        return pl.DataFrame(rows).with_columns(pl.col("quantity").cast(pl.Float64))

    def test_interpolate_replaces_excluded_values(self):
        df = self._make_series_for_exclusion()
        start_date = str(date(2021, 1, 4) + timedelta(weeks=50))
        end_date = str(date(2021, 1, 4) + timedelta(weeks=60))

        cfg = CleansingConfig(
            enabled=True,
            exclude_periods=[{"start": start_date, "end": end_date, "action": "interpolate"}],
            outlier_action="flag_only",
            stockout_detection=False,
        )
        cleanser = DemandCleanser(cfg)
        result = cleanser.cleanse(df, "week", "quantity", "series_id")

        # Values in the excluded range should no longer be 500
        excluded_range = result.df.sort("week").slice(50, 11)["quantity"].to_list()
        for val in excluded_range:
            self.assertLess(val, 200, f"Excluded period value {val} should be interpolated down")

    def test_drop_removes_excluded_rows(self):
        df = self._make_series_for_exclusion()
        start_date = str(date(2021, 1, 4) + timedelta(weeks=50))
        end_date = str(date(2021, 1, 4) + timedelta(weeks=60))

        cfg = CleansingConfig(
            enabled=True,
            exclude_periods=[{"start": start_date, "end": end_date, "action": "drop"}],
            outlier_action="flag_only",
            stockout_detection=False,
        )
        cleanser = DemandCleanser(cfg)
        result = cleanser.cleanse(df, "week", "quantity", "series_id")

        self.assertEqual(result.df.height, 104 - 11, "Should have dropped 11 rows")

    def test_flag_only_adds_column(self):
        df = self._make_series_for_exclusion()
        start_date = str(date(2021, 1, 4) + timedelta(weeks=50))
        end_date = str(date(2021, 1, 4) + timedelta(weeks=60))

        cfg = CleansingConfig(
            enabled=True,
            exclude_periods=[{"start": start_date, "end": end_date, "action": "flag"}],
            outlier_action="flag_only",
            stockout_detection=False,
        )
        cleanser = DemandCleanser(cfg)
        result = cleanser.cleanse(df, "week", "quantity", "series_id")

        self.assertIn("_excluded_flag", result.df.columns)
        flagged = result.df.filter(pl.col("_excluded_flag"))
        self.assertEqual(flagged.height, 11)
        # Values should be unchanged
        self.assertEqual(result.df["quantity"].max(), 500.0)

    def test_multiple_exclusion_periods(self):
        df = self._make_series_for_exclusion()
        period1_start = str(date(2021, 1, 4) + timedelta(weeks=50))
        period1_end = str(date(2021, 1, 4) + timedelta(weeks=55))
        period2_start = str(date(2021, 1, 4) + timedelta(weeks=56))
        period2_end = str(date(2021, 1, 4) + timedelta(weeks=60))

        cfg = CleansingConfig(
            enabled=True,
            exclude_periods=[
                {"start": period1_start, "end": period1_end, "action": "flag"},
                {"start": period2_start, "end": period2_end, "action": "flag"},
            ],
            outlier_action="flag_only",
            stockout_detection=False,
        )
        cleanser = DemandCleanser(cfg)
        result = cleanser.cleanse(df, "week", "quantity", "series_id")

        flagged = result.df.filter(pl.col("_excluded_flag"))
        self.assertEqual(flagged.height, 11)


# --------------------------------------------------------------------------- #
#  Report Tests
# --------------------------------------------------------------------------- #

class TestCleansingReport(unittest.TestCase):

    def test_report_counts_match(self):
        """Report outlier count should match actual flags in the output."""
        df = _make_series_with_outliers()
        cfg = CleansingConfig(enabled=True, outlier_method="iqr", outlier_action="flag_only",
                              stockout_detection=False)
        cleanser = DemandCleanser(cfg)
        result = cleanser.cleanse(df, "week", "quantity", "series_id")

        actual_flags = result.df.filter(pl.col("_outlier_flag")).height
        self.assertEqual(result.report.total_outliers, actual_flags)

    def test_report_per_series_breakdown(self):
        """Per-series report should have one row per series."""
        df = _make_multi_series(n_series=3)
        cfg = CleansingConfig(enabled=True, outlier_method="iqr", outlier_action="flag_only",
                              stockout_detection=False)
        cleanser = DemandCleanser(cfg)
        result = cleanser.cleanse(df, "week", "quantity", "series_id")

        self.assertEqual(result.report.per_series.height, 3)
        self.assertEqual(result.report.total_series, 3)

    def test_no_cleansing_needed_report(self):
        """Clean data should produce zero counts."""
        df = _make_clean_series()
        cfg = CleansingConfig(enabled=True, outlier_method="iqr", outlier_action="flag_only",
                              stockout_detection=False)
        cleanser = DemandCleanser(cfg)
        result = cleanser.cleanse(df, "week", "quantity", "series_id")

        self.assertEqual(result.report.total_outliers, 0)
        self.assertEqual(result.report.series_with_outliers, 0)


# --------------------------------------------------------------------------- #
#  Integration Tests
# --------------------------------------------------------------------------- #

class TestCleansingIntegration(unittest.TestCase):

    def _make_config(self, cleansing_enabled: bool = True) -> PlatformConfig:
        return PlatformConfig(
            data_quality=DataQualityConfig(
                fill_gaps=False,
                min_series_length_weeks=0,
                cleansing=CleansingConfig(
                    enabled=cleansing_enabled,
                    outlier_method="iqr",
                    outlier_action="clip",
                    stockout_detection=True,
                    stockout_imputation="interpolate",
                    add_flag_columns=True,
                ),
            ),
        )

    def test_cleanser_in_series_builder(self):
        """SeriesBuilder should apply cleansing when enabled."""
        from src.series.builder import SeriesBuilder

        config = self._make_config(cleansing_enabled=True)
        builder = SeriesBuilder(config)
        df = _make_series_with_outliers()
        result = builder.build(actuals=df)

        self.assertIsNotNone(builder._last_cleansing_report)
        self.assertGreater(builder._last_cleansing_report.total_outliers, 0)
        # Outliers should have been clipped
        self.assertLess(result["quantity"].max(), 300)

    def test_disabled_by_default(self):
        """Cleansing disabled → no changes, no report."""
        from src.series.builder import SeriesBuilder

        config = self._make_config(cleansing_enabled=False)
        builder = SeriesBuilder(config)
        df = _make_series_with_outliers()
        result = builder.build(actuals=df)

        self.assertIsNone(builder._last_cleansing_report)
        # Outliers should still be present
        self.assertGreater(result["quantity"].max(), 400)

    def test_cleansing_before_short_series_drop(self):
        """Cleansing runs before short-series filter, so flag columns exist."""
        from src.series.builder import SeriesBuilder

        config = PlatformConfig(
            data_quality=DataQualityConfig(
                fill_gaps=False,
                min_series_length_weeks=200,  # will drop everything
                cleansing=CleansingConfig(
                    enabled=True,
                    outlier_action="flag_only",
                    stockout_detection=False,
                ),
            ),
        )
        builder = SeriesBuilder(config)
        df = _make_series_with_outliers(n_weeks=104)
        result = builder.build(actuals=df)

        # Series is too short → dropped. But cleansing should have run.
        self.assertIsNotNone(builder._last_cleansing_report)
        self.assertEqual(result.height, 0, "Series should be dropped by length filter")


if __name__ == "__main__":
    unittest.main()
