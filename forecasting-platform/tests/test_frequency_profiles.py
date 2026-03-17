"""
Tests for multi-frequency support infrastructure.

Validates that FREQUENCY_PROFILES, helper functions, and frequency-aware
config properties work correctly for all supported frequencies (D/W/M/Q).
"""

import unittest
from datetime import date, timedelta

import polars as pl

from src.config.schema import (
    FREQUENCY_PROFILES,
    BacktestConfig,
    DataQualityConfig,
    ForecastConfig,
    freq_timedelta,
    get_frequency_profile,
)


class TestFrequencyProfiles(unittest.TestCase):
    """Tests for the FREQUENCY_PROFILES dict."""

    def test_all_four_frequencies_present(self):
        self.assertEqual(sorted(FREQUENCY_PROFILES.keys()), ["D", "M", "Q", "W"])

    def test_each_profile_has_required_keys(self):
        required = {
            "season_length", "secondary_season", "default_lags",
            "min_series_length", "default_val_periods", "default_horizon",
            "statsforecast_freq", "timedelta_kwargs",
        }
        for freq, profile in FREQUENCY_PROFILES.items():
            with self.subTest(freq=freq):
                self.assertTrue(
                    required.issubset(profile.keys()),
                    f"Missing keys for {freq}: {required - profile.keys()}"
                )

    def test_season_lengths(self):
        self.assertEqual(FREQUENCY_PROFILES["D"]["season_length"], 7)
        self.assertEqual(FREQUENCY_PROFILES["W"]["season_length"], 52)
        self.assertEqual(FREQUENCY_PROFILES["M"]["season_length"], 12)
        self.assertEqual(FREQUENCY_PROFILES["Q"]["season_length"], 4)

    def test_statsforecast_freq_strings(self):
        self.assertEqual(FREQUENCY_PROFILES["D"]["statsforecast_freq"], "D")
        self.assertEqual(FREQUENCY_PROFILES["W"]["statsforecast_freq"], "W")
        self.assertEqual(FREQUENCY_PROFILES["M"]["statsforecast_freq"], "MS")
        self.assertEqual(FREQUENCY_PROFILES["Q"]["statsforecast_freq"], "QS")

    def test_default_lags_are_sorted(self):
        for freq, profile in FREQUENCY_PROFILES.items():
            with self.subTest(freq=freq):
                lags = profile["default_lags"]
                self.assertEqual(lags, sorted(lags), f"Lags not sorted for {freq}")


class TestGetFrequencyProfile(unittest.TestCase):
    """Tests for the get_frequency_profile() helper."""

    def test_valid_frequencies(self):
        for freq in ("D", "W", "M", "Q"):
            profile = get_frequency_profile(freq)
            self.assertIsInstance(profile, dict)
            self.assertIn("season_length", profile)

    def test_invalid_frequency_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get_frequency_profile("Y")
        self.assertIn("Unsupported frequency", str(ctx.exception))
        self.assertIn("'Y'", str(ctx.exception))


class TestFreqTimedelta(unittest.TestCase):
    """Tests for the freq_timedelta() helper."""

    def test_daily(self):
        self.assertEqual(freq_timedelta("D"), timedelta(days=1))
        self.assertEqual(freq_timedelta("D", 7), timedelta(days=7))

    def test_weekly(self):
        self.assertEqual(freq_timedelta("W"), timedelta(weeks=1))
        self.assertEqual(freq_timedelta("W", 4), timedelta(weeks=4))

    def test_monthly(self):
        self.assertEqual(freq_timedelta("M"), timedelta(days=30))
        self.assertEqual(freq_timedelta("M", 3), timedelta(days=90))

    def test_quarterly(self):
        self.assertEqual(freq_timedelta("Q"), timedelta(days=91))
        self.assertEqual(freq_timedelta("Q", 2), timedelta(days=182))

    def test_date_arithmetic(self):
        """freq_timedelta should work with date addition."""
        d = date(2024, 1, 1)
        self.assertEqual(d + freq_timedelta("W", 1), date(2024, 1, 8))
        self.assertEqual(d + freq_timedelta("D", 1), date(2024, 1, 2))


class TestForecastConfigProperties(unittest.TestCase):
    """Tests for frequency-derived ForecastConfig properties."""

    def test_weekly_defaults(self):
        fc = ForecastConfig()
        self.assertEqual(fc.frequency, "W")
        self.assertEqual(fc.season_length, 52)
        self.assertEqual(fc.statsforecast_freq, "W")
        self.assertEqual(fc.horizon_periods, 39)

    def test_daily_frequency(self):
        fc = ForecastConfig(frequency="D")
        self.assertEqual(fc.season_length, 7)
        self.assertEqual(fc.statsforecast_freq, "D")
        self.assertEqual(fc.default_lags, [1, 2, 3, 7, 14, 21, 28, 56, 91, 182, 364])

    def test_monthly_frequency(self):
        fc = ForecastConfig(frequency="M")
        self.assertEqual(fc.season_length, 12)
        self.assertEqual(fc.statsforecast_freq, "MS")
        self.assertEqual(fc.default_lags, [1, 2, 3, 6, 12])

    def test_quarterly_frequency(self):
        fc = ForecastConfig(frequency="Q")
        self.assertEqual(fc.season_length, 4)
        self.assertEqual(fc.statsforecast_freq, "QS")


class TestBacktestConfigAliases(unittest.TestCase):
    """Tests for BacktestConfig period aliases."""

    def test_val_periods_alias(self):
        bt = BacktestConfig(val_weeks=8)
        self.assertEqual(bt.val_periods, 8)

    def test_gap_periods_alias(self):
        bt = BacktestConfig(gap_weeks=2)
        self.assertEqual(bt.gap_periods, 2)


class TestDataQualityConfigAlias(unittest.TestCase):
    """Tests for DataQualityConfig min_series_length alias."""

    def test_min_series_length_alias(self):
        dq = DataQualityConfig(min_series_length_weeks=24)
        self.assertEqual(dq.min_series_length, 24)


class TestNaiveForecasterFrequency(unittest.TestCase):
    """Tests that SeasonalNaiveForecaster respects frequency parameter."""

    def test_monthly_season_length(self):
        from src.forecasting.naive import SeasonalNaiveForecaster
        f = SeasonalNaiveForecaster(frequency="M")
        self.assertEqual(f.season_length, 12)
        self.assertEqual(f.frequency, "M")

    def test_weekly_default_unchanged(self):
        from src.forecasting.naive import SeasonalNaiveForecaster
        f = SeasonalNaiveForecaster()
        self.assertEqual(f.season_length, 52)
        self.assertEqual(f.frequency, "W")

    def test_explicit_season_length_preserved(self):
        from src.forecasting.naive import SeasonalNaiveForecaster
        f = SeasonalNaiveForecaster(season_length=26, frequency="M")
        self.assertEqual(f.season_length, 26)

    def test_monthly_predict_dates(self):
        """Monthly frequency should produce ~30-day-apart forecast dates."""
        from src.forecasting.naive import SeasonalNaiveForecaster
        f = SeasonalNaiveForecaster(frequency="M")
        df = pl.DataFrame({
            "series_id": ["A"] * 24,
            "week": [date(2022, 1, 1) + timedelta(days=30 * i) for i in range(24)],
            "quantity": [float(i % 12 + 1) for i in range(24)],
        })
        f.fit(df)
        preds = f.predict(3)
        dates = preds["week"].sort().to_list()
        # Check that forecast dates are roughly monthly apart
        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i - 1]).days
            self.assertGreaterEqual(gap, 28)
            self.assertLessEqual(gap, 32)


class TestRegistryFrequencyFiltering(unittest.TestCase):
    """Tests that registry.build() handles unknown kwargs gracefully."""

    def test_build_with_frequency(self):
        from src.forecasting.registry import registry
        f = registry.build("naive_seasonal", frequency="M")
        self.assertEqual(f.frequency, "M")

    def test_build_drops_unknown_kwargs(self):
        """Models that don't accept frequency should not fail."""
        from src.forecasting.registry import registry
        # Croston doesn't accept frequency — should not raise
        try:
            f = registry.build("croston", frequency="M")
            self.assertEqual(f.name, "croston")
        except TypeError:
            self.fail("registry.build() should drop unknown kwargs")


if __name__ == "__main__":
    unittest.main()
