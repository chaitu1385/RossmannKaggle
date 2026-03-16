"""Tests for forecastability signal computations."""

import math
import unittest

import numpy as np
import polars as pl

from src.analytics.forecastability import (
    ForecastabilityAnalyzer,
    SeriesSignals,
    compute_approximate_entropy,
    compute_cv,
    compute_forecastability_score,
    compute_seasonal_strength,
    compute_snr,
    compute_spectral_entropy,
    compute_trend_strength,
)


# --------------------------------------------------------------------------- #
#  Factory helpers
# --------------------------------------------------------------------------- #

def _make_sine_wave(n=104, period=52, amplitude=50.0, base=100.0, noise=1.0, seed=42):
    """Pure seasonal signal — high seasonal_strength, low spectral_entropy."""
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=float)
    return base + amplitude * np.sin(2 * np.pi * t / period) + rng.normal(0, noise, n)


def _make_white_noise(n=104, mean=100.0, std=20.0, seed=42):
    """Random — high entropy, low forecastability."""
    rng = np.random.RandomState(seed)
    return rng.normal(mean, std, n)


def _make_trending(n=104, slope=0.5, base=100.0, noise=2.0, seed=42):
    """Linear trend — high trend_strength."""
    rng = np.random.RandomState(seed)
    return base + slope * np.arange(n) + rng.normal(0, noise, n)


def _make_constant(n=104, value=100.0):
    """Constant series — zero variance."""
    return np.full(n, value)


def _make_intermittent(n=104, demand_prob=0.3, mean_demand=50.0, seed=42):
    """Sparse demand — many zeros with occasional bursts."""
    rng = np.random.RandomState(seed)
    mask = rng.random(n) < demand_prob
    values = np.zeros(n)
    values[mask] = rng.exponential(mean_demand, mask.sum())
    return values


def _make_forecastability_df(series_dict, n_weeks=104, seed=42):
    """Build Polars DataFrame with multiple named series."""
    rows = []
    base_date = np.datetime64("2020-01-06")
    for sid, values in series_dict.items():
        for i, v in enumerate(values):
            rows.append({
                "series_id": sid,
                "week": (base_date + np.timedelta64(i * 7, "D")).astype("datetime64[ms]").astype(object),
                "quantity": float(v),
            })
    return pl.DataFrame(rows).with_columns(pl.col("week").cast(pl.Date))


# --------------------------------------------------------------------------- #
#  Tests: Coefficient of Variation
# --------------------------------------------------------------------------- #

class TestCV(unittest.TestCase):
    def test_constant_series_zero_cv(self):
        assert compute_cv(_make_constant()) == 0.0

    def test_known_values(self):
        vals = np.array([10.0, 20.0, 30.0])
        cv = compute_cv(vals)
        expected = np.std(vals, ddof=1) / np.mean(vals)
        self.assertAlmostEqual(cv, expected, places=6)

    def test_high_variance_higher_cv(self):
        low_var = _make_sine_wave(noise=1.0)
        high_var = _make_white_noise(std=50.0)
        self.assertGreater(compute_cv(high_var), compute_cv(low_var))


# --------------------------------------------------------------------------- #
#  Tests: Approximate Entropy
# --------------------------------------------------------------------------- #

class TestApproximateEntropy(unittest.TestCase):
    def test_constant_zero_entropy(self):
        assert compute_approximate_entropy(_make_constant()) == 0.0

    def test_sine_lower_than_noise(self):
        sine_apen = compute_approximate_entropy(_make_sine_wave(noise=1.0))
        noise_apen = compute_approximate_entropy(_make_white_noise())
        self.assertLess(sine_apen, noise_apen)

    def test_short_series_returns_zero(self):
        assert compute_approximate_entropy(np.array([1.0, 2.0])) == 0.0

    def test_nonnegative(self):
        vals = _make_white_noise()
        self.assertGreaterEqual(compute_approximate_entropy(vals), 0.0)


# --------------------------------------------------------------------------- #
#  Tests: Spectral Entropy
# --------------------------------------------------------------------------- #

class TestSpectralEntropy(unittest.TestCase):
    def test_constant_zero(self):
        assert compute_spectral_entropy(_make_constant()) == 0.0

    def test_sine_low_entropy(self):
        # Pure sine with minimal noise → concentrated spectrum
        se = compute_spectral_entropy(_make_sine_wave(noise=0.01))
        self.assertLess(se, 0.5)

    def test_noise_high_entropy(self):
        se = compute_spectral_entropy(_make_white_noise())
        self.assertGreater(se, 0.7)

    def test_bounded_zero_one(self):
        for vals in [_make_sine_wave(), _make_white_noise(), _make_trending()]:
            se = compute_spectral_entropy(vals)
            self.assertGreaterEqual(se, 0.0)
            self.assertLessEqual(se, 1.0)


# --------------------------------------------------------------------------- #
#  Tests: Signal-to-Noise Ratio
# --------------------------------------------------------------------------- #

class TestSNR(unittest.TestCase):
    def test_clean_signal_high_snr(self):
        vals = _make_sine_wave(noise=0.1)
        self.assertGreater(compute_snr(vals), 2.0)

    def test_noisy_lower_snr(self):
        clean = compute_snr(_make_sine_wave(noise=0.1))
        noisy = compute_snr(_make_white_noise())
        self.assertGreater(clean, noisy)

    def test_short_series_fallback(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        snr = compute_snr(vals, season_length=52)
        self.assertIsInstance(snr, float)


# --------------------------------------------------------------------------- #
#  Tests: Trend Strength
# --------------------------------------------------------------------------- #

class TestTrendStrength(unittest.TestCase):
    def test_linear_trend_high(self):
        vals = _make_trending(noise=0.1)
        self.assertGreater(compute_trend_strength(vals), 0.8)

    def test_flat_series_low(self):
        vals = _make_white_noise()
        self.assertLess(compute_trend_strength(vals), 0.3)

    def test_constant_zero(self):
        assert compute_trend_strength(_make_constant()) == 0.0

    def test_bounded(self):
        ts = compute_trend_strength(_make_trending())
        self.assertGreaterEqual(ts, 0.0)
        self.assertLessEqual(ts, 1.0)


# --------------------------------------------------------------------------- #
#  Tests: Seasonal Strength
# --------------------------------------------------------------------------- #

class TestSeasonalStrength(unittest.TestCase):
    def test_strong_seasonality(self):
        vals = _make_sine_wave(noise=1.0, amplitude=50.0)
        self.assertGreater(compute_seasonal_strength(vals, season_length=52), 0.5)

    def test_noise_low_seasonality(self):
        vals = _make_white_noise()
        self.assertLess(compute_seasonal_strength(vals, season_length=52), 0.5)

    def test_short_series_returns_zero(self):
        vals = np.array([1.0, 2.0, 3.0])
        assert compute_seasonal_strength(vals, season_length=52) == 0.0


# --------------------------------------------------------------------------- #
#  Tests: Composite Forecastability Score
# --------------------------------------------------------------------------- #

class TestForecastabilityScore(unittest.TestCase):
    def test_predictable_series_high_score(self):
        sig = SeriesSignals(
            series_id="S1", cv=0.1, apen=0.1, spectral_entropy=0.2,
            snr=10.0, trend_strength=0.5, seasonal_strength=0.8,
            demand_class="smooth", forecastability_score=0.0,
        )
        score = compute_forecastability_score(sig)
        self.assertGreater(score, 0.6)

    def test_chaotic_series_low_score(self):
        sig = SeriesSignals(
            series_id="S2", cv=2.5, apen=2.0, spectral_entropy=0.9,
            snr=0.5, trend_strength=0.05, seasonal_strength=0.05,
            demand_class="erratic", forecastability_score=0.0,
        )
        score = compute_forecastability_score(sig)
        self.assertLess(score, 0.3)

    def test_bounded_zero_one(self):
        for apen, se, cv in [(0.0, 0.0, 0.0), (5.0, 1.0, 10.0)]:
            sig = SeriesSignals(
                series_id="X", cv=cv, apen=apen, spectral_entropy=se,
                snr=1.0, trend_strength=0.5, seasonal_strength=0.5,
                demand_class="smooth", forecastability_score=0.0,
            )
            score = compute_forecastability_score(sig)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_sine_beats_noise(self):
        """Sine wave should be more forecastable than white noise."""
        sine_vals = _make_sine_wave(noise=1.0)
        noise_vals = _make_white_noise()

        sine_sig = SeriesSignals(
            series_id="sine", cv=compute_cv(sine_vals),
            apen=compute_approximate_entropy(sine_vals),
            spectral_entropy=compute_spectral_entropy(sine_vals),
            snr=compute_snr(sine_vals),
            trend_strength=compute_trend_strength(sine_vals),
            seasonal_strength=compute_seasonal_strength(sine_vals),
            demand_class="smooth", forecastability_score=0.0,
        )
        noise_sig = SeriesSignals(
            series_id="noise", cv=compute_cv(noise_vals),
            apen=compute_approximate_entropy(noise_vals),
            spectral_entropy=compute_spectral_entropy(noise_vals),
            snr=compute_snr(noise_vals),
            trend_strength=compute_trend_strength(noise_vals),
            seasonal_strength=compute_seasonal_strength(noise_vals),
            demand_class="erratic", forecastability_score=0.0,
        )
        self.assertGreater(
            compute_forecastability_score(sine_sig),
            compute_forecastability_score(noise_sig),
        )


# --------------------------------------------------------------------------- #
#  Tests: ForecastabilityAnalyzer (end-to-end)
# --------------------------------------------------------------------------- #

class TestForecastabilityAnalyzer(unittest.TestCase):
    def setUp(self):
        self.df = _make_forecastability_df({
            "seasonal": _make_sine_wave(),
            "noisy": _make_white_noise(),
            "trending": _make_trending(),
        })
        self.analyzer = ForecastabilityAnalyzer(season_length=52)

    def test_report_fields_populated(self):
        report = self.analyzer.analyze(self.df, "quantity", "week", "series_id")
        self.assertEqual(report.n_series, 3)
        self.assertGreater(report.overall_score, 0.0)
        self.assertIn("high", report.score_distribution)
        self.assertIn("medium", report.score_distribution)
        self.assertIn("low", report.score_distribution)
        total = sum(report.score_distribution.values())
        self.assertEqual(total, 3)

    def test_per_series_dataframe(self):
        report = self.analyzer.analyze(self.df, "quantity", "week", "series_id")
        self.assertIsNotNone(report.per_series)
        self.assertEqual(report.per_series.height, 3)
        expected_cols = {"series_id", "cv", "apen", "spectral_entropy", "snr",
                         "trend_strength", "seasonal_strength", "demand_class",
                         "forecastability_score"}
        self.assertTrue(expected_cols.issubset(set(report.per_series.columns)))

    def test_seasonal_scores_highest(self):
        report = self.analyzer.analyze(self.df, "quantity", "week", "series_id")
        ps = report.per_series
        seasonal_score = ps.filter(pl.col("series_id") == "seasonal")["forecastability_score"][0]
        noisy_score = ps.filter(pl.col("series_id") == "noisy")["forecastability_score"][0]
        self.assertGreater(seasonal_score, noisy_score)
