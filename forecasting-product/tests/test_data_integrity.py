"""
Data integrity tests — verifying that ML pipelines don't leak, contaminate,
or misuse features.

Covers:
  - fill_weekly_gaps forward-fill vs zero-fill behaviour
  - Temporal causality validation for external regressors
  - Contemporaneous feature dropping at prediction time
  - ML beats naive on data with clear seasonal signal (regression guard)
"""

import math
import random
import unittest
from datetime import date, timedelta
from typing import List

import polars as pl
import pytest

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _make_seasonal_series(
    n_series: int = 3,
    n_weeks: int = 156,  # 3 years
    seed: int = 7,
) -> pl.DataFrame:
    """
    Synthetic data with clear weekly seasonality + trend.

    This is designed so that a competent ML model with lag/rolling features
    should beat seasonal naive.
    """
    random.seed(seed)
    rows: List[dict] = []
    for s in range(1, n_series + 1):
        base = 500.0 + s * 100
        for w in range(n_weeks):
            week_date = date(2020, 1, 5) + timedelta(weeks=w)  # Sunday
            # Seasonality: higher in weeks 45-52 (holiday) and 10-15 (spring)
            woy = week_date.isocalendar()[1]
            seasonal = 80 * math.sin(2 * math.pi * woy / 52)
            # Trend: slow growth
            trend = 0.5 * w
            noise = random.gauss(0, 15)
            rows.append({
                "series_id": f"S{s}",
                "week": week_date,
                "quantity": max(base + seasonal + trend + noise, 0.0),
            })
    return pl.DataFrame(rows).with_columns(pl.col("quantity").cast(pl.Float64))


def _make_gapped_series() -> pl.DataFrame:
    """Series with intentional gaps (missing weeks)."""
    rows = []
    for w in range(52):
        week_date = date(2022, 1, 2) + timedelta(weeks=w)
        # Skip weeks 10, 11, 12 to create a gap
        if w in (10, 11, 12):
            continue
        rows.append({
            "series_id": "G1",
            "week": week_date,
            "quantity": 100.0 + 5 * w,
        })
    return pl.DataFrame(rows).with_columns(pl.col("quantity").cast(pl.Float64))


# --------------------------------------------------------------------------- #
#  fill_weekly_gaps tests
# --------------------------------------------------------------------------- #

class TestFillWeeklyGaps(unittest.TestCase):

    def test_forward_fill_produces_nonzero_for_gaps(self):
        """Forward-fill strategy should propagate the last known value, not zero."""
        from src.forecasting.base import BaseForecaster

        gapped = _make_gapped_series()
        filled = BaseForecaster.fill_weekly_gaps(
            gapped, strategy="forward_fill"
        )
        # Weeks 10-12 should now be filled
        self.assertEqual(filled.shape[0], 52)
        gap_weeks = filled.filter(
            pl.col("week").is_between(
                date(2022, 3, 13), date(2022, 3, 27),  # weeks 10-12
            )
        )
        # Values should be forward-filled from week 9 (not zero)
        for val in gap_weeks["quantity"].to_list():
            self.assertGreater(val, 0.0, "Forward-fill should not produce zeros for gaps")

    def test_zero_fill_produces_zeros_for_gaps(self):
        """Zero-fill (default) should fill gaps with 0."""
        from src.forecasting.base import BaseForecaster

        gapped = _make_gapped_series()
        filled = BaseForecaster.fill_weekly_gaps(gapped, strategy="zero")
        gap_weeks = filled.filter(
            pl.col("week").is_between(
                date(2022, 3, 13), date(2022, 3, 27),
            )
        )
        for val in gap_weeks["quantity"].to_list():
            self.assertAlmostEqual(val, 0.0)

    def test_forward_fill_preserves_real_values(self):
        """Forward-fill should not alter existing non-null values."""
        from src.forecasting.base import BaseForecaster

        gapped = _make_gapped_series()
        original_values = gapped["quantity"].to_list()

        filled = BaseForecaster.fill_weekly_gaps(
            gapped, strategy="forward_fill"
        )
        # Join back to original to check values weren't altered
        merged = gapped.join(
            filled, on=["series_id", "week"], how="inner", suffix="_filled"
        )
        for orig, fil in zip(
            merged["quantity"].to_list(),
            merged["quantity_filled"].to_list(),
        ):
            self.assertAlmostEqual(orig, fil, places=6)


class TestFillGaps(unittest.TestCase):
    """Tests for the new frequency-aware fill_gaps static method."""

    def test_fill_gaps_weekly_matches_legacy(self):
        """fill_gaps(freq='W') should produce the same result as fill_weekly_gaps."""
        import warnings
        from src.forecasting.base import BaseForecaster

        gapped = _make_gapped_series()
        new_result = BaseForecaster.fill_gaps(gapped, strategy="forward_fill", freq="W")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy_result = BaseForecaster.fill_weekly_gaps(gapped, strategy="forward_fill")
        self.assertEqual(new_result.shape, legacy_result.shape)
        self.assertEqual(
            new_result["quantity"].to_list(),
            legacy_result["quantity"].to_list(),
        )

    def test_fill_gaps_daily(self):
        """fill_gaps(freq='D') should fill daily gaps correctly."""
        from src.forecasting.base import BaseForecaster

        # Create daily series with a 3-day gap
        rows = []
        for d in range(14):
            day = date(2024, 1, 1) + timedelta(days=d)
            if d in (5, 6, 7):  # skip days 5-7
                continue
            rows.append({"series_id": "D1", "date": day, "quantity": 10.0 + d})
        gapped = pl.DataFrame(rows).with_columns(pl.col("quantity").cast(pl.Float64))

        filled = BaseForecaster.fill_gaps(
            gapped, time_col="date", strategy="forward_fill", freq="D",
        )
        self.assertEqual(filled.shape[0], 14)  # all 14 days present
        # Day 5 should be forward-filled from day 4 (value 14.0)
        day5 = filled.filter(pl.col("date") == date(2024, 1, 6))
        self.assertAlmostEqual(day5["quantity"][0], 14.0)

    def test_fill_weekly_gaps_emits_deprecation_warning(self):
        """fill_weekly_gaps should emit a DeprecationWarning."""
        import warnings
        from src.forecasting.base import BaseForecaster

        gapped = _make_gapped_series()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BaseForecaster.fill_weekly_gaps(gapped, strategy="zero")
            deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
            self.assertEqual(len(deprecations), 1)
            self.assertIn("fill_gaps", str(deprecations[0].message))


# --------------------------------------------------------------------------- #
#  Regressor validation tests
# --------------------------------------------------------------------------- #

class TestRegressorTemporalCausality(unittest.TestCase):

    def test_contemporaneous_feature_flagged(self):
        """validate_regressors should warn about contemporaneous features without future values."""
        from src.data.regressors import validate_regressors

        actuals = pl.DataFrame({
            "week": [date(2023, 1, 1) + timedelta(weeks=w) for w in range(52)],
            "quantity": [100.0] * 52,
        })
        # External features only cover training period (no future)
        features = pl.DataFrame({
            "week": [date(2023, 1, 1) + timedelta(weeks=w) for w in range(52)],
            "promo_ratio": [0.5] * 52,
        })
        issues = validate_regressors(
            external_features=features,
            actuals=actuals,
            feature_columns=["promo_ratio"],
            horizon_weeks=13,
            feature_types={"promo_ratio": "contemporaneous"},
        )
        contemporaneous_issues = [i for i in issues if "contemporaneous" in i]
        self.assertTrue(
            len(contemporaneous_issues) > 0,
            "Should flag contemporaneous feature without future values",
        )

    def test_known_ahead_not_flagged(self):
        """known_ahead features should not trigger temporal causality warning."""
        from src.data.regressors import validate_regressors

        actuals = pl.DataFrame({
            "week": [date(2023, 1, 1) + timedelta(weeks=w) for w in range(52)],
            "quantity": [100.0] * 52,
        })
        # Features cover beyond actuals
        features = pl.DataFrame({
            "week": [date(2023, 1, 1) + timedelta(weeks=w) for w in range(65)],
            "holiday_flag": [0] * 65,
        })
        issues = validate_regressors(
            external_features=features,
            actuals=actuals,
            feature_columns=["holiday_flag"],
            horizon_weeks=13,
            feature_types={"holiday_flag": "known_ahead"},
        )
        contemporaneous_issues = [i for i in issues if "contemporaneous" in i]
        self.assertEqual(len(contemporaneous_issues), 0)


# --------------------------------------------------------------------------- #
#  Feature manager tests
# --------------------------------------------------------------------------- #

class TestFeatureManagerTemporalValidation(unittest.TestCase):

    def test_contemporaneous_feature_dropped_without_future(self):
        """Contemporaneous features should be dropped when no future values provided."""
        from src.forecasting.feature_manager import MLForecastFeatureManager

        mgr = MLForecastFeatureManager(
            feature_types={"promo_ratio": "contemporaneous"}
        )
        # Simulate detected features
        mgr._feature_cols = ["promo_ratio"]

        eligible = mgr._eligible_predict_cols()
        self.assertEqual(eligible, [], "Contemporaneous features should be dropped")

    def test_known_ahead_feature_kept_without_future(self):
        """known_ahead features should be eligible for forward-fill fallback."""
        from src.forecasting.feature_manager import MLForecastFeatureManager

        mgr = MLForecastFeatureManager(
            feature_types={"holiday_flag": "known_ahead"}
        )
        mgr._feature_cols = ["holiday_flag"]

        eligible = mgr._eligible_predict_cols()
        self.assertEqual(eligible, ["holiday_flag"])

    def test_all_features_eligible_with_explicit_future(self):
        """All features are eligible when explicit future values are provided."""
        import datetime
        import polars as pl
        from src.forecasting.feature_manager import MLForecastFeatureManager

        mgr = MLForecastFeatureManager(
            feature_types={"promo_ratio": "contemporaneous"}
        )
        mgr._feature_cols = ["promo_ratio"]
        # Simulate user-provided future features (Polars DataFrame)
        mgr._future_features = pl.DataFrame({
            "unique_id": ["S1"] * 4,
            "ds": [datetime.date(2024, 1, 1) + datetime.timedelta(weeks=i) for i in range(4)],
            "promo_ratio": [0.3, 0.0, 0.5, 0.0],
        })

        eligible = mgr._eligible_predict_cols()
        self.assertEqual(eligible, ["promo_ratio"])


# --------------------------------------------------------------------------- #
#  ML regression guard
# --------------------------------------------------------------------------- #

class TestMLBeatsNaive(unittest.TestCase):

    def test_ml_beats_naive_on_clear_seasonal_data(self):
        """
        On synthetic data with clear trend + seasonality, ML (with enriched
        features) should achieve lower WMAPE than seasonal naive.

        This is a regression guard — if this fails, the ML pipeline is broken.
        """
        from src.forecasting.ml import LGBMDirectForecaster
        from src.forecasting.naive import SeasonalNaiveForecaster

        data = _make_seasonal_series(n_series=3, n_weeks=156)

        # Train/test split: last 13 weeks as holdout
        max_week = data["week"].max()
        cutoff = max_week - timedelta(weeks=13)
        train = data.filter(pl.col("week") <= cutoff)
        test = data.filter(pl.col("week") > cutoff)

        horizon = test.select("week").unique().shape[0]

        # Naive
        naive = SeasonalNaiveForecaster()
        naive.fit(train, target_col="quantity", time_col="week", id_col="series_id")
        naive_preds = naive.predict(horizon=horizon, id_col="series_id", time_col="week")

        # LightGBM
        lgbm = LGBMDirectForecaster(num_threads=1)
        lgbm.fit(train, target_col="quantity", time_col="week", id_col="series_id")
        lgbm_preds = lgbm.predict(horizon=horizon, id_col="series_id", time_col="week")

        def compute_wmape(preds, actuals):
            merged = preds.join(actuals, on=["series_id", "week"], how="inner")
            if merged.is_empty():
                return 1.0
            abs_error = (merged["forecast"] - merged["quantity"]).abs().sum()
            total = merged["quantity"].abs().sum()
            return float(abs_error / total) if total > 0 else 1.0

        naive_wmape = compute_wmape(naive_preds, test)
        lgbm_wmape = compute_wmape(lgbm_preds, test)

        self.assertLess(
            lgbm_wmape, naive_wmape,
            f"LightGBM ({lgbm_wmape:.4f}) should beat naive ({naive_wmape:.4f}) "
            f"on clear seasonal data",
        )


if __name__ == "__main__":
    unittest.main()
