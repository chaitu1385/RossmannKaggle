"""
Tests for the CausalAnalyzer — price elasticity, cannibalization, promo lift.
"""

import random
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

pytestmark = pytest.mark.unit


# ═══════════════════════════════════════════════════════════════════════════════
# Test data generators
# ═══════════════════════════════════════════════════════════════════════════════

def _make_demand_with_price(
    n_series: int = 3,
    n_weeks: int = 104,
    start_date: date = date(2022, 1, 3),
    seed: int = 42,
) -> pl.DataFrame:
    """Synthetic demand with price and promo columns."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    rows = []
    for i in range(n_series):
        sid = f"SKU-{i:03d}"
        base_demand = 100 + i * 50
        base_price = 10.0 + i * 2.0
        elasticity = -0.5 - i * 0.3  # increasingly elastic

        for w in range(n_weeks):
            week_date = start_date + timedelta(weeks=w)

            # Price variation (±20%)
            price_factor = 1.0 + np_rng.uniform(-0.2, 0.2)
            price = base_price * price_factor

            # Promo ~20% of weeks
            is_promo = rng.random() < 0.2
            promo_flag = 1.0 if is_promo else 0.0
            promo_lift_factor = 1.4 if is_promo else 1.0

            # Demand with price elasticity + promo lift + noise
            log_demand = (
                np.log(base_demand)
                + elasticity * np.log(price / base_price)
                + np.log(promo_lift_factor)
                + np_rng.normal(0, 0.1)
            )
            quantity = max(1.0, np.exp(log_demand))

            # Seasonal pattern
            seasonal = 1.3 if week_date.month >= 10 else 1.0
            quantity *= seasonal

            rows.append({
                "series_id": sid,
                "week": week_date,
                "quantity": round(quantity, 2),
                "unit_price": round(price, 2),
                "promo_flag": promo_flag,
                "category": f"Cat-{i % 2}",
            })

    return pl.DataFrame(rows)


def _make_cannibalizing_pair(
    n_weeks: int = 104,
    start_date: date = date(2022, 1, 3),
    seed: int = 42,
) -> pl.DataFrame:
    """Two SKUs where one cannibalizes the other."""
    np_rng = np.random.RandomState(seed)
    rows = []

    for w in range(n_weeks):
        week_date = start_date + timedelta(weeks=w)
        # Shared driver: when product A gains, product B loses
        shock = np_rng.normal(0, 10)

        rows.append({
            "series_id": "PROD-A",
            "week": week_date,
            "quantity": round(100 + shock + np_rng.normal(0, 3), 2),
            "category": "widgets",
        })
        rows.append({
            "series_id": "PROD-B",
            "week": week_date,
            "quantity": round(100 - shock + np_rng.normal(0, 3), 2),
            "category": "widgets",
        })

    return pl.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# CausalAnalyzer import
# ═══════════════════════════════════════════════════════════════════════════════

from src.analytics.causal import CausalAnalyzer, _normal_cdf, _welch_t_test


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Price Elasticity
# ═══════════════════════════════════════════════════════════════════════════════

class TestPriceElasticity:
    """Tests for price elasticity estimation."""

    def test_basic_elasticity_estimation(self):
        df = _make_demand_with_price()
        ca = CausalAnalyzer()
        result = ca.estimate_price_elasticity(df, price_col="unit_price")

        assert isinstance(result, pl.DataFrame)
        assert result.height == 3  # 3 series
        assert "elasticity" in result.columns
        assert "p_value" in result.columns
        assert "is_significant" in result.columns

    def test_elasticity_values_are_negative(self):
        """Normal goods should have negative price elasticity."""
        df = _make_demand_with_price(n_weeks=200)
        ca = CausalAnalyzer()
        result = ca.estimate_price_elasticity(df, price_col="unit_price")

        for row in result.iter_rows(named=True):
            if row["elasticity"] is not None and row["is_significant"]:
                assert row["elasticity"] < 0, (
                    f"Expected negative elasticity for {row['series_id']}, "
                    f"got {row['elasticity']}"
                )

    def test_elasticity_with_controls(self):
        df = _make_demand_with_price()
        ca = CausalAnalyzer()
        result = ca.estimate_price_elasticity(
            df, price_col="unit_price", control_cols=["promo_flag"]
        )
        assert result.height == 3

    def test_elasticity_r_squared(self):
        df = _make_demand_with_price()
        ca = CausalAnalyzer()
        result = ca.estimate_price_elasticity(df, price_col="unit_price")

        for row in result.iter_rows(named=True):
            if row["r_squared"] is not None:
                assert 0 <= row["r_squared"] <= 1.0

    def test_elasticity_insufficient_data(self):
        df = _make_demand_with_price(n_weeks=5)
        ca = CausalAnalyzer(min_observations=20)
        result = ca.estimate_price_elasticity(df, price_col="unit_price")

        for row in result.iter_rows(named=True):
            assert row["interpretation"] == "Insufficient data"
            assert row["is_significant"] is False

    def test_elasticity_missing_price_column(self):
        df = _make_demand_with_price()
        ca = CausalAnalyzer()
        with pytest.raises(ValueError, match="Price column"):
            ca.estimate_price_elasticity(df, price_col="nonexistent")

    def test_elasticity_interpretation_categories(self):
        df = _make_demand_with_price(n_weeks=200)
        ca = CausalAnalyzer()
        result = ca.estimate_price_elasticity(df, price_col="unit_price")

        valid_interps = {
            "Not statistically significant",
            "Elastic — demand highly sensitive to price",
            "Inelastic — demand moderately sensitive to price",
            "Perfectly inelastic — price has no effect",
            "Positive elasticity — Giffen/Veblen good or data issue",
            "Insufficient data",
            "Singular matrix — insufficient price variation",
        }
        for row in result.iter_rows(named=True):
            assert row["interpretation"] in valid_interps

    def test_elasticity_n_obs_correct(self):
        n_weeks = 80
        df = _make_demand_with_price(n_series=1, n_weeks=n_weeks)
        ca = CausalAnalyzer()
        result = ca.estimate_price_elasticity(df, price_col="unit_price")
        # n_obs should be <= n_weeks (some may be filtered for zero price/qty)
        assert result["n_obs"][0] <= n_weeks


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Cannibalization Detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestCannibalization:
    """Tests for cannibalization detection."""

    def test_detect_cannibalizing_pair(self):
        df = _make_cannibalizing_pair()
        ca = CausalAnalyzer()
        result = ca.detect_cannibalization(df)

        assert result.height >= 1
        cannib = result.filter(pl.col("is_cannibalizing"))
        assert cannib.height >= 1, "Should detect cannibalization"
        assert cannib["residual_correlation"][0] < -0.3

    def test_cannibalization_strength_labels(self):
        df = _make_cannibalizing_pair()
        ca = CausalAnalyzer()
        result = ca.detect_cannibalization(df)

        valid_strengths = {"strong", "moderate", "weak", "none"}
        for row in result.iter_rows(named=True):
            assert row["strength"] in valid_strengths

    def test_no_cannibalization_independent_series(self):
        """Independent series should not be flagged as cannibalizing."""
        df = _make_demand_with_price(n_series=2)
        ca = CausalAnalyzer()
        result = ca.detect_cannibalization(df)

        # May or may not find spurious correlation, but should not be "strong"
        strong = result.filter(pl.col("strength") == "strong")
        assert strong.height == 0, "Independent series should not show strong cannibalization"

    def test_cannibalization_with_group(self):
        df = _make_demand_with_price(n_series=4)
        ca = CausalAnalyzer()
        result = ca.detect_cannibalization(df, group_col="category")
        assert "group" in result.columns

    def test_cannibalization_single_series(self):
        """Single series should return empty."""
        df = _make_demand_with_price(n_series=1)
        ca = CausalAnalyzer()
        result = ca.detect_cannibalization(df)
        assert result.height == 0

    def test_cannibalization_sorted_by_correlation(self):
        df = _make_demand_with_price(n_series=4)
        ca = CausalAnalyzer()
        result = ca.detect_cannibalization(df)
        if result.height > 1:
            corrs = result["residual_correlation"].to_list()
            assert corrs == sorted(corrs), "Results should be sorted by correlation"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Promotional Lift
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromoLift:
    """Tests for promotional lift estimation."""

    def test_basic_promo_lift(self):
        df = _make_demand_with_price()
        ca = CausalAnalyzer()
        result = ca.estimate_promo_lift(df, promo_col="promo_flag")

        assert result.height == 3
        assert "lift_ratio" in result.columns
        assert "lift_pct" in result.columns
        assert "is_significant" in result.columns

    def test_promo_lift_positive(self):
        """Promoted periods should have higher demand (by construction)."""
        df = _make_demand_with_price(n_weeks=200)
        ca = CausalAnalyzer()
        result = ca.estimate_promo_lift(df, promo_col="promo_flag")

        for row in result.iter_rows(named=True):
            if row["lift_ratio"] is not None:
                assert row["lift_ratio"] > 1.0, (
                    f"Expected positive lift for {row['series_id']}"
                )

    def test_promo_lift_with_price_adjustment(self):
        df = _make_demand_with_price()
        ca = CausalAnalyzer()
        result = ca.estimate_promo_lift(
            df, promo_col="promo_flag", price_col="unit_price"
        )
        assert "price_adjusted_lift" in result.columns

    def test_promo_lift_missing_column(self):
        df = _make_demand_with_price()
        ca = CausalAnalyzer()
        with pytest.raises(ValueError, match="Promo column"):
            ca.estimate_promo_lift(df, promo_col="nonexistent")

    def test_promo_lift_insufficient_promos(self):
        """Series with <3 promo weeks should get None lift."""
        # Create data with very few promos
        df = _make_demand_with_price(n_weeks=20)
        # Override promo_flag to have almost no promos
        df = df.with_columns(pl.lit(0.0).alias("promo_flag"))
        ca = CausalAnalyzer()
        result = ca.estimate_promo_lift(df, promo_col="promo_flag")

        for row in result.iter_rows(named=True):
            assert row["lift_ratio"] is None

    def test_promo_lift_incremental_volume(self):
        df = _make_demand_with_price()
        ca = CausalAnalyzer()
        result = ca.estimate_promo_lift(df, promo_col="promo_flag")

        for row in result.iter_rows(named=True):
            if row["incremental_volume_per_week"] is not None:
                # Should be approximately: promo_demand - baseline_demand
                expected = row["promo_demand"] - row["baseline_demand"]
                assert abs(row["incremental_volume_per_week"] - expected) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Full Report
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullCausalReport:

    def test_full_report_all_columns(self):
        df = _make_demand_with_price()
        ca = CausalAnalyzer()
        report = ca.full_causal_report(
            df, price_col="unit_price", promo_col="promo_flag",
            group_col="category",
        )
        assert "elasticity" in report
        assert "cannibalization" in report
        assert "promo_lift" in report

    def test_full_report_no_price(self):
        df = _make_demand_with_price().drop("unit_price")
        ca = CausalAnalyzer()
        report = ca.full_causal_report(df, promo_col="promo_flag")
        assert "elasticity" not in report
        assert "cannibalization" in report
        assert "promo_lift" in report

    def test_full_report_no_promo(self):
        df = _make_demand_with_price().drop("promo_flag")
        ca = CausalAnalyzer()
        report = ca.full_causal_report(df, price_col="unit_price")
        assert "elasticity" in report
        assert "cannibalization" in report
        assert "promo_lift" not in report


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Statistical helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestStatisticalHelpers:

    def test_normal_cdf_values(self):
        assert abs(_normal_cdf(0) - 0.5) < 1e-4
        assert abs(_normal_cdf(1.96) - 0.975) < 1e-3
        assert abs(_normal_cdf(-1.96) - 0.025) < 1e-3

    def test_welch_t_test_same_distribution(self):
        rng = np.random.RandomState(42)
        a = rng.normal(100, 10, size=100)
        b = rng.normal(100, 10, size=100)
        t_stat, p_value = _welch_t_test(a, b)
        assert p_value > 0.05, "Same distribution should not be significant"

    def test_welch_t_test_different_means(self):
        rng = np.random.RandomState(42)
        a = rng.normal(100, 10, size=100)
        b = rng.normal(120, 10, size=100)
        t_stat, p_value = _welch_t_test(a, b)
        assert p_value < 0.05, "Different means should be significant"

    def test_welch_t_test_direction(self):
        a = np.array([110.0, 120.0, 130.0, 115.0, 125.0])
        b = np.array([90.0, 80.0, 85.0, 95.0, 88.0])
        t_stat, _ = _welch_t_test(a, b)
        assert t_stat > 0, "a > b should give positive t-statistic"
