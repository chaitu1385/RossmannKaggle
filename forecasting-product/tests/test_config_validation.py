"""Tests for config validation, frequency-aware defaults, and TransitionEngine frequency support."""

from datetime import date, timedelta

import polars as pl
import pytest

from src.config.schema import (
    BacktestConfig,
    DataQualityConfig,
    ForecastConfig,
    FREQUENCY_PROFILES,
    PlatformConfig,
    TransitionConfig,
    VALID_RAMP_SHAPES,
)

pytestmark = pytest.mark.unit


# ===========================================================================
# ForecastConfig validation
# ===========================================================================


class TestForecastConfigValidation:
    """__post_init__ validates frequency and horizon."""

    def test_valid_frequencies_accepted(self):
        for freq in ("D", "W", "M", "Q"):
            cfg = ForecastConfig(frequency=freq)
            assert cfg.frequency == freq

    def test_invalid_frequency_rejected(self):
        with pytest.raises(ValueError, match="Unsupported frequency"):
            ForecastConfig(frequency="X")

    def test_zero_horizon_rejected(self):
        with pytest.raises(ValueError, match="horizon_weeks must be >= 1"):
            ForecastConfig(horizon_weeks=0)

    def test_negative_horizon_rejected(self):
        with pytest.raises(ValueError, match="horizon_weeks must be >= 1"):
            ForecastConfig(horizon_weeks=-5)

    def test_valid_horizon_accepted(self):
        cfg = ForecastConfig(horizon_weeks=12)
        assert cfg.horizon_periods == 12


# ===========================================================================
# BacktestConfig validation
# ===========================================================================


class TestBacktestConfigValidation:
    """__post_init__ validates folds, val_weeks, gap_weeks."""

    def test_zero_folds_rejected(self):
        with pytest.raises(ValueError, match="n_folds must be >= 1"):
            BacktestConfig(n_folds=0)

    def test_zero_val_weeks_rejected(self):
        with pytest.raises(ValueError, match="val_weeks must be >= 1"):
            BacktestConfig(val_weeks=0)

    def test_negative_gap_rejected(self):
        with pytest.raises(ValueError, match="gap_weeks must be >= 0"):
            BacktestConfig(gap_weeks=-1)

    def test_zero_gap_accepted(self):
        cfg = BacktestConfig(gap_weeks=0)
        assert cfg.gap_periods == 0

    def test_valid_config_accepted(self):
        cfg = BacktestConfig(n_folds=5, val_weeks=4, gap_weeks=1)
        assert cfg.val_periods == 4


# ===========================================================================
# TransitionConfig validation
# ===========================================================================


class TestTransitionConfigValidation:
    """__post_init__ validates ramp_shape and window."""

    def test_invalid_ramp_shape_rejected(self):
        with pytest.raises(ValueError, match="Unknown ramp_shape"):
            TransitionConfig(ramp_shape="banana")

    def test_valid_ramp_shapes_accepted(self):
        for shape in VALID_RAMP_SHAPES:
            cfg = TransitionConfig(ramp_shape=shape)
            assert cfg.ramp_shape == shape

    def test_zero_window_rejected(self):
        with pytest.raises(ValueError, match="transition_window_weeks must be >= 1"):
            TransitionConfig(transition_window_weeks=0)

    def test_transition_window_periods_alias(self):
        cfg = TransitionConfig(transition_window_weeks=8)
        assert cfg.transition_window_periods == 8


# ===========================================================================
# DataQualityConfig.effective_min_series_length
# ===========================================================================


class TestEffectiveMinSeriesLength:
    """Frequency-aware minimum series length."""

    def test_default_52_returns_profile_value(self):
        """When user hasn't overridden the default, use the profile value."""
        dq = DataQualityConfig()
        assert dq.min_series_length_weeks == 52  # default

        for freq in ("D", "W", "M", "Q"):
            expected = FREQUENCY_PROFILES[freq]["min_series_length"]
            assert dq.effective_min_series_length(freq) == expected

    def test_weekly_default_unchanged(self):
        dq = DataQualityConfig()
        assert dq.effective_min_series_length("W") == 52

    def test_daily_default_is_90(self):
        dq = DataQualityConfig()
        assert dq.effective_min_series_length("D") == 90

    def test_monthly_default_is_24(self):
        dq = DataQualityConfig()
        assert dq.effective_min_series_length("M") == 24

    def test_quarterly_default_is_8(self):
        dq = DataQualityConfig()
        assert dq.effective_min_series_length("Q") == 8

    def test_explicit_override_respected(self):
        """When user explicitly sets a value, use it regardless of frequency."""
        dq = DataQualityConfig(min_series_length_weeks=100)
        assert dq.effective_min_series_length("D") == 100
        assert dq.effective_min_series_length("W") == 100
        assert dq.effective_min_series_length("M") == 100


# ===========================================================================
# TransitionEngine frequency-awareness
# ===========================================================================


class TestTransitionEngineFrequency:
    """TransitionEngine uses freq_timedelta instead of timedelta(weeks=...)."""

    def _make_mapping(self, old_sku="OLD", new_sku="NEW", proportion=1.0):
        return pl.DataFrame({
            "old_sku": [old_sku],
            "new_sku": [new_sku],
            "proportion": [proportion],
        })

    def _make_product_master(self, sku_id, launch_date):
        return pl.DataFrame({
            "sku_id": [sku_id],
            "launch_date": [launch_date],
        }).with_columns(pl.col("launch_date").cast(pl.Date))

    def test_weekly_transition_window_uses_weeks(self):
        """Weekly frequency: window_periods=4 → ramp spans ~4 weeks."""
        from src.series.transition import TransitionEngine, TransitionScenario

        config = TransitionConfig(transition_window_weeks=4)
        engine = TransitionEngine(config, frequency="W")

        origin = date(2024, 1, 1)
        launch = date(2024, 2, 1)  # within horizon

        plans = engine.compute_plans(
            self._make_mapping(),
            self._make_product_master("NEW", launch),
            origin,
            horizon_weeks=13,
        )

        assert len(plans) == 1
        plan = plans[0]
        assert plan.scenario == TransitionScenario.B_IN_HORIZON
        # 4 weeks / 2 = 2 weeks = 14 days
        assert plan.ramp_start == launch - timedelta(weeks=2)
        assert plan.ramp_end == launch + timedelta(weeks=2)

    def test_daily_transition_window_uses_days(self):
        """Daily frequency: window_periods=10 → ramp spans ~10 days."""
        from src.series.transition import TransitionEngine, TransitionScenario

        config = TransitionConfig(transition_window_weeks=10)
        engine = TransitionEngine(config, frequency="D")

        origin = date(2024, 1, 1)
        launch = date(2024, 1, 15)

        plans = engine.compute_plans(
            self._make_mapping(),
            self._make_product_master("NEW", launch),
            origin,
            horizon_weeks=30,
        )

        assert len(plans) == 1
        plan = plans[0]
        assert plan.scenario == TransitionScenario.B_IN_HORIZON
        # 10 days / 2 = 5 days
        assert plan.ramp_start == launch - timedelta(days=5)
        assert plan.ramp_end == launch + timedelta(days=5)

    def test_daily_horizon_uses_days(self):
        """Daily frequency: horizon_weeks=30 → 30 days, not 30 weeks."""
        from src.series.transition import TransitionEngine, TransitionScenario

        config = TransitionConfig(transition_window_weeks=4)
        engine = TransitionEngine(config, frequency="D")

        origin = date(2024, 1, 1)
        # Launch at day 20 — within 30-day horizon
        launch_in = date(2024, 1, 21)
        # Launch at day 40 — beyond 30-day horizon
        launch_out = date(2024, 2, 10)

        plans_in = engine.compute_plans(
            self._make_mapping(),
            self._make_product_master("NEW", launch_in),
            origin,
            horizon_weeks=30,
        )
        assert plans_in[0].scenario == TransitionScenario.B_IN_HORIZON

        plans_out = engine.compute_plans(
            self._make_mapping(),
            self._make_product_master("NEW", launch_out),
            origin,
            horizon_weeks=30,
        )
        assert plans_out[0].scenario == TransitionScenario.C_BEYOND_HORIZON

    def test_default_frequency_is_weekly(self):
        """Backward compat: TransitionEngine without explicit frequency defaults to 'W'."""
        from src.series.transition import TransitionEngine

        config = TransitionConfig(transition_window_weeks=13)
        engine = TransitionEngine(config)
        assert engine.frequency == "W"
