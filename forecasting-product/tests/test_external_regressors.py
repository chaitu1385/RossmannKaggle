"""Tests for external regressor integration."""

from datetime import date, timedelta

import polars as pl
import pytest

from src.config.schema import ExternalRegressorConfig, ForecastConfig, PlatformConfig
from src.data.regressors import validate_regressors
from src.series.builder import SeriesBuilder


def _make_actuals(n_weeks: int = 52, n_series: int = 2) -> pl.DataFrame:
    """Generate synthetic actuals."""
    rows = []
    base = date(2024, 1, 1)
    for s in range(n_series):
        for w in range(n_weeks):
            rows.append({
                "series_id": f"S{s:03d}",
                "week": base + timedelta(weeks=w),
                "quantity": float(100 + s * 10 + (w % 13) * 5),
            })
    return pl.DataFrame(rows)


def _make_external_features(
    n_weeks: int = 52, n_series: int = 2, include_future: int = 0
) -> pl.DataFrame:
    """Generate synthetic external features."""
    rows = []
    base = date(2024, 1, 1)
    for s in range(n_series):
        for w in range(n_weeks + include_future):
            rows.append({
                "series_id": f"S{s:03d}",
                "week": base + timedelta(weeks=w),
                "promotion_flag": 1 if w % 8 == 0 else 0,
                "price_index": 1.0 - (0.1 if w % 8 == 0 else 0.0),
            })
    return pl.DataFrame(rows)


class TestValidateRegressors:
    def test_valid_features(self):
        actuals = _make_actuals()
        features = _make_external_features()
        issues = validate_regressors(
            features, actuals,
            feature_columns=["promotion_flag", "price_index"],
            time_column="week",
            id_column="series_id",
        )
        assert len(issues) == 0

    def test_missing_columns(self):
        actuals = _make_actuals()
        features = _make_external_features()
        issues = validate_regressors(
            features, actuals,
            feature_columns=["promotion_flag", "nonexistent_col"],
        )
        assert len(issues) > 0
        assert "nonexistent_col" in issues[0]

    def test_future_coverage_warning(self):
        actuals = _make_actuals(n_weeks=52)
        features = _make_external_features(n_weeks=52, include_future=0)
        issues = validate_regressors(
            features, actuals,
            feature_columns=["promotion_flag"],
            time_column="week",
            horizon_weeks=13,
        )
        assert any("horizon" in i.lower() or "future" in i.lower() for i in issues)


class TestSeriesBuilderWithFeatures:
    def test_build_with_external_features(self):
        config = PlatformConfig(
            forecast=ForecastConfig(
                external_regressors=ExternalRegressorConfig(
                    enabled=True,
                    feature_columns=["promotion_flag", "price_index"],
                ),
            ),
        )
        builder = SeriesBuilder(config)
        actuals = _make_actuals()
        features = _make_external_features()

        result = builder.build(actuals, external_features=features)

        assert "promotion_flag" in result.columns
        assert "price_index" in result.columns
        assert result["promotion_flag"].null_count() == 0

    def test_build_without_external_features(self):
        """Backward compatibility — no features = same as before."""
        config = PlatformConfig()
        builder = SeriesBuilder(config)
        actuals = _make_actuals()

        result = builder.build(actuals)

        assert "promotion_flag" not in result.columns

    def test_broadcast_features_no_series_id(self):
        """Features without series_id are broadcast to all series."""
        config = PlatformConfig(
            forecast=ForecastConfig(
                external_regressors=ExternalRegressorConfig(
                    enabled=True,
                    feature_columns=["promotion_flag"],
                ),
            ),
        )
        builder = SeriesBuilder(config)
        actuals = _make_actuals(n_series=3)

        # Features without series_id column (global features like holidays)
        base = date(2024, 1, 1)
        features = pl.DataFrame({
            "week": [base + timedelta(weeks=w) for w in range(52)],
            "promotion_flag": [1 if w % 8 == 0 else 0 for w in range(52)],
        })

        result = builder.build(actuals, external_features=features)
        assert "promotion_flag" in result.columns
