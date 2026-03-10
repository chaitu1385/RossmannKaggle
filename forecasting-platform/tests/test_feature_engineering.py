"""Tests for feature engineering."""

import pandas as pd
import numpy as np
import pytest

from src.data.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Store": [1, 1, 1, 2, 2, 2],
        "Date": pd.date_range("2015-01-01", periods=6, freq="D"),
        "Sales": [5000, 5500, 4800, 6000, 6200, 5900],
    })


def test_temporal_features(sample_df):
    engineer = FeatureEngineer()
    result = engineer.create_temporal_features(sample_df)

    assert "Year" in result.columns
    assert "Month" in result.columns
    assert "DayOfWeek" in result.columns
    assert "IsWeekend" in result.columns
    assert result["Year"].iloc[0] == 2015
    assert result["Month"].iloc[0] == 1


def test_lag_features(sample_df):
    engineer = FeatureEngineer(lag_periods=[1, 2])
    result = engineer.create_lag_features(sample_df, target_col="Sales", group_col="Store")

    assert "Sales_lag_1" in result.columns
    assert "Sales_lag_2" in result.columns
    assert result["Sales_lag_1"].isna().sum() > 0


def test_rolling_features(sample_df):
    engineer = FeatureEngineer(rolling_windows=[2])
    result = engineer.create_rolling_features(sample_df, target_col="Sales", group_col="Store")

    assert "Sales_roll_mean_2" in result.columns
    assert "Sales_roll_std_2" in result.columns
