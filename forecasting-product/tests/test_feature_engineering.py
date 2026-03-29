"""Tests for feature engineering."""

import polars as pl
import pytest

from src.data.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return pl.DataFrame({
        "Store": [1, 1, 1, 1, 2, 2, 2, 2],
        "Date": [
            "2015-01-01", "2015-01-02", "2015-01-03", "2015-01-04",
            "2015-01-01", "2015-01-02", "2015-01-03", "2015-01-04",
        ],
        "Sales": [5000, 5500, 4800, 6100, 6000, 6200, 5900, 6400],
    }).with_columns(pl.col("Date").str.to_date())


# ---------------------------------------------------------------------------
# create_temporal_features
# ---------------------------------------------------------------------------

class TestCreateTemporalFeatures:
    def test_required_columns_present(self, sample_df):
        engineer = FeatureEngineer()
        result = engineer.create_temporal_features(sample_df)

        expected_cols = {
            "Year", "Month", "Day", "DayOfWeek",
            "WeekOfYear", "Quarter", "IsWeekend",
            "IsMonthStart", "IsMonthEnd",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_year_month_values(self, sample_df):
        result = FeatureEngineer().create_temporal_features(sample_df)
        assert result["Year"].to_list() == [2015] * 8
        assert result["Month"].to_list() == [1] * 8

    def test_day_of_week_thursday(self, sample_df):
        # 2015-01-01 is a Thursday — Polars weekday: Mon=1 … Sun=7 → Thu=4
        result = FeatureEngineer().create_temporal_features(sample_df)
        assert result["DayOfWeek"][0] == 4

    def test_is_weekend_false_for_weekday(self, sample_df):
        result = FeatureEngineer().create_temporal_features(sample_df)
        # 2015-01-01 (Thursday) must not be marked as weekend
        assert result["IsWeekend"][0] == 0

    def test_is_weekend_true_for_saturday(self):
        df = pl.DataFrame({"Date": ["2015-01-03"]}).with_columns(
            pl.col("Date").str.to_date()
        )
        result = FeatureEngineer().create_temporal_features(df)
        assert result["IsWeekend"][0] == 1

    def test_is_month_start(self, sample_df):
        result = FeatureEngineer().create_temporal_features(sample_df)
        # First row is 2015-01-01 → month start
        assert result["IsMonthStart"][0] == 1
        # Second row is 2015-01-02 → not month start
        assert result["IsMonthStart"][1] == 0

    def test_is_month_end(self):
        df = pl.DataFrame({"Date": ["2015-01-31", "2015-01-30"]}).with_columns(
            pl.col("Date").str.to_date()
        )
        result = FeatureEngineer().create_temporal_features(df)
        assert result["IsMonthEnd"][0] == 1
        assert result["IsMonthEnd"][1] == 0

    def test_quarter_january(self, sample_df):
        result = FeatureEngineer().create_temporal_features(sample_df)
        assert result["Quarter"].to_list() == [1] * 8

    def test_no_extra_temp_columns_leaked(self, sample_df):
        result = FeatureEngineer().create_temporal_features(sample_df)
        # The internal _month_end helper column must be dropped
        assert "_month_end" not in result.columns

    def test_original_rows_preserved(self, sample_df):
        result = FeatureEngineer().create_temporal_features(sample_df)
        assert len(result) == len(sample_df)


# ---------------------------------------------------------------------------
# create_lag_features
# ---------------------------------------------------------------------------

class TestCreateLagFeatures:
    def test_lag_columns_created(self, sample_df):
        engineer = FeatureEngineer(lag_periods=[1, 2])
        result = engineer.create_lag_features(sample_df, target_col="Sales", group_col="Store")
        assert "Sales_lag_1" in result.columns
        assert "Sales_lag_2" in result.columns

    def test_lag_1_values_correct(self, sample_df):
        engineer = FeatureEngineer(lag_periods=[1])
        result = engineer.create_lag_features(sample_df, target_col="Sales", group_col="Store")
        # Sort to get deterministic order: Store 1 rows
        store1 = result.filter(pl.col("Store") == 1).sort("Date")
        # First row lag should be null
        assert store1["Sales_lag_1"][0] is None
        # Second row lag should be the first row's Sales
        assert store1["Sales_lag_1"][1] == store1["Sales"][0]

    def test_lag_respects_group_boundary(self, sample_df):
        engineer = FeatureEngineer(lag_periods=[1])
        result = engineer.create_lag_features(sample_df, target_col="Sales", group_col="Store")
        # First row of Store 2 should have a null lag, not a value from Store 1
        store2_first = result.filter(pl.col("Store") == 2).sort("Date")["Sales_lag_1"][0]
        assert store2_first is None

    def test_null_introduced_for_first_row_per_group(self, sample_df):
        engineer = FeatureEngineer(lag_periods=[1])
        result = engineer.create_lag_features(sample_df, target_col="Sales", group_col="Store")
        null_count = result["Sales_lag_1"].null_count()
        # One null per group (2 groups)
        assert null_count == 2

    def test_default_lag_periods(self, sample_df):
        engineer = FeatureEngineer()
        result = engineer.create_lag_features(sample_df)
        for lag in [1, 7, 14, 30]:
            assert f"Sales_lag_{lag}" in result.columns

    def test_row_count_unchanged(self, sample_df):
        engineer = FeatureEngineer(lag_periods=[1])
        result = engineer.create_lag_features(sample_df)
        assert len(result) == len(sample_df)


# ---------------------------------------------------------------------------
# create_rolling_features
# ---------------------------------------------------------------------------

class TestCreateRollingFeatures:
    def test_rolling_columns_created(self, sample_df):
        engineer = FeatureEngineer(rolling_windows=[2])
        result = engineer.create_rolling_features(sample_df, target_col="Sales", group_col="Store")
        assert "Sales_roll_mean_2" in result.columns
        assert "Sales_roll_std_2" in result.columns

    def test_multiple_windows(self, sample_df):
        engineer = FeatureEngineer(rolling_windows=[2, 3])
        result = engineer.create_rolling_features(sample_df, target_col="Sales", group_col="Store")
        for window in [2, 3]:
            assert f"Sales_roll_mean_{window}" in result.columns
            assert f"Sales_roll_std_{window}" in result.columns

    def test_rolling_mean_first_row_per_group(self, sample_df):
        # Rolling uses shift(1) so the first row has no prior values;
        # with min_periods=1, Polars still returns a value (the single available point)
        engineer = FeatureEngineer(rolling_windows=[2])
        result = engineer.create_rolling_features(sample_df, target_col="Sales", group_col="Store")
        store1 = result.filter(pl.col("Store") == 1).sort("Date")
        # First row: shift(1) → null, rolling_mean with min_periods=1 still returns null
        # because there are no non-null values in the window
        assert store1["Sales_roll_mean_2"][0] is None

    def test_rolling_mean_second_row_equals_first_sales(self, sample_df):
        engineer = FeatureEngineer(rolling_windows=[2])
        result = engineer.create_rolling_features(sample_df, target_col="Sales", group_col="Store")
        store1 = result.filter(pl.col("Store") == 1).sort("Date")
        # Row index 1: shift(1) gives row 0's Sales → mean of [Sales[0]] = Sales[0]
        assert store1["Sales_roll_mean_2"][1] == pytest.approx(store1["Sales"][0])

    def test_default_rolling_windows(self, sample_df):
        engineer = FeatureEngineer()
        result = engineer.create_rolling_features(sample_df)
        for window in [7, 14, 30]:
            assert f"Sales_roll_mean_{window}" in result.columns

    def test_row_count_unchanged(self, sample_df):
        engineer = FeatureEngineer(rolling_windows=[2])
        result = engineer.create_rolling_features(sample_df)
        assert len(result) == len(sample_df)


# ---------------------------------------------------------------------------
# create_competition_features
# ---------------------------------------------------------------------------

class TestCreateCompetitionFeatures:
    def test_competition_open_created(self):
        df = pl.DataFrame({
            "Year": [2015, 2015],
            "Month": [6, 7],
            "CompetitionOpenSinceYear": [2015, 2015],
            "CompetitionOpenSinceMonth": [1, 1],
        })
        result = FeatureEngineer().create_competition_features(df)
        assert "CompetitionOpen" in result.columns

    def test_competition_open_zero_when_before_open(self):
        df = pl.DataFrame({
            "Year": [2014],
            "Month": [12],
            "CompetitionOpenSinceYear": [2015],
            "CompetitionOpenSinceMonth": [1],
        })
        result = FeatureEngineer().create_competition_features(df)
        assert result["CompetitionOpen"][0] == 0

    def test_competition_open_months_calculated(self):
        df = pl.DataFrame({
            "Year": [2015],
            "Month": [6],
            "CompetitionOpenSinceYear": [2015],
            "CompetitionOpenSinceMonth": [1],
        })
        result = FeatureEngineer().create_competition_features(df)
        # 12*(2015-2015) + (6-1) = 5 months
        assert result["CompetitionOpen"][0] == 5

    def test_no_op_when_column_missing(self, sample_df):
        # sample_df has no CompetitionOpenSinceMonth column — should be a no-op
        result = FeatureEngineer().create_competition_features(sample_df)
        assert "CompetitionOpen" not in result.columns
        assert len(result) == len(sample_df)


# ---------------------------------------------------------------------------
# create_promo_features
# ---------------------------------------------------------------------------

class TestCreatePromoFeatures:
    def test_promo2open_created(self):
        df = pl.DataFrame({
            "Year": [2015],
            "WeekOfYear": [26],
            "Promo2SinceWeek": [1],
            "Promo2SinceYear": [2015],
        })
        result = FeatureEngineer().create_promo_features(df)
        assert "Promo2Open" in result.columns

    def test_promo2open_zero_when_before_promo(self):
        df = pl.DataFrame({
            "Year": [2014],
            "WeekOfYear": [50],
            "Promo2SinceWeek": [1],
            "Promo2SinceYear": [2015],
        })
        result = FeatureEngineer().create_promo_features(df)
        assert result["Promo2Open"][0] == 0.0

    def test_promo2open_positive_after_start(self):
        df = pl.DataFrame({
            "Year": [2015],
            "WeekOfYear": [13],
            "Promo2SinceWeek": [1],
            "Promo2SinceYear": [2015],
        })
        result = FeatureEngineer().create_promo_features(df)
        assert result["Promo2Open"][0] > 0

    def test_no_op_when_column_missing(self, sample_df):
        result = FeatureEngineer().create_promo_features(sample_df)
        assert "Promo2Open" not in result.columns
        assert len(result) == len(sample_df)


# ---------------------------------------------------------------------------
# fit_transform (integration)
# ---------------------------------------------------------------------------

class TestFitTransform:
    def test_fit_transform_adds_temporal_and_lag_and_rolling(self, sample_df):
        engineer = FeatureEngineer(lag_periods=[1], rolling_windows=[2])
        result = engineer.fit_transform(sample_df)

        assert "Year" in result.columns
        assert "Sales_lag_1" in result.columns
        assert "Sales_roll_mean_2" in result.columns

    def test_fit_transform_row_count_preserved(self, sample_df):
        engineer = FeatureEngineer(lag_periods=[1], rolling_windows=[2])
        result = engineer.fit_transform(sample_df)
        assert len(result) == len(sample_df)

    def test_fit_transform_no_extra_internal_columns(self, sample_df):
        engineer = FeatureEngineer(lag_periods=[1], rolling_windows=[2])
        result = engineer.fit_transform(sample_df)
        assert "_month_end" not in result.columns
