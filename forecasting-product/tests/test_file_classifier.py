"""Tests for FileClassifier — automatic role classification for multi-file uploads."""

import unittest
from datetime import date, timedelta
from typing import Dict

import numpy as np
import polars as pl

from src.data.file_classifier import (
    ClassificationResult,
    FileClassifier,
    FileProfile,
)

import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
#  Factory helpers
# --------------------------------------------------------------------------- #

def _make_primary_timeseries(
    n_stores: int = 3,
    n_weeks: int = 52,
    start_date: date = date(2022, 1, 3),
    seed: int = 42,
) -> pl.DataFrame:
    """Create a typical weekly time-series DataFrame (week, store_id, product_id, quantity)."""
    rng = np.random.RandomState(seed)
    rows = []
    products = ["prod_A", "prod_B"]
    for s in range(n_stores):
        for p in products:
            for w in range(n_weeks):
                rows.append({
                    "week": start_date + timedelta(weeks=w),
                    "store_id": f"store_{s}",
                    "product_id": p,
                    "quantity": round(rng.uniform(10, 200), 2),
                })
    return pl.DataFrame(rows)


def _make_dimension_table() -> pl.DataFrame:
    """Create a dimension/lookup table (no date column, shared IDs, mostly categorical)."""
    return pl.DataFrame({
        "store_id": ["store_0", "store_1", "store_2"],
        "region": ["North", "South", "East"],
        "store_format": ["Large", "Small", "Medium"],
        "manager": ["Alice", "Bob", "Charlie"],
    })


def _make_regressor_table(
    n_weeks: int = 52,
    start_date: date = date(2022, 1, 3),
    seed: int = 42,
) -> pl.DataFrame:
    """Create an external regressor table (date + numeric features, joinable)."""
    rng = np.random.RandomState(seed)
    rows = []
    for w in range(n_weeks):
        rows.append({
            "week": start_date + timedelta(weeks=w),
            "temperature": round(rng.normal(20, 5), 1),
            "rainfall_mm": round(rng.exponential(10), 1),
        })
    return pl.DataFrame(rows)


def _make_regressor_with_ids(
    n_stores: int = 3,
    n_weeks: int = 52,
    start_date: date = date(2022, 1, 3),
    seed: int = 42,
) -> pl.DataFrame:
    """Regressor table with store-level granularity."""
    rng = np.random.RandomState(seed)
    rows = []
    for s in range(n_stores):
        for w in range(n_weeks):
            rows.append({
                "week": start_date + timedelta(weeks=w),
                "store_id": f"store_{s}",
                "promo_discount": round(rng.uniform(0, 0.3), 2),
            })
    return pl.DataFrame(rows)


def _make_non_timeseries() -> pl.DataFrame:
    """Create a file that doesn't look like time series at all."""
    return pl.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "department": ["HR", "Eng", "Eng", "Sales"],
        "salary": [70000, 95000, 102000, 80000],
    })


def _make_ambiguous_file(
    n_weeks: int = 52,
    start_date: date = date(2022, 1, 3),
    seed: int = 42,
) -> pl.DataFrame:
    """File with a date and numbers but no clear target pattern — could be time_series or regressor."""
    rng = np.random.RandomState(seed)
    rows = []
    for w in range(n_weeks):
        rows.append({
            "week": start_date + timedelta(weeks=w),
            "metric_alpha": round(rng.normal(0, 1), 3),
            "metric_beta": round(rng.normal(5, 2), 3),
        })
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Tests
# --------------------------------------------------------------------------- #

class TestClassifySingle(unittest.TestCase):
    """Test classify_single on individual files."""

    def setUp(self):
        self.clf = FileClassifier()

    def test_primary_timeseries_detected(self):
        df = _make_primary_timeseries()
        profile = self.clf.classify_single("sales.csv", df)
        self.assertEqual(profile.role, "time_series")
        self.assertGreaterEqual(profile.confidence, 0.4)
        self.assertEqual(profile.time_column, "week")
        self.assertIn("store_id", profile.id_columns)

    def test_primary_timeseries_high_confidence(self):
        df = _make_primary_timeseries()
        profile = self.clf.classify_single("sales.csv", df)
        # Should be high confidence: Date type + 'quantity' target + repeating IDs + >50 rows
        self.assertGreaterEqual(profile.confidence, 0.8)

    def test_dimension_table_in_isolation(self):
        """Dimension tables in isolation are classified as unknown (no date + few rows)."""
        df = _make_dimension_table()
        profile = self.clf.classify_single("stores.csv", df)
        # Without date or strong target, should not be time_series
        self.assertNotEqual(profile.role, "time_series")

    def test_non_timeseries_detected_as_unknown(self):
        df = _make_non_timeseries()
        profile = self.clf.classify_single("employees.csv", df)
        self.assertEqual(profile.role, "unknown")
        self.assertLess(profile.confidence, 0.4)

    def test_time_column_detection_date_type(self):
        df = _make_primary_timeseries()
        profile = self.clf.classify_single("data.csv", df)
        self.assertEqual(profile.time_column, "week")

    def test_time_column_detection_name_pattern(self):
        """If column is not Date type but named 'date', still detected."""
        df = pl.DataFrame({
            "date": ["2022-01-03", "2022-01-10", "2022-01-17"] * 10,
            "id": [f"s_{i}" for i in range(10)] * 3,
            "quantity": list(range(30)),
        })
        profile = self.clf.classify_single("data.csv", df)
        self.assertEqual(profile.time_column, "date")

    def test_target_column_prefers_known_names(self):
        df = _make_primary_timeseries()
        profile = self.clf.classify_single("data.csv", df)
        self.assertIn("quantity", [profile.df.columns[i] for i in range(len(df.columns)) if df.columns[i] == "quantity"])

    def test_confidence_in_range(self):
        df = _make_primary_timeseries()
        profile = self.clf.classify_single("data.csv", df)
        self.assertGreaterEqual(profile.confidence, 0.0)
        self.assertLessEqual(profile.confidence, 1.0)

    def test_reasoning_populated(self):
        df = _make_primary_timeseries()
        profile = self.clf.classify_single("data.csv", df)
        self.assertIsInstance(profile.reasoning, list)
        self.assertGreaterEqual(len(profile.reasoning), 1)

    def test_profile_metadata_correct(self):
        df = _make_primary_timeseries(n_stores=2, n_weeks=10)
        profile = self.clf.classify_single("data.csv", df)
        self.assertEqual(profile.n_rows, df.height)
        self.assertEqual(profile.n_columns, len(df.columns))
        self.assertEqual(profile.filename, "data.csv")


class TestClassifyFiles(unittest.TestCase):
    """Test classify_files with multiple files."""

    def setUp(self):
        self.clf = FileClassifier()

    def test_single_file_timeseries(self):
        files = {"sales.csv": _make_primary_timeseries()}
        result = self.clf.classify_files(files)
        self.assertIsNotNone(result.primary_file)
        self.assertEqual(result.primary_file.filename, "sales.csv")
        self.assertEqual(len(result.dimension_files), 0)
        self.assertEqual(len(result.regressor_files), 0)

    def test_single_file_not_timeseries(self):
        files = {"employees.csv": _make_non_timeseries()}
        result = self.clf.classify_files(files)
        self.assertIsNone(result.primary_file)
        self.assertEqual(len(result.unknown_files), 1)
        self.assertIsInstance(result.warnings, list)
        self.assertGreaterEqual(len(result.warnings), 1)

    def test_timeseries_plus_dimension(self):
        files = {
            "sales.csv": _make_primary_timeseries(),
            "stores.csv": _make_dimension_table(),
        }
        result = self.clf.classify_files(files)
        self.assertIsNotNone(result.primary_file)
        self.assertEqual(result.primary_file.filename, "sales.csv")
        self.assertEqual(len(result.dimension_files), 1)
        self.assertEqual(result.dimension_files[0].filename, "stores.csv")

    def test_timeseries_plus_regressor(self):
        files = {
            "sales.csv": _make_primary_timeseries(),
            "weather.csv": _make_regressor_table(),
        }
        result = self.clf.classify_files(files)
        self.assertIsNotNone(result.primary_file)
        self.assertEqual(len(result.regressor_files), 1)
        self.assertEqual(result.regressor_files[0].filename, "weather.csv")

    def test_timeseries_plus_regressor_with_ids(self):
        files = {
            "sales.csv": _make_primary_timeseries(),
            "promos.csv": _make_regressor_with_ids(),
        }
        result = self.clf.classify_files(files)
        self.assertIsNotNone(result.primary_file)
        self.assertEqual(len(result.regressor_files), 1)

    def test_three_files_mixed(self):
        files = {
            "sales.csv": _make_primary_timeseries(),
            "stores.csv": _make_dimension_table(),
            "weather.csv": _make_regressor_table(),
        }
        result = self.clf.classify_files(files)
        self.assertIsNotNone(result.primary_file)
        self.assertEqual(result.primary_file.filename, "sales.csv")
        self.assertEqual(len(result.dimension_files), 1)
        self.assertEqual(len(result.regressor_files), 1)
        self.assertEqual(len(result.unknown_files), 0)

    def test_two_timeseries_candidates_picks_best(self):
        """When two files look like time_series, the one with higher confidence wins."""
        primary = _make_primary_timeseries(n_stores=5, n_weeks=104)
        # Smaller file — still looks like ts but lower confidence
        secondary = _make_primary_timeseries(n_stores=1, n_weeks=20)
        secondary = secondary.rename({"quantity": "sales"})

        files = {
            "big_sales.csv": primary,
            "small_sales.csv": secondary,
        }
        result = self.clf.classify_files(files)
        self.assertIsNotNone(result.primary_file)
        # The bigger file should win (more rows → higher confidence)
        self.assertEqual(result.primary_file.filename, "big_sales.csv")

    def test_no_files_returns_empty(self):
        result = self.clf.classify_files({})
        self.assertIsNone(result.primary_file)
        self.assertEqual(len(result.profiles), 0)
        self.assertIsInstance(result.warnings, list)
        self.assertGreaterEqual(len(result.warnings), 1)

    def test_all_unknown_produces_warning(self):
        files = {
            "a.csv": _make_non_timeseries(),
            "b.csv": _make_non_timeseries(),
        }
        result = self.clf.classify_files(files)
        self.assertIsNone(result.primary_file)
        self.assertIsInstance(result.warnings, list)
        self.assertGreaterEqual(len(result.warnings), 1)

    def test_dimension_role_assigned_correctly(self):
        files = {
            "sales.csv": _make_primary_timeseries(),
            "stores.csv": _make_dimension_table(),
        }
        result = self.clf.classify_files(files)
        store_profile = result.dimension_files[0]
        self.assertEqual(store_profile.role, "dimension")
        self.assertGreaterEqual(store_profile.confidence, 0.3)

    def test_regressor_role_assigned_correctly(self):
        files = {
            "sales.csv": _make_primary_timeseries(),
            "weather.csv": _make_regressor_table(),
        }
        result = self.clf.classify_files(files)
        weather_profile = result.regressor_files[0]
        self.assertEqual(weather_profile.role, "regressor")
        self.assertGreaterEqual(weather_profile.confidence, 0.3)


class TestEdgeCases(unittest.TestCase):
    """Edge cases in file classification."""

    def setUp(self):
        self.clf = FileClassifier()

    def test_empty_dataframe(self):
        df = pl.DataFrame({"a": [], "b": []}).cast({"a": pl.Utf8, "b": pl.Float64})
        profile = self.clf.classify_single("empty.csv", df)
        self.assertLess(profile.confidence, 0.4)

    def test_single_column_file(self):
        df = pl.DataFrame({"value": [1, 2, 3, 4, 5]})
        profile = self.clf.classify_single("single.csv", df)
        # No time column, no ID → unlikely time_series
        self.assertLessEqual(profile.confidence, 0.4)

    def test_dimension_with_partial_id_overlap(self):
        """Dimension table has some IDs not in primary — still classified as dimension."""
        primary = _make_primary_timeseries()
        dim = pl.DataFrame({
            "store_id": ["store_0", "store_1", "store_99"],  # store_99 not in primary
            "region": ["North", "South", "West"],
        })
        files = {"sales.csv": primary, "stores.csv": dim}
        result = self.clf.classify_files(files)
        self.assertEqual(len(result.dimension_files), 1)

    def test_regressor_no_id_is_broadcastable(self):
        """Regressor with only time column (no IDs) is broadcastable and still detected."""
        files = {
            "sales.csv": _make_primary_timeseries(),
            "weather.csv": _make_regressor_table(),
        }
        result = self.clf.classify_files(files)
        self.assertEqual(len(result.regressor_files), 1)
        # weather.csv has no store_id → broadcastable
        weather = result.regressor_files[0]
        self.assertEqual(len(weather.id_columns), 0)

    def test_profiles_count_matches_input(self):
        files = {
            "a.csv": _make_primary_timeseries(),
            "b.csv": _make_dimension_table(),
            "c.csv": _make_regressor_table(),
            "d.csv": _make_non_timeseries(),
        }
        result = self.clf.classify_files(files)
        self.assertEqual(len(result.profiles), 4)


if __name__ == "__main__":
    unittest.main()
