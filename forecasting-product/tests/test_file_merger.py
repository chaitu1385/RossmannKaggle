"""Tests for MultiFileMerger — join key detection and multi-file merge."""

import unittest
from datetime import date, timedelta

import numpy as np
import polars as pl

from src.data.file_classifier import FileClassifier
from src.data.file_merger import MultiFileMerger
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
    """Weekly time series: week, store_id, product_id, quantity."""
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
    """Dimension table: store_id, region, format."""
    return pl.DataFrame({
        "store_id": ["store_0", "store_1", "store_2"],
        "region": ["North", "South", "East"],
        "store_format": ["Large", "Small", "Medium"],
    })


def _make_regressor_table(
    n_weeks: int = 52,
    start_date: date = date(2022, 1, 3),
    seed: int = 42,
) -> pl.DataFrame:
    """Broadcast regressor: week, temperature, rainfall_mm."""
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
    """Store-level regressor: week, store_id, promo_discount."""
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


def _make_dimension_with_conflict() -> pl.DataFrame:
    """Dimension table with a column that conflicts with the primary."""
    return pl.DataFrame({
        "store_id": ["store_0", "store_1", "store_2"],
        "region": ["North", "South", "East"],
        "quantity": [100, 200, 300],  # Same name as primary's target
    })


def _classify(files: dict):
    """Helper: classify files and return result."""
    return FileClassifier().classify_files(files)


# --------------------------------------------------------------------------- #
#  Tests: join key detection
# --------------------------------------------------------------------------- #

class TestJoinKeyDetection(unittest.TestCase):
    """Test MultiFileMerger.detect_join_keys."""

    def setUp(self):
        self.merger = MultiFileMerger()
        self.primary_df = _make_primary_timeseries()

    def test_dimension_join_keys(self):
        result = _classify({
            "sales.csv": self.primary_df,
            "stores.csv": _make_dimension_table(),
        })
        spec = self.merger.detect_join_keys(
            result.primary_file, result.dimension_files[0],
        )
        self.assertIn("store_id", spec.join_keys)
        self.assertNotIn("week", spec.join_keys)

    def test_regressor_join_keys_broadcast(self):
        result = _classify({
            "sales.csv": self.primary_df,
            "weather.csv": _make_regressor_table(),
        })
        spec = self.merger.detect_join_keys(
            result.primary_file, result.regressor_files[0],
        )
        self.assertIn("week", spec.join_keys)

    def test_regressor_join_keys_with_ids(self):
        result = _classify({
            "sales.csv": self.primary_df,
            "promos.csv": _make_regressor_with_ids(),
        })
        spec = self.merger.detect_join_keys(
            result.primary_file, result.regressor_files[0],
        )
        self.assertIn("week", spec.join_keys)
        self.assertIn("store_id", spec.join_keys)

    def test_key_overlap_percentage(self):
        result = _classify({
            "sales.csv": self.primary_df,
            "stores.csv": _make_dimension_table(),
        })
        spec = self.merger.detect_join_keys(
            result.primary_file, result.dimension_files[0],
        )
        self.assertGreater(spec.key_overlap_pct, 0.0)
        self.assertLessEqual(spec.key_overlap_pct, 1.0)

    def test_no_overlapping_keys_produces_warning(self):
        result = _classify({
            "sales.csv": self.primary_df,
            "stores.csv": _make_dimension_table(),
        })
        # Create a profile with no matching columns
        from src.data.file_classifier import FileProfile
        fake = FileProfile(
            filename="random.csv",
            df=pl.DataFrame({"x": [1, 2], "y": ["a", "b"]}),
            role="dimension",
            confidence=0.5,
            time_column=None,
            id_columns=["x"],
            numeric_columns=[],
            categorical_columns=["y"],
            n_rows=2,
            n_columns=2,
        )
        spec = self.merger.detect_join_keys(result.primary_file, fake)
        self.assertEqual(len(spec.join_keys), 0)
        self.assertIsInstance(spec.warnings, list)
        self.assertGreaterEqual(len(spec.warnings), 1)


# --------------------------------------------------------------------------- #
#  Tests: merge preview
# --------------------------------------------------------------------------- #

class TestMergePreview(unittest.TestCase):
    """Test MultiFileMerger.preview_merge."""

    def setUp(self):
        self.merger = MultiFileMerger()

    def test_preview_single_file(self):
        result = _classify({"sales.csv": _make_primary_timeseries()})
        preview = self.merger.preview_merge(result)
        self.assertGreater(preview.total_rows, 0)
        self.assertEqual(preview.unmatched_primary_keys, 0)

    def test_preview_no_primary(self):
        from src.data.file_classifier import ClassificationResult
        result = ClassificationResult(
            profiles=[], primary_file=None,
            dimension_files=[], regressor_files=[],
            unknown_files=[], warnings=["test"],
        )
        preview = self.merger.preview_merge(result)
        self.assertEqual(preview.total_rows, 0)
        self.assertIsInstance(preview.warnings, list)
        self.assertGreaterEqual(len(preview.warnings), 1)

    def test_preview_with_dimension(self):
        result = _classify({
            "sales.csv": _make_primary_timeseries(),
            "stores.csv": _make_dimension_table(),
        })
        preview = self.merger.preview_merge(result)
        self.assertGreater(preview.total_rows, 0)
        # Should have more columns than primary alone
        primary_cols = len(_make_primary_timeseries().columns)
        self.assertGreater(preview.total_columns, primary_cols)

    def test_preview_sample_rows_limited(self):
        result = _classify({
            "sales.csv": _make_primary_timeseries(),
            "stores.csv": _make_dimension_table(),
        })
        preview = self.merger.preview_merge(result)
        self.assertLessEqual(preview.sample_rows.height, 10)


# --------------------------------------------------------------------------- #
#  Tests: full merge
# --------------------------------------------------------------------------- #

class TestMerge(unittest.TestCase):
    """Test MultiFileMerger.merge."""

    def setUp(self):
        self.merger = MultiFileMerger()

    def test_merge_preserves_primary_rows(self):
        primary = _make_primary_timeseries()
        result = _classify({
            "sales.csv": primary,
            "stores.csv": _make_dimension_table(),
        })
        merged = self.merger.merge(result)
        self.assertEqual(merged.df.height, primary.height)

    def test_merge_adds_dimension_columns(self):
        result = _classify({
            "sales.csv": _make_primary_timeseries(),
            "stores.csv": _make_dimension_table(),
        })
        merged = self.merger.merge(result)
        self.assertIn("region", merged.df.columns)
        self.assertIn("store_format", merged.df.columns)

    def test_merge_adds_regressor_columns(self):
        result = _classify({
            "sales.csv": _make_primary_timeseries(),
            "weather.csv": _make_regressor_table(),
        })
        merged = self.merger.merge(result)
        self.assertIn("temperature", merged.df.columns)
        self.assertIn("rainfall_mm", merged.df.columns)

    def test_merge_three_files(self):
        primary = _make_primary_timeseries()
        result = _classify({
            "sales.csv": primary,
            "stores.csv": _make_dimension_table(),
            "weather.csv": _make_regressor_table(),
        })
        merged = self.merger.merge(result)
        self.assertEqual(merged.df.height, primary.height)
        # Has columns from all three
        self.assertIn("region", merged.df.columns)
        self.assertIn("temperature", merged.df.columns)

    def test_merge_fills_regressor_nulls(self):
        result = _classify({
            "sales.csv": _make_primary_timeseries(),
            "weather.csv": _make_regressor_table(),
        })
        merged = self.merger.merge(result)
        # Regressor nulls should be filled with 0
        for col in ["temperature", "rainfall_mm"]:
            if col in merged.df.columns:
                null_count = merged.df[col].null_count()
                self.assertEqual(null_count, 0, f"Column {col} has {null_count} nulls")

    def test_merge_single_file(self):
        primary = _make_primary_timeseries()
        result = _classify({"sales.csv": primary})
        merged = self.merger.merge(result)
        self.assertEqual(merged.df.height, primary.height)
        self.assertEqual(len(merged.df.columns), len(primary.columns))

    def test_merge_no_primary_returns_empty(self):
        from src.data.file_classifier import ClassificationResult
        result = ClassificationResult(
            profiles=[], primary_file=None,
            dimension_files=[], regressor_files=[],
            unknown_files=[], warnings=[],
        )
        merged = self.merger.merge(result)
        self.assertEqual(merged.df.height, 0)

    def test_merge_join_specs_populated(self):
        result = _classify({
            "sales.csv": _make_primary_timeseries(),
            "stores.csv": _make_dimension_table(),
            "weather.csv": _make_regressor_table(),
        })
        merged = self.merger.merge(result)
        self.assertEqual(len(merged.join_specs), 2)

    def test_merge_regressor_with_ids(self):
        primary = _make_primary_timeseries()
        result = _classify({
            "sales.csv": primary,
            "promos.csv": _make_regressor_with_ids(),
        })
        merged = self.merger.merge(result)
        self.assertEqual(merged.df.height, primary.height)
        self.assertIn("promo_discount", merged.df.columns)


class TestColumnConflicts(unittest.TestCase):
    """Test duplicate column name resolution."""

    def setUp(self):
        self.merger = MultiFileMerger()

    def test_conflicting_column_renamed(self):
        result = _classify({
            "sales.csv": _make_primary_timeseries(),
            "stores.csv": _make_dimension_with_conflict(),
        })
        merged = self.merger.merge(result)
        # 'quantity' from stores.csv should be renamed to 'quantity_stores'
        self.assertIn("quantity", merged.df.columns)  # original
        self.assertIn("quantity_stores", merged.df.columns)  # renamed

    def test_conflict_recorded_in_preview(self):
        result = _classify({
            "sales.csv": _make_primary_timeseries(),
            "stores.csv": _make_dimension_with_conflict(),
        })
        merged = self.merger.merge(result)
        self.assertIsInstance(merged.preview.column_name_conflicts, list)
        self.assertGreaterEqual(len(merged.preview.column_name_conflicts), 1)


if __name__ == "__main__":
    unittest.main()
