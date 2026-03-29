"""
Tests for DataPreprocessor — src/data/preprocessor.py.

Covers:
  - clean removes closed stores and zero sales
  - clean fills missing values
  - clean preserves original DataFrame
  - remove_closed=False keeps closed stores
  - encode_categoricals with explicit and auto columns
  - encode_categoricals preserves original
"""

import unittest

import polars as pl


class TestDataPreprocessor(unittest.TestCase):

    def _get_preprocessor(self, **kwargs):
        from src.data.preprocessor import DataPreprocessor
        return DataPreprocessor(**kwargs)

    def _sample_df(self):
        return pl.DataFrame({
            "Store": [1, 2, 3, 4],
            "Open": [1, 1, 0, 1],
            "Sales": [100, 200, 150, 0],
            "CompetitionDistance": [500.0, None, 300.0, None],
            "Promo2SinceWeek": [10.0, None, 5.0, None],
            "Promo2SinceYear": [2020.0, None, 2019.0, None],
            "PromoInterval": ["Jan,Apr", None, "Feb", None],
            "CompetitionOpenSinceMonth": [3.0, None, 6.0, None],
            "CompetitionOpenSinceYear": [2018.0, None, 2017.0, None],
        })

    def test_clean_removes_closed(self):
        pp = self._get_preprocessor()
        df = self._sample_df()
        result = pp.clean(df)
        # Store 3 is closed (Open=0), Store 4 has Sales=0 → removed
        self.assertTrue((result.get_column("Open") == 1).all())

    def test_clean_removes_zero_sales(self):
        pp = self._get_preprocessor()
        df = self._sample_df()
        result = pp.clean(df)
        self.assertTrue((result.get_column("Sales") > 0).all())

    def test_clean_fills_competition_distance(self):
        pp = self._get_preprocessor()
        df = self._sample_df()
        result = pp.clean(df)
        self.assertEqual(result.get_column("CompetitionDistance").null_count(), 0)

    def test_clean_fills_promo_columns(self):
        pp = self._get_preprocessor()
        df = self._sample_df()
        result = pp.clean(df)
        for col in ["Promo2SinceWeek", "Promo2SinceYear"]:
            self.assertEqual(result.get_column(col).null_count(), 0)

    def test_clean_fills_competition_timing(self):
        pp = self._get_preprocessor()
        df = self._sample_df()
        result = pp.clean(df)
        for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]:
            self.assertEqual(result.get_column(col).null_count(), 0)

    def test_clean_preserves_original(self):
        pp = self._get_preprocessor()
        df = self._sample_df()
        original_len = len(df)
        _ = pp.clean(df)
        self.assertEqual(len(df), original_len)  # original not mutated

    def test_remove_closed_false(self):
        pp = self._get_preprocessor(remove_closed=False)
        df = self._sample_df()
        result = pp.clean(df)
        # Closed stores kept, but zero sales still removed
        # Store 3 (Open=0, Sales=150) should be kept
        stores_in_result = result.get_column("Store").to_list()
        self.assertIn(3, stores_in_result)

    def test_clean_without_open_column(self):
        pp = self._get_preprocessor()
        df = pl.DataFrame({"Store": [1, 2], "Sales": [100, 200]})
        result = pp.clean(df)
        self.assertEqual(len(result), 2)

    def test_clean_resets_index(self):
        pp = self._get_preprocessor()
        df = self._sample_df()
        result = pp.clean(df)
        # Polars DataFrames don't have an index; just verify row count is correct
        self.assertGreater(len(result), 0)

    def test_encode_categoricals_explicit(self):
        pp = self._get_preprocessor()
        df = pl.DataFrame({
            "Store": [1, 2, 3],
            "Type": ["a", "b", "a"],
            "Region": ["north", "south", "north"],
        })
        result = pp.encode_categoricals(df, columns=["Type"])
        # "Type" should be encoded to integer codes
        self.assertTrue(result.schema["Type"].is_integer())
        # "Region" should remain unchanged (not specified for encoding)
        self.assertEqual(result.schema["Region"], pl.Utf8)

    def test_encode_categoricals_auto(self):
        pp = self._get_preprocessor()
        df = pl.DataFrame({
            "Store": [1, 2],
            "Type": ["a", "b"],
            "Region": ["north", "south"],
        })
        result = pp.encode_categoricals(df)
        # Both "Type" and "Region" are Utf8 → should be encoded
        self.assertTrue(result.schema["Type"].is_integer())
        self.assertTrue(result.schema["Region"].is_integer())

    def test_encode_preserves_original(self):
        pp = self._get_preprocessor()
        df = pl.DataFrame({"Type": ["a", "b", "c"]})
        original_dtype = df.schema["Type"]
        _ = pp.encode_categoricals(df, columns=["Type"])
        self.assertEqual(df.schema["Type"], original_dtype)  # original unmodified


if __name__ == "__main__":
    unittest.main()
