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

import numpy as np
import pandas as pd


class TestDataPreprocessor(unittest.TestCase):

    def _get_preprocessor(self, **kwargs):
        from src.data.preprocessor import DataPreprocessor
        return DataPreprocessor(**kwargs)

    def _sample_df(self):
        return pd.DataFrame({
            "Store": [1, 2, 3, 4],
            "Open": [1, 1, 0, 1],
            "Sales": [100, 200, 150, 0],
            "CompetitionDistance": [500.0, np.nan, 300.0, np.nan],
            "Promo2SinceWeek": [10, np.nan, 5, np.nan],
            "Promo2SinceYear": [2020, np.nan, 2019, np.nan],
            "PromoInterval": ["Jan,Apr", np.nan, "Feb", np.nan],
            "CompetitionOpenSinceMonth": [3, np.nan, 6, np.nan],
            "CompetitionOpenSinceYear": [2018, np.nan, 2017, np.nan],
        })

    def test_clean_removes_closed(self):
        pp = self._get_preprocessor()
        df = self._sample_df()
        result = pp.clean(df)
        # Store 3 is closed (Open=0), Store 4 has Sales=0 → removed
        self.assertTrue((result["Open"] == 1).all())

    def test_clean_removes_zero_sales(self):
        pp = self._get_preprocessor()
        df = self._sample_df()
        result = pp.clean(df)
        self.assertTrue((result["Sales"] > 0).all())

    def test_clean_fills_competition_distance(self):
        pp = self._get_preprocessor()
        df = self._sample_df()
        result = pp.clean(df)
        self.assertFalse(result["CompetitionDistance"].isna().any())

    def test_clean_fills_promo_columns(self):
        pp = self._get_preprocessor()
        df = self._sample_df()
        result = pp.clean(df)
        for col in ["Promo2SinceWeek", "Promo2SinceYear"]:
            self.assertFalse(result[col].isna().any())

    def test_clean_fills_competition_timing(self):
        pp = self._get_preprocessor()
        df = self._sample_df()
        result = pp.clean(df)
        for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]:
            self.assertFalse(result[col].isna().any())

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
        stores_in_result = result["Store"].tolist()
        self.assertIn(3, stores_in_result)

    def test_clean_without_open_column(self):
        pp = self._get_preprocessor()
        df = pd.DataFrame({"Store": [1, 2], "Sales": [100, 200]})
        result = pp.clean(df)
        self.assertEqual(len(result), 2)

    def test_clean_resets_index(self):
        pp = self._get_preprocessor()
        df = self._sample_df()
        result = pp.clean(df)
        self.assertEqual(list(result.index), list(range(len(result))))

    def test_encode_categoricals_explicit(self):
        pp = self._get_preprocessor()
        df = pd.DataFrame({
            "Store": [1, 2, 3],
            "Type": ["a", "b", "a"],
            "Region": ["north", "south", "north"],
        })
        result = pp.encode_categoricals(df, columns=["Type"])
        # "Type" should be encoded to integer codes
        self.assertTrue(pd.api.types.is_integer_dtype(result["Type"]))
        # "Region" should remain unchanged (not specified for encoding)
        self.assertFalse(pd.api.types.is_integer_dtype(result["Region"]))

    def test_encode_categoricals_auto(self):
        pp = self._get_preprocessor()
        df = pd.DataFrame({
            "Store": [1, 2],
            "Type": ["a", "b"],
            "Region": ["north", "south"],
        })
        result = pp.encode_categoricals(df)
        # Both "Type" and "Region" are object → should be encoded
        self.assertTrue(pd.api.types.is_integer_dtype(result["Type"]))
        self.assertTrue(pd.api.types.is_integer_dtype(result["Region"]))

    def test_encode_preserves_original(self):
        pp = self._get_preprocessor()
        df = pd.DataFrame({"Type": ["a", "b", "c"]})
        original_dtype = df["Type"].dtype
        _ = pp.encode_categoricals(df, columns=["Type"])
        self.assertEqual(df["Type"].dtype, original_dtype)  # original unmodified


if __name__ == "__main__":
    unittest.main()
