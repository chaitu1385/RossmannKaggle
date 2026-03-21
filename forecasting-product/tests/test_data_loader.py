"""
Tests for DataLoader — src/data/loader.py.

Covers:
  - load_train / load_test / load_store
  - load_all returns 3-tuple
  - merge_with_store
  - missing file raises FileNotFoundError
  - date column parsing
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd


def _write_csv(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestDataLoader(unittest.TestCase):

    def _make_loader(self, tmpdir: str):
        from src.data.loader import DataLoader
        return DataLoader(tmpdir)

    def test_load_train(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_csv(
                Path(tmpdir) / "train.csv",
                "Date,Store,Sales\n2024-01-01,1,100\n2024-01-02,2,200\n",
            )
            loader = self._make_loader(tmpdir)
            df = loader.load_train()
            self.assertEqual(len(df), 2)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["Date"]))

    def test_load_test(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_csv(
                Path(tmpdir) / "test.csv",
                "Date,Store,Id\n2024-02-01,1,1\n2024-02-02,2,2\n",
            )
            loader = self._make_loader(tmpdir)
            df = loader.load_test()
            self.assertEqual(len(df), 2)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["Date"]))

    def test_load_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_csv(
                Path(tmpdir) / "store.csv",
                "Store,StoreType,Assortment\n1,a,a\n2,b,b\n",
            )
            loader = self._make_loader(tmpdir)
            df = loader.load_store()
            self.assertEqual(len(df), 2)
            self.assertIn("StoreType", df.columns)

    def test_load_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_csv(Path(tmpdir) / "train.csv", "Date,Store,Sales\n2024-01-01,1,100\n")
            _write_csv(Path(tmpdir) / "test.csv", "Date,Store,Id\n2024-02-01,1,1\n")
            _write_csv(Path(tmpdir) / "store.csv", "Store,StoreType\n1,a\n")
            loader = self._make_loader(tmpdir)
            train, test, store = loader.load_all()
            self.assertIsInstance(train, pd.DataFrame)
            self.assertIsInstance(test, pd.DataFrame)
            self.assertIsInstance(store, pd.DataFrame)

    def test_merge_with_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = self._make_loader(tmpdir)
            sales = pd.DataFrame({"Store": [1, 2], "Sales": [100, 200]})
            store = pd.DataFrame({"Store": [1, 2], "StoreType": ["a", "b"]})
            merged = loader.merge_with_store(sales, store)
            self.assertIn("StoreType", merged.columns)
            self.assertEqual(len(merged), 2)

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = self._make_loader(tmpdir)
            with self.assertRaises(FileNotFoundError):
                loader.load_train()


if __name__ == "__main__":
    unittest.main()
