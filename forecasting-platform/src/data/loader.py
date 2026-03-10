"""Data loading utilities for the forecasting platform."""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


class DataLoader:
    """Loads and validates raw data for forecasting."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_train(self) -> pd.DataFrame:
        """Load training data."""
        path = self.data_dir / "train.csv"
        df = pd.read_csv(path, parse_dates=["Date"], low_memory=False)
        return df

    def load_test(self) -> pd.DataFrame:
        """Load test data."""
        path = self.data_dir / "test.csv"
        df = pd.read_csv(path, parse_dates=["Date"], low_memory=False)
        return df

    def load_store(self) -> pd.DataFrame:
        """Load store metadata."""
        path = self.data_dir / "store.csv"
        df = pd.read_csv(path, low_memory=False)
        return df

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all datasets and return (train, test, store)."""
        train = self.load_train()
        test = self.load_test()
        store = self.load_store()
        return train, test, store

    def merge_with_store(
        self, df: pd.DataFrame, store: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge sales data with store metadata."""
        return df.merge(store, on="Store", how="left")
