#!/usr/bin/env python3
"""Inference script for generating sales forecasts."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader, DataPreprocessor, FeatureEngineer
from src.models.base import BaseForecaster
from src.utils import get_logger, load_config

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate sales forecasts")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--model-path", required=True, help="Path to saved model directory")
    parser.add_argument("--model-name", required=True, help="Model name (used to find .pkl file)")
    parser.add_argument("--output-path", required=True, help="Path to save predictions CSV")
    parser.add_argument("--data-dir", help="Override data directory from config")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    data_dir = args.data_dir or config["data"]["data_dir"]

    logger.info(f"Loading data from {data_dir}")
    loader = DataLoader(data_dir)
    _, test_df, store_df = loader.load_all()
    test_df = loader.merge_with_store(test_df, store_df)

    logger.info("Preprocessing test data")
    preprocessor = DataPreprocessor(remove_closed=False)
    test_df = preprocessor.encode_categoricals(
        test_df, config["features"]["categorical_cols"]
    )

    logger.info("Engineering features")
    engineer = FeatureEngineer(
        lag_periods=config["features"]["lag_periods"],
        rolling_windows=config["features"]["rolling_windows"],
    )
    test_df = engineer.fit_transform(test_df)

    logger.info(f"Loading model from {args.model_path}")
    model = BaseForecaster.load(args.model_path, args.model_name)

    logger.info("Generating predictions")
    exclude_cols = {"Sales", "Date", "Id"}
    feature_cols = [c for c in test_df.columns if c not in exclude_cols]
    # Convert to Polars for model API (models accept pl.DataFrame)
    import polars as pl
    test_features_pl = pl.from_pandas(test_df[feature_cols])
    predictions = model.predict(test_features_pl)

    output = pd.DataFrame({"Id": test_df["Id"], "Sales": predictions})
    output["Sales"] = output["Sales"].clip(lower=0)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    logger.info(f"Saved {len(output)} predictions to {args.output_path}")


if __name__ == "__main__":
    main()
