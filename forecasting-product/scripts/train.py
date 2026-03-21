#!/usr/bin/env python3
"""Training script for the forecasting product."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader, DataPreprocessor, FeatureEngineer
from src.models import XGBoostForecaster, LightGBMForecaster
from src.evaluation import ModelEvaluator
from src.utils import get_logger, load_config

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a forecasting model")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--data-dir", help="Override data directory from config")
    parser.add_argument("--output-dir", help="Override output directory from config")
    return parser.parse_args()


def build_model(config):
    model_type = config["model"]["type"]
    name = config["model"]["name"]
    params = config["model"].get("params", {})

    if model_type == "xgboost":
        return XGBoostForecaster(name=name, params=params)
    elif model_type == "lightgbm":
        return LightGBMForecaster(name=name, params=params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    args = parse_args()
    config = load_config(args.config)

    data_dir = args.data_dir or config["data"]["data_dir"]
    output_dir = args.output_dir or config["training"]["output_dir"]

    logger.info(f"Loading data from {data_dir}")
    loader = DataLoader(data_dir)
    train_df, test_df, store_df = loader.load_all()

    train_df = loader.merge_with_store(train_df, store_df)
    test_df = loader.merge_with_store(test_df, store_df)

    logger.info("Preprocessing data")
    preprocessor = DataPreprocessor()
    train_df = preprocessor.clean(train_df)
    train_df = preprocessor.encode_categoricals(
        train_df, config["features"]["categorical_cols"]
    )

    logger.info("Engineering features")
    engineer = FeatureEngineer(
        lag_periods=config["features"]["lag_periods"],
        rolling_windows=config["features"]["rolling_windows"],
    )
    train_df = engineer.fit_transform(train_df)
    train_df = train_df.dropna()

    target = config["data"]["target_col"]
    exclude_cols = {target, "Date", "Id"}
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    val_size = config["training"]["val_size"]
    split_idx = int(len(train_df) * (1 - val_size))
    X_train = train_df.iloc[:split_idx][feature_cols]
    y_train = train_df.iloc[:split_idx][target]
    X_val = train_df.iloc[split_idx:][feature_cols]
    y_val = train_df.iloc[split_idx:][target]

    logger.info("Training model")
    model = build_model(config)
    model.fit(X_train, y_train, X_val, y_val)

    logger.info("Evaluating model")
    evaluator = ModelEvaluator()
    scores = evaluator.evaluate(model, X_val, y_val)
    for metric, score in scores.items():
        logger.info(f"  {metric}: {score:.4f}")

    logger.info(f"Saving model to {output_dir}")
    model.save(output_dir)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
