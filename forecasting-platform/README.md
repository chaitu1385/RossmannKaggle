# Forecasting Platform

A modular time series forecasting platform built for scalable, production-ready sales forecasting. This platform supports multiple forecasting approaches including classical statistical models, machine learning, and deep learning methods.

## Features

- **Multiple Model Support**: ARIMA, Prophet, XGBoost, LightGBM, LSTM, Transformer-based models
- **Automated Feature Engineering**: Temporal features, lag features, rolling statistics
- **Model Registry**: Track and version trained models
- **Evaluation Framework**: Comprehensive metrics (RMSE, MAE, MAPE, RMSPE)
- **Pipeline Orchestration**: End-to-end training and inference pipelines
- **Data Validation**: Automated data quality checks

## Project Structure

```
forecasting-platform/
├── src/
│   ├── models/          # Forecasting model implementations
│   ├── data/            # Data loading and preprocessing
│   ├── evaluation/      # Model evaluation and metrics
│   └── utils/           # Utility functions
├── tests/               # Unit and integration tests
├── configs/             # Configuration files
├── notebooks/           # Jupyter notebooks for exploration
└── scripts/             # Training and inference scripts
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model
python scripts/train.py --config configs/xgboost_config.yaml

# Generate forecasts
python scripts/predict.py --model-path models/xgboost_v1 --output-path forecasts/
```

## Configuration

Models are configured via YAML files in the `configs/` directory. See `configs/base_config.yaml` for available options.

## Evaluation

The platform uses RMSPE (Root Mean Square Percentage Error) as the primary metric.

```
RMSPE = sqrt(mean(((y_true - y_pred) / y_true)^2))
```
