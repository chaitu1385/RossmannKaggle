# Forecasting Platform

A modular time series forecasting system for predicting store sales, with an integrated SKU mapping discovery pipeline for tracking product transitions.

## Project Structure

```
forecasting-platform/
├── src/
│   ├── models/                    # Forecasting models
│   │   ├── base.py                # Abstract base forecaster
│   │   ├── xgboost_model.py       # XGBoost forecaster
│   │   └── lightgbm_model.py      # LightGBM forecaster
│   ├── data/                      # Data pipeline
│   │   ├── loader.py              # CSV data loading & merging
│   │   ├── preprocessor.py        # Cleaning & encoding
│   │   └── feature_engineering.py # Temporal, lag & rolling features
│   ├── evaluation/                # Model evaluation
│   │   ├── metrics.py             # RMSPE, RMSE, MAE, MAPE
│   │   └── evaluator.py           # Multi-model comparison
│   ├── sku_mapping/               # SKU transition discovery
│   │   ├── data/                  # Product master loader & mock data
│   │   ├── methods/               # Attribute matching, naming convention
│   │   ├── fusion/                # Candidate scoring & fusion
│   │   └── output/                # Mapping CSV writer
│   └── utils/                     # Config loading, logging
├── scripts/
│   ├── train.py                   # Model training CLI
│   ├── predict.py                 # Inference CLI
│   └── run_sku_mapping.py         # SKU mapping CLI
├── configs/                       # YAML configuration files
├── tests/                         # Unit tests
├── setup.py
└── requirements.txt
```

## Setup

```bash
cd forecasting-platform
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Sales Forecasting

### Models

- **XGBoost** — gradient boosting with early stopping (1000 estimators, depth 6, lr 0.05)
- **LightGBM** — light gradient boosting with automatic categorical feature detection

Both models extend a common `BaseForecaster` interface with `fit()`, `predict()`, `save()`, and `load()` methods.

### Feature Engineering

- **Temporal**: year, month, day, day-of-week, week-of-year, quarter, weekend/month-boundary flags
- **Lag features**: configurable per-store lags (default: 1, 7, 14, 30 days)
- **Rolling statistics**: mean and std over configurable windows (default: 7, 14, 30 days)
- **Competition**: months since competitor opened
- **Promo**: Promo2 active duration

### Training

```bash
python scripts/train.py --config configs/xgboost_config.yaml \
    --train-file data/train.csv --store-file data/store.csv
```

### Prediction

```bash
python scripts/predict.py --config configs/base_config.yaml \
    --model-path models/xgboost_model.pkl --test-file data/test.csv
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| RMSPE  | Root Mean Square Percentage Error (primary) |
| RMSE   | Root Mean Square Error |
| MAE    | Mean Absolute Error |
| MAPE   | Mean Absolute Percentage Error |

## SKU Mapping Discovery

Automatically discovers which discontinued SKUs map to new replacement SKUs during product transitions.

### Pipeline

```
ProductMaster → [AttributeMatching, NamingConvention] → CandidateFusion → CSV Output
```

### Discovery Methods

1. **Attribute Matching** — pairs old (Discontinued/Declining) with new (Active/Planned) SKUs within the same product family and segment, scoring on price tier, form factor, category, and launch gap
2. **Naming Convention** — fuzzy string matching via rapidfuzz to catch SKUs that follow naming patterns

### Fusion & Confidence

Candidates from both methods are fused with weighted scoring (attribute 2/3, naming 1/3) plus a multi-method agreement bonus. Confidence levels: High (≥0.75), Medium (≥0.50), Low (≥0.30), Very Low (<0.30).

### Usage

```bash
python scripts/run_sku_mapping.py --use-mock-data
python scripts/run_sku_mapping.py --input data/product_master.csv --output mappings.csv
```

## Configuration

All model hyperparameters and pipeline settings are driven by YAML configs in `configs/`:

- `base_config.yaml` — data paths, feature columns, validation split
- `xgboost_config.yaml` — XGBoost-specific parameters
- `lightgbm_config.yaml` — LightGBM-specific parameters
- `sku_mapping_config.yaml` — SKU mapping thresholds and weights

## Testing

```bash
pytest forecasting-platform/tests/
```

## Key Design Decisions

- **Temporal validation split** instead of random split to respect time series ordering
- **Training-only aggregated features** to prevent data leakage
- **Log-transformed target** (`log(Sales + 1)`) to reduce outlier sensitivity
- **Configuration-driven** — YAML configs separate hyperparameters from code
- **Extensible architecture** — abstract base classes for models and discovery methods
