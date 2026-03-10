# Sales Forecasting

A time series forecasting project for predicting daily store sales.

## Project Structure

```
├── Data-Exploration/
│   └── notes                          # Research notes
├── Problem Sets/                       # ML coursework exercises
├── Inferential-Stats/                  # Statistical inference exercises
├── requirements.txt                    # Python dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Approach

1. **EDA**: Sales distributions, trends by store type, correlation analysis, day-of-week/promo effects
2. **Feature Engineering**:
   - Time features: day, week, month, year, weekend flag, month start/end
   - Store features: type, assortment, competition distance & duration, promo2 duration
   - Lag features: 7-day and 30-day sales lags per store
   - Rolling averages: 7-day and 30-day moving averages per store
   - Aggregated features: average sales/customers per store (computed from training data only)
3. **Modeling**: XGBoost + LightGBM ensemble with temporal validation (last 6 weeks held out)
4. **Evaluation**: Root Mean Square Percentage Error (RMSPE)

## Key Design Decisions

- **Temporal validation split** instead of random split to respect the time series nature of the data
- **Training-only aggregated features** to prevent data leakage
- **Log-transformed target** (`log(Sales + 1)`) to reduce sensitivity to outliers
- **Intelligent missing value handling**: median fill for distances, 0 for unknown dates
- **Model blending**: simple average of XGBoost and LightGBM predictions
