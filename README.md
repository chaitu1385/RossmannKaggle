# Rossmann Store Sales Forecasting

Capstone project for the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) Kaggle competition. Predicts daily sales for 1,115 Rossmann drug stores across Germany, 6 weeks into the future.

## Project Structure

```
├── Data-Exploration/
│   ├── Capstone Project.ipynb          # Main modeling notebook (XGBoost + LightGBM)
│   ├── Rossmann Stores Forecasting.ipynb  # Exploratory data analysis
│   ├── Capstone Project Report.pdf     # Written report
│   ├── Capstone Project Deck.pdf       # Presentation slides
│   └── Rossmann.zip                    # Competition data
├── Problem Sets/                       # ML coursework exercises
├── Inferential-Stats/                  # Statistical inference exercises
├── requirements.txt                    # Python dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Extract `Data-Exploration/Rossmann.zip` into the `Data-Exploration/` directory so that `train.csv`, `test.csv`, and `store.csv` are available.

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
