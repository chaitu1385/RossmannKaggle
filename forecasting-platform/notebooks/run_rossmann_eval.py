"""
Rossmann evaluation script — reruns the key benchmarks from the notebook
to verify ML improvements after the data integrity fixes.
"""
import sys
import os
import warnings
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# Add platform root to path
platform_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if platform_root not in sys.path:
    sys.path.insert(0, platform_root)

from src.config.schema import (
    PlatformConfig, ForecastConfig, BacktestConfig, OutputConfig,
    HierarchyConfig, ReconciliationConfig, TransitionConfig,
)
from src.forecasting.naive import SeasonalNaiveForecaster
from src.forecasting.statistical import AutoETSForecaster
from src.forecasting.ml import LGBMDirectForecaster
from src.metrics.definitions import wmape
from src.metrics.fva import compute_fva_cascade, compute_total_fva
from src.backtesting.cross_validator import WalkForwardCV

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

# ══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════
DATA_DIR = Path(platform_root) / "data" / "rossmann"
TRAIN_PATH = DATA_DIR / "train.csv"
STORE_PATH = DATA_DIR / "store.csv"

if not TRAIN_PATH.exists() or not STORE_PATH.exists():
    print("ERROR: Rossmann data not found.")
    sys.exit(1)

raw_train = (
    pl.read_csv(str(TRAIN_PATH), infer_schema_length=10000,
                schema_overrides={"StateHoliday": pl.Utf8})
    .with_columns(pl.col("Date").str.to_date("%m/%d/%Y"))
)
raw_store = pl.read_csv(str(STORE_PATH))

df = (
    raw_train
    .filter((pl.col("Open") == 1) & (pl.col("Sales") > 0))
    .join(raw_store.select(["Store", "StoreType", "Assortment"]), on="Store", how="left")
)

# Aggregate daily → weekly
weekly = (
    df
    .with_columns(pl.col("Date").dt.truncate("1w").alias("week"))
    .group_by(["Store", "StoreType", "week"])
    .agg([
        pl.col("Sales").sum().alias("quantity"),
        pl.col("Promo").mean().alias("promo_ratio"),
    ])
    .sort(["Store", "week"])
)

# Stratified subsample
np.random.seed(42)
STORES_PER_TYPE = 12
sampled_stores = []
for st in sorted(weekly["StoreType"].unique().to_list()):
    stores_in_type = weekly.filter(pl.col("StoreType") == st)["Store"].unique().to_list()
    n = min(STORES_PER_TYPE, len(stores_in_type))
    sampled_stores.extend(np.random.choice(stores_in_type, size=n, replace=False).tolist())

data = (
    weekly
    .filter(pl.col("Store").is_in(sampled_stores))
    .with_columns([
        pl.col("Store").cast(pl.Utf8).alias("series_id"),
        pl.col("StoreType").alias("store_type"),
        pl.col("Store").cast(pl.Utf8).alias("store"),
    ])
)

# Fill missing weeks
_min_w, _max_w = data["week"].min(), data["week"].max()
_all_weeks = pl.date_range(_min_w, _max_w, interval="1w", eager=True)
_all_ids = data.select(["series_id", "store_type", "store"]).unique()
_grid = _all_ids.join(pl.DataFrame({"week": _all_weeks}), how="cross")
data = (
    _grid.join(data.select(["series_id", "week", "quantity", "promo_ratio"]),
               on=["series_id", "week"], how="left")
    .with_columns([
        pl.col("quantity").fill_null(0.0),
        pl.col("promo_ratio").fill_null(0.0),
    ])
    .sort(["series_id", "week"])
)

n_stores = data["series_id"].n_unique()
n_weeks = data["week"].n_unique()
print(f"Data loaded: {n_stores} stores × {n_weeks} weeks")
print(f"Date range: {data['week'].min()} → {data['week'].max()}")
print()

# ══════════════════════════════════════════════════════════════════════
# 2. BACKTEST HELPER
# ══════════════════════════════════════════════════════════════════════

def manual_backtest(series_df, forecaster_factory, horizon=13, n_folds=2):
    cv = WalkForwardCV(n_folds=n_folds, val_weeks=horizon, gap_weeks=0)
    folds = cv.split_data(series_df, time_col="week")

    all_results = []
    for fold_idx, (fold_info, train, val) in enumerate(folds):
        model = forecaster_factory()
        model.fit(train, target_col="quantity", time_col="week", id_col="series_id")
        preds = model.predict(horizon=horizon, id_col="series_id", time_col="week")
        val_weeks = sorted(val["week"].unique().to_list())
        pred_weeks = sorted(preds["week"].unique().to_list())
        week_map = dict(zip(pred_weeks[:len(val_weeks)], val_weeks[:len(pred_weeks)]))
        preds = preds.filter(pl.col("week").is_in(list(week_map.keys())))
        preds = preds.with_columns(pl.col("week").replace(week_map).alias("week"))

        merged = val.join(preds, on=["series_id", "week"], how="inner")
        for sid in merged["series_id"].unique().to_list():
            s = merged.filter(pl.col("series_id") == sid)
            actual = s["quantity"]
            forecast = s["forecast"]
            w = float((actual - forecast).abs().sum() / actual.abs().sum()) if actual.abs().sum() > 0 else 0
            all_results.append({"series_id": sid, "fold": fold_idx, "wmape": w})

    return pl.DataFrame(all_results)


# ══════════════════════════════════════════════════════════════════════
# 3. BENCHMARK: LightGBM vs Naive (Backtest)
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("BENCHMARK 1: Walk-Forward Backtest (2 folds × 13 weeks)")
print("=" * 70)

data_no_promo = data.select(["series_id", "week", "quantity"])

print("\nRunning Seasonal Naive backtest...")
wmape_naive = manual_backtest(
    data_no_promo,
    forecaster_factory=lambda: SeasonalNaiveForecaster(season_length=52),
    horizon=13, n_folds=2,
)
naive_mean = wmape_naive["wmape"].mean()
print(f"  Naive mean WMAPE: {naive_mean:.4f} ({naive_mean*100:.1f}%)")

print("\nRunning LightGBM backtest (IMPROVED — enriched features + tuned hyperparams)...")
wmape_lgbm = manual_backtest(
    data_no_promo,
    forecaster_factory=lambda: LGBMDirectForecaster(num_threads=1),
    horizon=13, n_folds=2,
)
lgbm_mean = wmape_lgbm["wmape"].mean()
print(f"  LightGBM mean WMAPE: {lgbm_mean:.4f} ({lgbm_mean*100:.1f}%)")

delta_bp = (naive_mean - lgbm_mean) * 100
print(f"\n  Delta (Naive - LightGBM): {delta_bp:+.2f} pp")
if lgbm_mean < naive_mean:
    print(f"  ✓ LightGBM BEATS naive by {abs(delta_bp):.1f} percentage points")
else:
    print(f"  ✗ LightGBM LOSES to naive by {abs(delta_bp):.1f} percentage points")

# Per-store comparison
comp = (
    wmape_naive.group_by("series_id").agg(pl.col("wmape").mean().alias("naive_wmape"))
    .join(
        wmape_lgbm.group_by("series_id").agg(pl.col("wmape").mean().alias("lgbm_wmape")),
        on="series_id",
    )
    .with_columns((pl.col("lgbm_wmape") < pl.col("naive_wmape")).alias("lgbm_wins"))
)
pct_wins = comp["lgbm_wins"].sum() / comp.shape[0] * 100
print(f"  Stores where LightGBM wins: {pct_wins:.0f}%")

# ══════════════════════════════════════════════════════════════════════
# 4. BENCHMARK: Promo vs No-Promo
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("BENCHMARK 2: External Regressors (promo_ratio)")
print("=" * 70)

data_with_promo = data.select(["series_id", "week", "quantity", "promo_ratio"])

# 2a: Old behaviour — promo_ratio treated as default (known_ahead, forward-filled)
print("\nRunning LightGBM WITH promo (OLD: forward-fill, no temporal validation)...")
wmape_with_promo = manual_backtest(
    data_with_promo,
    forecaster_factory=lambda: LGBMDirectForecaster(num_threads=1),
    horizon=13, n_folds=2,
)
promo_mean_old = wmape_with_promo["wmape"].mean()
print(f"  LightGBM (promo, forward-fill): {promo_mean_old:.4f} ({promo_mean_old*100:.1f}%)")

# 2b: New behaviour — promo_ratio marked as contemporaneous → dropped at predict
from src.forecasting.feature_manager import MLForecastFeatureManager

def make_lgbm_with_temporal_validation():
    m = LGBMDirectForecaster(num_threads=1)
    # Mark promo_ratio as contemporaneous so it gets dropped at prediction time
    m._feature_mgr = MLForecastFeatureManager(
        feature_types={"promo_ratio": "contemporaneous"}
    )
    return m

print("Running LightGBM WITH promo (NEW: contemporaneous feature dropped at predict)...")
wmape_with_promo_new = manual_backtest(
    data_with_promo,
    forecaster_factory=make_lgbm_with_temporal_validation,
    horizon=13, n_folds=2,
)
promo_mean_new = wmape_with_promo_new["wmape"].mean()
print(f"  LightGBM (promo, temporal validation): {promo_mean_new:.4f} ({promo_mean_new*100:.1f}%)")

print(f"\n  Summary:")
print(f"    No promo:                         {lgbm_mean*100:.1f}%")
print(f"    Promo (old: forward-fill):         {promo_mean_old*100:.1f}%  ({(lgbm_mean - promo_mean_old)*100:+.1f} pp)")
print(f"    Promo (new: temporal validation):   {promo_mean_new*100:.1f}%  ({(lgbm_mean - promo_mean_new)*100:+.1f} pp)")

promo_mean = promo_mean_new  # use for summary
promo_delta = (lgbm_mean - promo_mean) * 100

# ══════════════════════════════════════════════════════════════════════
# 5. FVA CASCADE: Naive → ETS → LightGBM
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("BENCHMARK 3: FVA Cascade (Naive → ETS → LightGBM)")
print("=" * 70)

HOLDOUT_WEEKS = 13
max_week = data["week"].max()
cutoff = max_week - timedelta(weeks=HOLDOUT_WEEKS)
fva_train = data_no_promo.filter(pl.col("week") <= cutoff)
fva_holdout = data_no_promo.filter(pl.col("week") > cutoff)

print("\nFitting Naive...")
naive_model = SeasonalNaiveForecaster(season_length=52)
naive_model.fit(fva_train, target_col="quantity", time_col="week", id_col="series_id")
naive_preds = naive_model.predict(horizon=HOLDOUT_WEEKS, id_col="series_id", time_col="week")

print("Fitting ETS...")
ets_model = AutoETSForecaster()
ets_model.fit(fva_train, target_col="quantity", time_col="week", id_col="series_id")
ets_preds = ets_model.predict(horizon=HOLDOUT_WEEKS, id_col="series_id", time_col="week")

print("Fitting LightGBM...")
lgbm_model = LGBMDirectForecaster(num_threads=1)
lgbm_model.fit(fva_train, target_col="quantity", time_col="week", id_col="series_id")
lgbm_preds = lgbm_model.predict(horizon=HOLDOUT_WEEKS, id_col="series_id", time_col="week")

# Align predictions with holdout
def align_preds(preds, holdout):
    holdout_weeks = sorted(holdout["week"].unique().to_list())
    pred_weeks = sorted(preds["week"].unique().to_list())
    week_map = dict(zip(pred_weeks[:len(holdout_weeks)], holdout_weeks[:len(pred_weeks)]))
    preds = preds.filter(pl.col("week").is_in(list(week_map.keys())))
    preds = preds.with_columns(pl.col("week").replace(week_map).alias("week"))
    return preds.join(holdout, on=["series_id", "week"], how="inner")

naive_eval = align_preds(naive_preds, fva_holdout)
ets_eval = align_preds(ets_preds, fva_holdout)
lgbm_eval = align_preds(lgbm_preds, fva_holdout)

# FVA cascade
common_stores = (
    set(naive_eval["series_id"].unique().to_list()) &
    set(ets_eval["series_id"].unique().to_list()) &
    set(lgbm_eval["series_id"].unique().to_list())
)
print(f"\nCommon stores across all 3 layers: {len(common_stores)}")

fva_results = []
for sid in sorted(common_stores):
    actual_s = naive_eval.filter(pl.col("series_id") == sid)["quantity"]
    forecasts_dict = {
        "naive": naive_eval.filter(pl.col("series_id") == sid)["forecast"],
        "statistical": ets_eval.filter(pl.col("series_id") == sid)["forecast"],
        "ml": lgbm_eval.filter(pl.col("series_id") == sid)["forecast"],
    }
    cascade = compute_fva_cascade(actual_s, forecasts_dict)
    for layer_result in cascade:
        fva_results.append({"series_id": sid, **layer_result})

fva_df = pl.DataFrame(fva_results)
fva_summary = (
    fva_df
    .group_by("layer")
    .agg([
        pl.col("wmape").mean().alias("mean_wmape"),
        pl.col("fva_wmape").mean().alias("mean_fva_wmape"),
    ])
)

print("\nFVA Cascade Results:")
print("-" * 50)
for layer in ["naive", "statistical", "ml"]:
    row = fva_summary.filter(pl.col("layer") == layer)
    if row.is_empty():
        continue
    w = row["mean_wmape"][0]
    d = row["mean_fva_wmape"][0]
    marker = "✓" if d <= 0 else "✗"
    print(f"  {layer:12s}  WMAPE: {w*100:6.2f}%  FVA delta: {d*100:+6.2f} pp  {marker}")

# ML adds value?
ml_row = fva_summary.filter(pl.col("layer") == "ml")
if not ml_row.is_empty():
    ml_delta = ml_row["mean_fva_wmape"][0]
    if ml_delta < 0:
        print(f"\n  ✓ ML layer ADDS value (improves by {abs(ml_delta)*100:.1f} pp)")
    else:
        print(f"\n  ✗ ML layer DESTROYS value (worsens by {abs(ml_delta)*100:.1f} pp)")

# ══════════════════════════════════════════════════════════════════════
# 6. END-TO-END BACKTEST PIPELINE
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("BENCHMARK 4: Full BacktestPipeline (champion selection)")
print("=" * 70)

import tempfile
from src.pipeline.backtest import BacktestPipeline

tmpdir = tempfile.mkdtemp()
bt_config = PlatformConfig(
    lob="rossmann",
    output=OutputConfig(metrics_path=tmpdir),
    forecast=ForecastConfig(
        horizon_weeks=13,
        forecasters=["naive_seasonal", "auto_ets", "lgbm_direct"],
    ),
    backtest=BacktestConfig(
        n_folds=2,
        val_weeks=13,
        selection_strategy="champion",
    ),
)

pipeline = BacktestPipeline(bt_config)
bt_results = pipeline.run(data.select(["series_id", "week", "quantity"]))

print(f"\nBacktest Results:")
print(f"  Total rows: {bt_results['backtest_results'].shape[0]:,}")
print(f"  Models: {bt_results['backtest_results']['model_id'].unique().to_list()}")
print(f"  Folds: {bt_results['backtest_results']['fold'].unique().to_list()}")

print(f"\nChampion: {bt_results['champions']}")

leaderboard = bt_results["leaderboard"]
if not leaderboard.is_empty():
    print("\nLeaderboard:")
    for row in leaderboard.iter_rows(named=True):
        m = row.get("model_id", "?")
        w = row.get("wmape", 0)
        print(f"  {m:20s}  WMAPE: {w*100:.2f}%")

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("SUMMARY — Before vs After")
print("=" * 70)
print(f"""
  BEFORE (original notebook results):
    Naive WMAPE:              12.6%  (was 10.7% in some runs)
    LightGBM WMAPE:           16.0%  (3.5pp worse than naive)
    LightGBM + promo WMAPE:   26.6%  (10.7pp worse than no-promo)
    FVA: ML destroyed value

  AFTER (with enriched features + tuned hyperparams):
    Naive WMAPE:              {naive_mean*100:.1f}%
    LightGBM WMAPE:           {lgbm_mean*100:.1f}%
    LightGBM + promo WMAPE:   {promo_mean*100:.1f}%
    Delta (Naive - LightGBM): {delta_bp:+.1f} pp  {'✓ ML wins' if lgbm_mean < naive_mean else '✗ ML loses'}
    Delta (promo impact):     {promo_delta:+.1f} pp
    Stores where ML wins:     {pct_wins:.0f}%
""")
