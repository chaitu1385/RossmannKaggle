"""
Rossmann expanded model zoo evaluation — benchmarks original + new Nixtla models.

Compares 7 models on the Rossmann dataset via walk-forward backtesting:
  Original: SeasonalNaive, AutoETS, AutoARIMA, LightGBM, XGBoost
  New:      AutoTheta, MSTL

Neural models (N-BEATS, NHITS, TFT) are included if neuralforecast is installed.
"""
import sys
import os
import time
import warnings
import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

# Add platform root to path
platform_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if platform_root not in sys.path:
    sys.path.insert(0, platform_root)

from src.forecasting.naive import SeasonalNaiveForecaster
from src.forecasting.statistical import (
    AutoARIMAForecaster, AutoETSForecaster,
    AutoThetaForecaster, MSTLForecaster,
)
from src.forecasting.ml import LGBMDirectForecaster, XGBoostDirectForecaster
from src.forecasting.registry import registry
from src.metrics.fva import compute_fva_cascade
from src.backtesting.cross_validator import WalkForwardCV

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

WALL_START = time.time()

# ══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING (identical to run_rossmann_eval.py)
# ══════════════════════════════════════════════════════════════════════
DATA_DIR = Path(platform_root) / "data" / "rossmann"
TRAIN_PATH = DATA_DIR / "train.csv"
STORE_PATH = DATA_DIR / "store.csv"

if not TRAIN_PATH.exists() or not STORE_PATH.exists():
    print("ERROR: Rossmann data not found at", DATA_DIR)
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

# Stratified subsample: 12 stores per StoreType
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
    ])
)

# Fill missing weeks
_min_w, _max_w = data["week"].min(), data["week"].max()
_all_weeks = pl.date_range(_min_w, _max_w, interval="1w", eager=True)
_all_ids = data.select(["series_id", "store_type"]).unique()
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
print(f"Data loaded: {n_stores} stores × {n_weeks} weeks = {data.shape[0]:,} rows")
print(f"Date range: {data['week'].min()} → {data['week'].max()}")
print(f"Store types: {sorted(data['store_type'].unique().to_list())}")
print()

# ══════════════════════════════════════════════════════════════════════
# 2. BACKTEST HELPER
# ══════════════════════════════════════════════════════════════════════
data_core = data.select(["series_id", "week", "quantity"])

HORIZON = 13
N_FOLDS = 2


def manual_backtest(series_df, forecaster_factory, horizon=HORIZON, n_folds=N_FOLDS):
    """Walk-forward backtest, returns per-series per-fold WMAPE."""
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
# 3. MODEL ZOO BENCHMARK
# ══════════════════════════════════════════════════════════════════════
print("=" * 80)
print("EXPANDED MODEL ZOO BENCHMARK: Walk-Forward Backtest")
print(f"  {n_stores} stores × {N_FOLDS} folds × {HORIZON}-week horizon")
print("=" * 80)

# Define model factories
model_factories = {
    "naive_seasonal":  lambda: SeasonalNaiveForecaster(season_length=52),
    "auto_ets":        lambda: AutoETSForecaster(),
    "auto_arima":      lambda: AutoARIMAForecaster(),
    "auto_theta":      lambda: AutoThetaForecaster(),
    "mstl":            lambda: MSTLForecaster(),
    "lgbm_direct":     lambda: LGBMDirectForecaster(num_threads=1),
    "xgboost_direct":  lambda: XGBoostDirectForecaster(),
}

# Check for neural models
try:
    import neuralforecast  # noqa: F401
    from src.forecasting.neural import NBEATSForecaster, NHITSForecaster, TFTForecaster
    model_factories["nbeats"] = lambda: NBEATSForecaster(max_steps=200, accelerator="cpu")
    model_factories["nhits"]  = lambda: NHITSForecaster(max_steps=200, accelerator="cpu")
    model_factories["tft"]    = lambda: TFTForecaster(max_steps=200, accelerator="cpu")
    print("  neuralforecast detected — including N-BEATS, NHITS, TFT")
except ImportError:
    print("  neuralforecast not installed — skipping N-BEATS, NHITS, TFT")
    print("  (pip install neuralforecast to enable)")

print(f"\nModels to evaluate: {list(model_factories.keys())}")
print()

results = {}
timings = {}

for model_name, factory in model_factories.items():
    t0 = time.time()
    print(f"  Running {model_name}...", end=" ", flush=True)
    try:
        wmape_df = manual_backtest(data_core, factory, horizon=HORIZON, n_folds=N_FOLDS)
        mean_wmape = float(wmape_df["wmape"].mean())
        results[model_name] = {"wmape_df": wmape_df, "mean_wmape": mean_wmape}
        elapsed = time.time() - t0
        timings[model_name] = elapsed
        print(f"WMAPE: {mean_wmape*100:.2f}%  ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - t0
        timings[model_name] = elapsed
        print(f"FAILED ({elapsed:.1f}s): {e}")

# ══════════════════════════════════════════════════════════════════════
# 4. LEADERBOARD
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("LEADERBOARD (sorted by WMAPE)")
print("=" * 80)
print(f"  {'Rank':<5} {'Model':<20} {'WMAPE':>8} {'Time':>8}  Notes")
print(f"  {'─'*5} {'─'*20} {'─'*8} {'─'*8}  {'─'*25}")

sorted_models = sorted(results.items(), key=lambda x: x[1]["mean_wmape"])
naive_wmape = results.get("naive_seasonal", {}).get("mean_wmape", None)

for rank, (name, r) in enumerate(sorted_models, 1):
    w = r["mean_wmape"]
    t = timings.get(name, 0)
    notes = ""
    if name == "naive_seasonal":
        notes = "(baseline)"
    elif naive_wmape is not None:
        delta = (w - naive_wmape) * 100
        notes = f"({'−' if delta < 0 else '+'}{abs(delta):.1f}pp vs naive)"
    is_new = " *NEW*" if name in ("auto_theta", "mstl", "nbeats", "nhits", "tft") else ""
    print(f"  {rank:<5} {name:<20} {w*100:>7.2f}% {t:>7.1f}s  {notes}{is_new}")

# ══════════════════════════════════════════════════════════════════════
# 5. FVA CASCADE: Naive → Best Statistical → Best ML
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("FVA CASCADE — Which layers add value?")
print("=" * 80)

# Classify models into layers
stat_models = {n: r for n, r in results.items() if n in ("auto_ets", "auto_arima", "auto_theta", "mstl")}
ml_models   = {n: r for n, r in results.items() if n in ("lgbm_direct", "xgboost_direct")}
neural_models = {n: r for n, r in results.items() if n in ("nbeats", "nhits", "tft")}

best_stat = min(stat_models.items(), key=lambda x: x[1]["mean_wmape"]) if stat_models else None
best_ml   = min(ml_models.items(), key=lambda x: x[1]["mean_wmape"]) if ml_models else None
best_neural = min(neural_models.items(), key=lambda x: x[1]["mean_wmape"]) if neural_models else None

print(f"\n  Layer champions:")
print(f"    Baseline:     naive_seasonal  ({results['naive_seasonal']['mean_wmape']*100:.2f}%)")
if best_stat:
    print(f"    Statistical:  {best_stat[0]:<16s} ({best_stat[1]['mean_wmape']*100:.2f}%)")
if best_ml:
    print(f"    ML:           {best_ml[0]:<16s} ({best_ml[1]['mean_wmape']*100:.2f}%)")
if best_neural:
    print(f"    Neural:       {best_neural[0]:<16s} ({best_neural[1]['mean_wmape']*100:.2f}%)")

# FVA deltas
if best_stat:
    stat_fva = (results["naive_seasonal"]["mean_wmape"] - best_stat[1]["mean_wmape"]) * 100
    print(f"\n  Statistical → Naive FVA: {stat_fva:+.2f} pp  {'✓ adds value' if stat_fva > 0 else '✗ no value'}")
if best_ml:
    baseline = best_stat[1]["mean_wmape"] if best_stat else results["naive_seasonal"]["mean_wmape"]
    ml_fva = (baseline - best_ml[1]["mean_wmape"]) * 100
    print(f"  ML → Statistical FVA:    {ml_fva:+.2f} pp  {'✓ adds value' if ml_fva > 0 else '✗ no value'}")
if best_neural:
    baseline_n = best_ml[1]["mean_wmape"] if best_ml else (best_stat[1]["mean_wmape"] if best_stat else results["naive_seasonal"]["mean_wmape"])
    neural_fva = (baseline_n - best_neural[1]["mean_wmape"]) * 100
    print(f"  Neural → ML FVA:         {neural_fva:+.2f} pp  {'✓ adds value' if neural_fva > 0 else '✗ no value'}")

# ══════════════════════════════════════════════════════════════════════
# 6. PER-STORE HEAD-TO-HEAD: New models vs originals
# ══════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("HEAD-TO-HEAD: New models vs originals (per-store win rate)")
print("=" * 80)

for new_model in ["auto_theta", "mstl", "nbeats", "nhits", "tft"]:
    if new_model not in results:
        continue
    new_r = results[new_model]
    new_per_store = new_r["wmape_df"].group_by("series_id").agg(
        pl.col("wmape").mean().alias("new_wmape")
    )

    for orig_model in ["naive_seasonal", "auto_ets", "lgbm_direct"]:
        if orig_model not in results:
            continue
        orig_r = results[orig_model]
        orig_per_store = orig_r["wmape_df"].group_by("series_id").agg(
            pl.col("wmape").mean().alias("orig_wmape")
        )

        comp = new_per_store.join(orig_per_store, on="series_id")
        wins = comp.filter(pl.col("new_wmape") < pl.col("orig_wmape")).shape[0]
        total = comp.shape[0]
        pct = wins / total * 100 if total > 0 else 0
        marker = "✓" if pct > 50 else "✗"
        print(f"  {new_model:16s} vs {orig_model:16s}: wins {wins}/{total} ({pct:.0f}%) {marker}")
    print()

# ══════════════════════════════════════════════════════════════════════
# 7. FULL BACKTEST PIPELINE (with expanded model zoo)
# ══════════════════════════════════════════════════════════════════════
print("=" * 80)
print("FULL BACKTEST PIPELINE — Champion selection with expanded zoo")
print("=" * 80)

import tempfile
from src.config.schema import (
    PlatformConfig, ForecastConfig, BacktestConfig, OutputConfig,
)
from src.pipeline.backtest import BacktestPipeline

tmpdir = tempfile.mkdtemp()
bt_config = PlatformConfig(
    lob="rossmann",
    output=OutputConfig(metrics_path=tmpdir),
    forecast=ForecastConfig(
        horizon_weeks=HORIZON,
        forecasters=[
            "naive_seasonal", "auto_ets", "auto_arima",
            "auto_theta", "mstl",
            "lgbm_direct", "xgboost_direct",
        ],
    ),
    backtest=BacktestConfig(
        n_folds=N_FOLDS,
        val_weeks=HORIZON,
        selection_strategy="champion",
    ),
)

pipeline = BacktestPipeline(bt_config)
bt_results = pipeline.run(data_core)

print(f"\n  Models evaluated: {bt_results['backtest_results']['model_id'].unique().to_list()}")
print(f"  Champion: {bt_results['champions']}")

leaderboard = bt_results["leaderboard"]
if not leaderboard.is_empty():
    print("\n  Pipeline Leaderboard:")
    for row in leaderboard.iter_rows(named=True):
        m = row.get("model_id", "?")
        w = row.get("wmape", 0)
        is_new = " *NEW*" if m in ("auto_theta", "mstl") else ""
        print(f"    {m:20s}  WMAPE: {w*100:.2f}%{is_new}")

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
wall_elapsed = time.time() - WALL_START
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"  Total wall time:     {wall_elapsed:.1f}s ({wall_elapsed/60:.1f} min)")
print(f"  Models evaluated:    {len(results)}")
print(f"  Dataset:             {n_stores} stores × {n_weeks} weeks")
print(f"  Backtest:            {N_FOLDS} folds × {HORIZON}-week horizon")
print()
print(f"  Overall champion:    {sorted_models[0][0]} ({sorted_models[0][1]['mean_wmape']*100:.2f}%)")
if len(sorted_models) > 1:
    print(f"  Runner-up:           {sorted_models[1][0]} ({sorted_models[1][1]['mean_wmape']*100:.2f}%)")
print()
new_in_top3 = [name for name, _ in sorted_models[:3] if name in ("auto_theta", "mstl", "nbeats", "nhits", "tft")]
if new_in_top3:
    print(f"  New models in top-3: {', '.join(new_in_top3)}")
else:
    print(f"  No new models in top-3 (new models serve horizon/data-specific niches)")
print()
print("  Per-model timings:")
for name, t in sorted(timings.items(), key=lambda x: x[1]):
    print(f"    {name:20s}  {t:.1f}s")
