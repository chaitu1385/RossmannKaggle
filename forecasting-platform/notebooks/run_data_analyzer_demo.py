#!/usr/bin/env python
"""
DataAnalyzer live demo — Rossmann dataset.

Runs the full analysis pipeline (schema detection, hierarchy detection,
forecastability assessment, hypothesis generation, config recommendation)
and optionally queries Claude for interpretation.

Usage
-----
    python notebooks/run_data_analyzer_demo.py
    python notebooks/run_data_analyzer_demo.py --llm   # enable Claude interpretation
"""

import argparse
import os
import sys
import time
from dataclasses import asdict

import yaml

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__) + "/..")

import polars as pl


def load_rossmann():
    """Load the Rossmann sample from the Rossmann eval notebook."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    candidates = [
        os.path.join(data_path, "rossmann_weekly.parquet"),
        os.path.join(data_path, "rossmann_weekly.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            if path.endswith(".parquet"):
                return pl.read_parquet(path)
            return pl.read_csv(path, try_parse_dates=True)

    # Generate from Rossmann eval script's inline data generator
    print("Rossmann data not found on disk — generating synthetic retail dataset...")
    return _generate_synthetic_retail()


def _generate_synthetic_retail():
    """Generate a synthetic retail dataset for demo purposes."""
    import numpy as np
    from datetime import date, timedelta

    rng = np.random.RandomState(42)
    rows = []
    base = date(2020, 1, 6)
    n_weeks = 104

    stores = {f"store_{i}": rng.choice(["North", "South", "East", "West"]) for i in range(20)}
    categories = ["Food", "Electronics", "Clothing"]

    for store_id, region in stores.items():
        for cat in categories:
            base_demand = rng.uniform(50, 200)
            for w in range(n_weeks):
                seasonal = 30 * np.sin(2 * np.pi * w / 52)
                trend = 0.2 * w
                noise = rng.normal(0, 10)
                qty = max(0, base_demand + seasonal + trend + noise)
                rows.append({
                    "week": base + timedelta(weeks=w),
                    "store_id": store_id,
                    "region": region,
                    "category": cat,
                    "quantity": round(qty, 2),
                    "promo_intensity": round(rng.uniform(0, 1), 2),
                })

    return pl.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="DataAnalyzer demo on Rossmann data")
    parser.add_argument("--llm", action="store_true", help="Enable Claude interpretation")
    args = parser.parse_args()

    print("=" * 80)
    print("DATA ANALYZER DEMO — Retail Forecasting Dataset")
    print("=" * 80)

    # Load data
    df = load_rossmann()
    print(f"\nDataset: {df.height} rows, {df.width} columns")
    print(f"Columns: {df.columns}")
    print(f"Sample:\n{df.head(3)}\n")

    # Run analysis
    from src.analytics.analyzer import DataAnalyzer

    t0 = time.time()
    analyzer = DataAnalyzer(lob_name="rossmann_demo")
    report = analyzer.analyze(df)
    elapsed = time.time() - t0

    # Schema Detection
    print("=" * 80)
    print("SCHEMA DETECTION")
    print("=" * 80)
    s = report.schema
    print(f"  Time column:    {s.time_column}")
    print(f"  Target column:  {s.target_column}")
    print(f"  ID columns:     {s.id_columns}")
    print(f"  Dimensions:     {s.dimension_columns}")
    print(f"  Numeric cols:   {s.numeric_columns}")
    print(f"  Series count:   {s.n_series}")
    print(f"  Date range:     {s.date_range[0]} → {s.date_range[1]}")
    print(f"  Frequency:      {s.frequency_guess}")
    print(f"  Confidence:     {s.confidence:.2f}")

    # Hierarchy Detection
    print(f"\n{'=' * 80}")
    print("HIERARCHY DETECTION")
    print("=" * 80)
    for h in report.hierarchy.hierarchies:
        print(f"  {h.name}: {' → '.join(h.levels)} (leaf: {h.id_column}, fixed: {h.fixed})")
    if report.hierarchy.reasoning:
        print("\n  Reasoning:")
        for r in report.hierarchy.reasoning:
            print(f"    • {r}")

    # Forecastability
    print(f"\n{'=' * 80}")
    print("FORECASTABILITY ASSESSMENT")
    print("=" * 80)
    fc = report.forecastability
    print(f"  Overall score:   {fc.overall_score:.3f}")
    print(f"  Distribution:    {fc.score_distribution}")
    print(f"  Demand classes:  {fc.demand_class_distribution}")
    if fc.per_series is not None:
        print(f"\n  Per-series signals (first 10):")
        print(fc.per_series.head(10))

    # Hypotheses
    print(f"\n{'=' * 80}")
    print("HYPOTHESES")
    print("=" * 80)
    for i, h in enumerate(report.hypotheses, 1):
        print(f"  {i}. {h}")

    # Config Recommendation
    print(f"\n{'=' * 80}")
    print("RECOMMENDED PLATFORM CONFIG")
    print("=" * 80)
    config = report.recommended_config
    print(f"\n  LOB:             {config.lob}")
    print(f"  Forecasters:     {config.forecast.forecasters}")
    print(f"  Intermittent:    {config.forecast.intermittent_forecasters}")
    print(f"  Horizon:         {config.forecast.horizon_weeks} weeks")
    print(f"  Quantiles:       {config.forecast.quantiles}")
    print(f"  Backtest folds:  {config.backtest.n_folds}")
    print(f"  Champion gran:   {config.backtest.champion_granularity}")
    print(f"  Reconciliation:  {config.reconciliation.method}")
    print(f"  Cleansing:       {config.data_quality.cleansing.enabled}")
    print(f"  Metrics:         {config.metrics}")

    print(f"\n  Config reasoning:")
    for r in report.config_reasoning:
        print(f"    • {r}")

    # YAML export
    config_yaml = yaml.dump(asdict(config), default_flow_style=False, sort_keys=False)
    print(f"\n{'=' * 80}")
    print("GENERATED YAML CONFIG")
    print("=" * 80)
    print(config_yaml)

    # LLM Interpretation
    if args.llm:
        print(f"\n{'=' * 80}")
        print("CLAUDE INTERPRETATION (LLM)")
        print("=" * 80)

        from src.analytics.llm_analyzer import LLMAnalyzer

        llm = LLMAnalyzer()
        if not llm.available:
            print("  ⚠ Anthropic client not available (no API key or package)")
        else:
            t1 = time.time()
            insight = llm.interpret(report)
            llm_elapsed = time.time() - t1

            print(f"\n  --- Narrative ---")
            print(f"  {insight.narrative}")

            print(f"\n  --- Hypotheses ---")
            for h in insight.hypotheses:
                print(f"    • {h}")

            print(f"\n  --- Model Rationale ---")
            print(f"  {insight.model_rationale}")

            print(f"\n  --- Risk Factors ---")
            for r in insight.risk_factors:
                print(f"    • {r}")

            if insight.config_adjustments:
                print(f"\n  --- Config Adjustments ---")
                for a in insight.config_adjustments:
                    print(f"    • {a}")

            print(f"\n  LLM call took {llm_elapsed:.1f}s")

    print(f"\n{'=' * 80}")
    print(f"Analysis completed in {elapsed:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
