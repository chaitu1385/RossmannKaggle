"""Generate a small weekly retail demo dataset for E2E UI testing.

Creates 5 weekly retail series with realistic patterns:
  - trend, seasonality, noise, promotions
  - 2 years of history (~104 weeks)
  - Varying demand levels (high, medium, low)

Output: data/demo/weekly_retail.csv
"""
import math
import random
import csv
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

SERIES = [
    {"id": "GROCERY_001", "category": "grocery", "base": 800, "trend": 2.0, "noise": 0.10},
    {"id": "DAIRY_002",   "category": "dairy",   "base": 400, "trend": 1.0, "noise": 0.12},
    {"id": "SNACKS_003",  "category": "snacks",  "base": 250, "trend": 0.5, "noise": 0.08},
    {"id": "BAKERY_004",  "category": "bakery",  "base": 150, "trend": -0.3,"noise": 0.15},
    {"id": "FROZEN_005",  "category": "frozen",  "base": 600, "trend": 1.5, "noise": 0.09},
]

N_WEEKS = 104  # 2 years
START = datetime(2022, 1, 3)  # Monday

out_dir = Path(__file__).resolve().parent.parent / "data" / "demo"
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
for s in SERIES:
    for w in range(N_WEEKS):
        dt = START + timedelta(weeks=w)
        # Trend
        value = s["base"] + s["trend"] * w
        # Annual seasonality (peak in Nov-Dec for holidays)
        week_of_year = dt.isocalendar()[1]
        seasonal = 0.15 * math.sin(2 * math.pi * (week_of_year - 10) / 52)
        value *= (1 + seasonal)
        # Holiday bump (weeks 47-52)
        if 47 <= week_of_year <= 52:
            value *= 1.25
        # Random promo (~15% of weeks)
        promo = 1 if random.random() < 0.15 else 0
        if promo:
            value *= 1.20
        # Noise
        value *= (1 + random.gauss(0, s["noise"]))
        value = max(0, round(value))
        rows.append({
            "series_id": s["id"],
            "week": dt.strftime("%Y-%m-%d"),
            "quantity": value,
            "category": s["category"],
            "promo": promo,
        })

out_file = out_dir / "weekly_retail.csv"
with open(out_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["series_id", "week", "quantity", "category", "promo"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows ({len(SERIES)} series × {N_WEEKS} weeks) to {out_file}")
