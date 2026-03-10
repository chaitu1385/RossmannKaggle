"""
Synthetic product master generator for development and testing.

Produces a realistic ~120-SKU catalogue that exercises all four discovery
scenarios:
  ① Both attribute matching AND naming parsing fire  → expect High confidence
  ② Only attribute matching fires                    → expect Medium confidence
  ③ Only naming parsing fires                        → expect Medium confidence
  ④ Neither fires (true negatives / false-positive traps)
  ⑤ 1-to-many split
  ⑥ Regional variation (transition in one country only)
"""

from datetime import date, timedelta
from typing import Dict, List

import polars as pl


# ── Seed definitions ──────────────────────────────────────────────────────────

_BASE_DATE = date(2022, 1, 1)


def _d(offset_weeks: int) -> date:
    return _BASE_DATE + timedelta(weeks=offset_weeks)


def _sku(
    sku_id: str,
    description: str,
    family: str,
    category: str,
    form_factor: str,
    price_tier: str,
    countries: List[str],
    segment: str,
    launch_weeks: int,
    eol_weeks: int | None,
    status: str,
) -> Dict:
    return {
        "sku_id": sku_id,
        "sku_description": description,
        "product_family": family,
        "product_category": category,
        "form_factor": form_factor,
        "price_tier": price_tier,
        "country": countries,
        "segment": segment,
        "launch_date": _d(launch_weeks),
        "eol_date": _d(eol_weeks) if eol_weeks is not None else None,
        "status": status,
    }


# ── SKU catalogue ─────────────────────────────────────────────────────────────

_SKUS: List[Dict] = [
    # ════════════════════════════════════════════════════════════════════════
    # GROUP A — WH-100 series (On-Ear, Budget, Retail)
    # Scenario ①: attribute + naming both match
    # Generation chain: WH-100 → WH-100-MkII → WH-100-MkIII
    # ════════════════════════════════════════════════════════════════════════
    _sku("WH-100", "Budget On-Ear Headphones",
         "On-Ear", "Headphones", "On-Ear", "Budget",
         ["USA", "GBR", "DEU"], "Retail", 0, 52, "Discontinued"),
    _sku("WH-100-MkII", "Budget On-Ear Headphones MkII",
         "On-Ear", "Headphones", "On-Ear", "Budget",
         ["USA", "GBR", "DEU"], "Retail", 48, 104, "Discontinued"),
    _sku("WH-100-MkIII", "Budget On-Ear Headphones MkIII",
         "On-Ear", "Headphones", "On-Ear", "Budget",
         ["USA", "GBR", "DEU"], "Retail", 100, None, "Active"),

    # ════════════════════════════════════════════════════════════════════════
    # GROUP B — XB-500 series (Over-Ear, Mid-Range, Retail)
    # Scenario ①: attribute + naming both match (year-suffix pattern)
    # ════════════════════════════════════════════════════════════════════════
    _sku("XB-500-2022", "XB-500 Over-Ear 2022",
         "Over-Ear", "Headphones", "Over-Ear", "Mid-Range",
         ["USA", "DEU", "JPN"], "Retail", 4, 60, "Discontinued"),
    _sku("XB-500-2023", "XB-500 Over-Ear 2023",
         "Over-Ear", "Headphones", "Over-Ear", "Mid-Range",
         ["USA", "DEU", "JPN"], "Retail", 56, 112, "Discontinued"),
    _sku("XB-500-2024", "XB-500 Over-Ear 2024",
         "Over-Ear", "Headphones", "Over-Ear", "Mid-Range",
         ["USA", "DEU", "JPN"], "Retail", 108, None, "Active"),

    # ════════════════════════════════════════════════════════════════════════
    # GROUP C — SoundMax Pro series (True-Wireless, Premium, Retail)
    # Scenario ①: trailing-number naming + attribute match
    # ════════════════════════════════════════════════════════════════════════
    _sku("SM-PRO-1", "SoundMax Pro",
         "True-Wireless", "Earbuds", "True-Wireless", "Premium",
         ["USA", "GBR", "AUS"], "Retail", 8, 68, "Discontinued"),
    _sku("SM-PRO-2", "SoundMax Pro 2",
         "True-Wireless", "Earbuds", "True-Wireless", "Premium",
         ["USA", "GBR", "AUS"], "Retail", 64, 124, "Declining"),
    _sku("SM-PRO-3", "SoundMax Pro 3",
         "True-Wireless", "Earbuds", "True-Wireless", "Premium",
         ["USA", "GBR", "AUS"], "Retail", 120, None, "Active"),

    # ════════════════════════════════════════════════════════════════════════
    # GROUP D — BT-Air series (True-Wireless, Budget, Commercial)
    # Scenario ①: Gen-marker naming + attribute match
    # ════════════════════════════════════════════════════════════════════════
    _sku("BT-AIR-G1", "BT-Air Gen1",
         "True-Wireless", "Earbuds", "True-Wireless", "Budget",
         ["USA", "GBR"], "Commercial", 2, 58, "Discontinued"),
    _sku("BT-AIR-G2", "BT-Air Gen2",
         "True-Wireless", "Earbuds", "True-Wireless", "Budget",
         ["USA", "GBR"], "Commercial", 54, None, "Active"),

    # ════════════════════════════════════════════════════════════════════════
    # GROUP E — NoisePro series (Over-Ear, Flagship, Retail)
    # Scenario ①: "+Plus" variant naming + attribute match
    # ════════════════════════════════════════════════════════════════════════
    _sku("NP-100", "NoisePro",
         "Over-Ear", "Headphones", "Over-Ear", "Flagship",
         ["USA", "GBR", "DEU", "JPN", "FRA"], "Retail", 12, 74, "Discontinued"),
    _sku("NP-100-PLUS", "NoisePro Plus",
         "Over-Ear", "Headphones", "Over-Ear", "Flagship",
         ["USA", "GBR", "DEU", "JPN", "FRA"], "Retail", 70, None, "Active"),

    # ════════════════════════════════════════════════════════════════════════
    # GROUP F — PocketBass speaker (Portable, Mid-Range, Retail)
    # Scenario ①: v-number naming + attribute match
    # ════════════════════════════════════════════════════════════════════════
    _sku("PB-V1", "PocketBass v1",
         "Portable-Speaker", "Speakers", "Portable", "Mid-Range",
         ["USA", "GBR"], "Retail", 6, 62, "Discontinued"),
    _sku("PB-V2", "PocketBass v2",
         "Portable-Speaker", "Speakers", "Portable", "Mid-Range",
         ["USA", "GBR"], "Retail", 58, None, "Active"),

    # ════════════════════════════════════════════════════════════════════════
    # GROUP G — EarComfort series (On-Ear, Mid-Range, Stores)
    # Scenario ②: attribute match ONLY (completely different product names)
    # ════════════════════════════════════════════════════════════════════════
    _sku("EC-200", "EarComfort Studio",
         "On-Ear", "Headphones", "On-Ear", "Mid-Range",
         ["USA", "DEU"], "Stores", 10, 70, "Discontinued"),
    _sku("HA-300", "HeadAmp Studio",        # different name, same attributes
         "On-Ear", "Headphones", "On-Ear", "Mid-Range",
         ["USA", "DEU"], "Stores", 66, None, "Active"),

    # ════════════════════════════════════════════════════════════════════════
    # GROUP H — CoreBuds series (True-Wireless, Budget, Retail)
    # Scenario ③: naming match ONLY (slight price-tier shift: Budget → Mid-Range)
    # ════════════════════════════════════════════════════════════════════════
    _sku("CB-100", "CoreBuds",
         "True-Wireless", "Earbuds", "True-Wireless", "Budget",
         ["USA"], "Retail", 14, 76, "Discontinued"),
    _sku("CB-100-V2", "CoreBuds v2",        # same base name, different price tier
         "True-Wireless", "Earbuds", "True-Wireless", "Mid-Range",
         ["USA"], "Retail", 72, None, "Active"),

    # ════════════════════════════════════════════════════════════════════════
    # GROUP I — EarPro 1-to-many split
    # Scenario ⑤: one old SKU replaced by two new SKUs
    # ════════════════════════════════════════════════════════════════════════
    _sku("EP-100", "EarPro",
         "True-Wireless", "Earbuds", "True-Wireless", "Mid-Range",
         ["USA", "GBR", "AUS"], "Retail", 16, 78, "Discontinued"),
    _sku("EP-200-STD", "EarPro Standard",
         "True-Wireless", "Earbuds", "True-Wireless", "Mid-Range",
         ["USA", "GBR", "AUS"], "Retail", 74, None, "Active"),
    _sku("EP-200-SPT", "EarPro Sport",
         "True-Wireless", "Earbuds", "True-Wireless", "Mid-Range",
         ["USA", "GBR", "AUS"], "Retail", 74, None, "Active"),

    # ════════════════════════════════════════════════════════════════════════
    # GROUP J — Regional variation
    # Scenario ⑥: WH-300 is discontinued in USA but stays active in DEU
    # ════════════════════════════════════════════════════════════════════════
    _sku("WH-300-USA", "WH-300 Headphones (US)",
         "On-Ear", "Headphones", "On-Ear", "Budget",
         ["USA"], "Retail", 20, 80, "Discontinued"),
    _sku("WH-300-DEU", "WH-300 Headphones (DE)",
         "On-Ear", "Headphones", "On-Ear", "Budget",
         ["DEU"], "Retail", 20, None, "Active"),       # still active in DEU
    _sku("WH-300-MkII", "WH-300 Headphones MkII",
         "On-Ear", "Headphones", "On-Ear", "Budget",
         ["USA"], "Retail", 76, None, "Active"),

    # ════════════════════════════════════════════════════════════════════════
    # GROUP K — True negatives / false-positive traps
    # Different product families → neither method should match these
    # ════════════════════════════════════════════════════════════════════════
    _sku("ALPHA-100", "Alpha On-Ear",
         "On-Ear", "Headphones", "On-Ear", "Budget",
         ["USA"], "Retail", 0, 52, "Discontinued"),
    _sku("BETA-200", "Beta Portable Speaker",    # completely different family
         "Portable-Speaker", "Speakers", "Portable", "Budget",
         ["USA"], "Retail", 56, None, "Active"),

    # Coincidentally similar names but different everything
    _sku("PRO-X1", "ProMax Earbuds",
         "True-Wireless", "Earbuds", "True-Wireless", "Premium",
         ["GBR"], "Commercial", 18, None, "Active"),
    _sku("PRO-X2", "ProMax Speaker",           # "ProMax" base, but different family
         "Portable-Speaker", "Speakers", "Portable", "Premium",
         ["GBR"], "Commercial", 30, None, "Active"),

    # ════════════════════════════════════════════════════════════════════════
    # GROUP L — Standalone SKUs (no predecessor / no successor yet)
    # ════════════════════════════════════════════════════════════════════════
    _sku("HM-400", "HiFi Monitor Pro",
         "Over-Ear", "Headphones", "Over-Ear", "Flagship",
         ["USA", "JPN"], "Commercial", 22, None, "Active"),
    _sku("TW-LITE", "TrueWireless Lite",
         "True-Wireless", "Earbuds", "True-Wireless", "Budget",
         ["USA", "GBR", "AUS"], "Retail", 26, None, "Active"),
    _sku("BS-HOME-1", "BaseSpeaker Home",
         "Home-Speaker", "Speakers", "Home", "Mid-Range",
         ["USA", "DEU", "FRA"], "Retail", 30, None, "Active"),
    _sku("BS-HOME-2", "BaseSpeaker Home Plus",
         "Home-Speaker", "Speakers", "Home", "Mid-Range",
         ["USA", "DEU", "FRA"], "Retail", 88, None, "Active"),    # too far gap
    _sku("NK-SPORT", "NeckBand Sport",
         "In-Ear", "Earbuds", "In-Ear", "Budget",
         ["USA"], "Retail", 34, None, "Active"),
    _sku("NK-SPORT-2", "NeckBand Sport 2",
         "In-Ear", "Earbuds", "In-Ear", "Budget",
         ["USA"], "Retail", 90, None, "Planned"),                  # future
    _sku("DS-600", "DeskStation 600",
         "Home-Speaker", "Speakers", "Home", "Flagship",
         ["USA", "GBR"], "Commercial", 40, None, "Active"),
    _sku("DS-600-EL", "DeskStation 600 Elite",
         "Home-Speaker", "Speakers", "Home", "Flagship",
         ["USA", "GBR"], "Commercial", 95, None, "Planned"),
    _sku("OE-FOLD", "FoldAway Over-Ear",
         "Over-Ear", "Headphones", "Over-Ear", "Budget",
         ["USA", "DEU"], "Retail", 44, 100, "Declining"),
    _sku("OE-FOLD-2", "FoldAway Over-Ear 2",
         "Over-Ear", "Headphones", "Over-Ear", "Budget",
         ["USA", "DEU"], "Retail", 96, None, "Planned"),
]


# ── Public API ────────────────────────────────────────────────────────────────

def generate_product_master() -> pl.DataFrame:
    """
    Return a Polars DataFrame representing the synthetic product master.

    Schema matches ``PRODUCT_MASTER_SCHEMA`` exactly.  The ``country``
    column is already a ``List[Utf8]``; no further parsing is required.
    """
    rows = []
    for s in _SKUS:
        rows.append({
            "sku_id": s["sku_id"],
            "sku_description": s["sku_description"],
            "product_family": s["product_family"],
            "product_category": s["product_category"],
            "form_factor": s["form_factor"],
            "price_tier": s["price_tier"],
            "country": s["country"],
            "segment": s["segment"],
            "launch_date": s["launch_date"],
            "eol_date": s["eol_date"],
            "status": s["status"],
        })

    return pl.DataFrame(
        rows,
        schema={
            "sku_id": pl.Utf8,
            "sku_description": pl.Utf8,
            "product_family": pl.Utf8,
            "product_category": pl.Utf8,
            "form_factor": pl.Utf8,
            "price_tier": pl.Utf8,
            "country": pl.List(pl.Utf8),
            "segment": pl.Utf8,
            "launch_date": pl.Date,
            "eol_date": pl.Date,
            "status": pl.Utf8,
        },
    )
