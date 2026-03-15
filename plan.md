# Demand Cleansing Module — Implementation Plan

## Overview

New `DemandCleanser` class in `src/data/cleanser.py` that detects and corrects outliers, imputes stockouts, and supports exclusion of user-specified periods. Integrated into the pipeline via `SeriesBuilder.build()` — runs after gap-filling, before short-series dropping. All behavior is config-driven via a new `CleansingConfig` dataclass nested under `DataQualityConfig`.

## 1. Configuration — `src/config/schema.py`

Add a new `CleansingConfig` dataclass and nest it under `DataQualityConfig`:

```python
@dataclass
class CleansingConfig:
    """Demand cleansing settings."""
    enabled: bool = False                    # opt-in to avoid breaking existing pipelines

    # Outlier detection
    outlier_method: str = "iqr"              # "iqr" | "zscore" | "seasonal_residual"
    iqr_multiplier: float = 1.5              # IQR fence multiplier (1.5 = standard, 3.0 = extreme)
    zscore_threshold: float = 3.0            # z-score cutoff
    outlier_action: str = "clip"             # "clip" (winsorize to fence) | "interpolate" | "flag_only"

    # Stockout imputation
    stockout_detection: bool = True          # detect consecutive-zero runs → recovery pattern
    min_zero_run: int = 2                    # minimum consecutive zeros to consider stockout
    stockout_imputation: str = "seasonal"    # "seasonal" (same-week prior year) | "interpolate" | "none"

    # Period exclusion (e.g., COVID, warehouse fire)
    exclude_periods: List[Dict[str, str]] = field(default_factory=list)
    # Each entry: {"start": "2020-03-15", "end": "2020-06-30", "action": "interpolate"|"drop"|"flag"}
    # "interpolate" replaces values in the range; "drop" removes rows; "flag" just adds a column

    # Output
    add_flag_columns: bool = True            # add _outlier_flag, _stockout_flag, _excluded_flag columns
```

Update `DataQualityConfig`:
```python
@dataclass
class DataQualityConfig:
    # ... existing fields ...
    cleansing: CleansingConfig = field(default_factory=CleansingConfig)
```

## 2. Core Module — `src/data/cleanser.py` (~200 lines)

### Class: `DemandCleanser`

```
DemandCleanser(config: CleansingConfig)
```

### Public Methods

**`cleanse(df, time_col, value_col, sid_col) -> CleansingResult`**
- Main entry point. Applies all enabled steps per-series.
- Returns a `CleansingResult` dataclass with:
  - `df`: cleaned DataFrame
  - `report`: `CleansingReport` summary

**`detect_outliers(df, time_col, value_col, sid_col) -> DataFrame`**
- Per-series outlier detection. Returns df with `_outlier_flag` column.
- IQR method: compute Q1, Q3 per series; flag values outside [Q1 - k*IQR, Q3 + k*IQR]
- Z-score method: compute rolling or global z-score per series; flag |z| > threshold
- Seasonal residual method: decompose with STL (via statsforecast), flag residuals > threshold

**`detect_stockouts(df, time_col, value_col, sid_col) -> DataFrame`**
- Identifies consecutive-zero runs of length >= `min_zero_run` that are followed by a recovery (non-zero value).
- Distinguishes true zeros (end-of-life, seasonal closure) from stockouts (zero run → bounce back).
- Returns df with `_stockout_flag` column.

**`apply_corrections(df, time_col, value_col, sid_col) -> DataFrame`**
- Applies configured actions to flagged rows:
  - Outliers: clip to fence bounds, or interpolate, or leave as-is
  - Stockouts: impute from same-week-prior-year, or linear interpolate
  - Excluded periods: interpolate, drop, or flag-only

**`generate_report(df_before, df_after, sid_col) -> CleansingReport`**
- Summary statistics: outlier count/%, stockout periods detected, rows modified, per-series breakdown

### Dataclasses

```python
@dataclass
class CleansingReport:
    total_series: int
    series_with_outliers: int
    total_outliers: int
    outlier_pct: float
    series_with_stockouts: int
    total_stockout_periods: int
    total_stockout_weeks: int
    excluded_period_weeks: int
    rows_modified: int
    per_series: pl.DataFrame  # [series_id, outliers, stockouts, modified_rows]
```

```python
@dataclass
class CleansingResult:
    df: pl.DataFrame
    report: CleansingReport
```

### Implementation Details

**Outlier detection (IQR)**:
```python
# Per series: compute Q1/Q3, then flag
stats = df.group_by(sid_col).agg([
    pl.col(value_col).quantile(0.25).alias("q1"),
    pl.col(value_col).quantile(0.75).alias("q3"),
])
stats = stats.with_columns([
    (pl.col("q3") - pl.col("q1")).alias("iqr"),
])
stats = stats.with_columns([
    (pl.col("q1") - multiplier * pl.col("iqr")).alias("lower"),
    (pl.col("q3") + multiplier * pl.col("iqr")).alias("upper"),
])
# Join back, flag where value < lower or value > upper
```

**Stockout detection**:
```python
# Per series, sorted by time:
# 1. Identify zero runs (consecutive weeks with value == 0)
# 2. For each run, check if followed by non-zero within N weeks
# 3. If yes → stockout. If no → true demand pattern (end-of-life, seasonal)
```

**Seasonal imputation for stockouts**:
```python
# For each stockout week, look back 52 weeks for same-week value
# If not available, use average of ±1 week neighbors from prior year
# If still not available, fall back to linear interpolation
```

**Period exclusion**:
```python
# Parse date ranges from config
# For each range, apply action:
#   "interpolate": replace values with linear interpolation from boundaries
#   "drop": remove rows (let gap-filling handle later if needed)
#   "flag": add _excluded_flag=True, don't modify values
```

## 3. Pipeline Integration — `src/series/builder.py`

Add cleansing step between gap-filling (step 3) and short-series dropping (step 3a):

```python
# Step 3 (existing): Fill missing weeks
if dq.fill_gaps:
    df = self._fill_gaps(df, time_col, sid_col, value_col, dq.fill_value)

# NEW Step 3-cleanse: Demand cleansing
if dq.cleansing.enabled:
    from ..data.cleanser import DemandCleanser
    cleanser = DemandCleanser(dq.cleansing)
    result = cleanser.cleanse(df, time_col, value_col, sid_col)
    df = result.df
    self._last_cleansing_report = result.report
    logger.info(
        "Cleansing: %d outliers in %d series, %d stockout periods",
        result.report.total_outliers,
        result.report.series_with_outliers,
        result.report.total_stockout_periods,
    )

# Step 3a (existing): Drop short series
```

Expose `_last_cleansing_report` so pipelines can include it in output.

## 4. Tests — `tests/test_demand_cleansing.py` (~250 lines)

### Test Helpers

```python
def _make_series_with_outliers(n_weeks=104, seed=42) -> pl.DataFrame:
    """Normal demand ~100 with 5 injected outliers at 500+."""

def _make_series_with_stockout(n_weeks=104) -> pl.DataFrame:
    """Steady demand with a 4-week zero run at weeks 30-33, then recovery."""

def _make_series_with_excluded_period() -> pl.DataFrame:
    """Normal demand with a COVID-like spike at weeks 50-60."""
```

### Test Classes

**TestOutlierDetection** (~8 tests):
- `test_iqr_flags_extreme_values`: injected outliers are flagged
- `test_iqr_does_not_flag_normal_values`: normal variation not flagged
- `test_zscore_flags_extreme_values`: z-score method flags same outliers
- `test_clip_action_winsorizes_to_fence`: clipped values equal fence bounds
- `test_interpolate_action_fills_smoothly`: interpolated values are between neighbors
- `test_flag_only_preserves_original_values`: flag_only doesn't modify data
- `test_outlier_detection_per_series`: multi-series df, each series has independent stats
- `test_empty_dataframe_no_error`: graceful handling of empty input

**TestStockoutDetection** (~6 tests):
- `test_consecutive_zeros_with_recovery_flagged`: 4-week zero run + recovery = stockout
- `test_consecutive_zeros_at_end_not_flagged`: trailing zeros = not stockout (could be EOL)
- `test_short_zero_run_not_flagged`: 1-week zero not flagged when min_zero_run=2
- `test_seasonal_imputation_uses_prior_year`: imputed values match ±52 week lookback
- `test_interpolate_imputation_bridges_gap`: linear interpolation between boundaries
- `test_stockout_detection_multi_series`: independent detection per series

**TestPeriodExclusion** (~4 tests):
- `test_interpolate_replaces_excluded_values`: values in range are interpolated
- `test_drop_removes_excluded_rows`: rows in range are removed
- `test_flag_only_adds_column`: flag column added, values preserved
- `test_multiple_exclusion_periods`: multiple non-overlapping ranges

**TestCleansingReport** (~3 tests):
- `test_report_counts_match`: report.total_outliers matches actual flags
- `test_report_per_series_breakdown`: per-series detail is correct
- `test_no_cleansing_needed_report`: clean data → zero counts

**TestCleansingIntegration** (~3 tests):
- `test_cleanser_in_series_builder`: end-to-end with SeriesBuilder, config enabled
- `test_disabled_by_default`: cleansing.enabled=False → no changes
- `test_cleansing_before_short_series_drop`: ordering is correct

## 5. File Summary

| File | Action | ~Lines |
|------|--------|--------|
| `src/config/schema.py` | Edit: add `CleansingConfig`, update `DataQualityConfig` | +25 |
| `src/data/cleanser.py` | New file | ~200 |
| `src/series/builder.py` | Edit: add cleansing step after gap-filling | +15 |
| `tests/test_demand_cleansing.py` | New file | ~250 |

Total: ~490 lines of new/modified code.

## 6. Design Decisions

1. **Opt-in (`enabled: False`)**: Existing pipelines unaffected. Users enable via YAML config.
2. **Per-series detection**: All statistics computed per-series, not globally. A series selling 10 units/week has different outlier bounds than one selling 10,000.
3. **Runs after gap-filling**: Ensures complete weekly grid before detecting patterns. Stockout detection needs the full timeline.
4. **Runs before short-series drop**: Cleansing might change effective series length (e.g., period exclusion with "drop" action).
5. **Flag columns optional**: `add_flag_columns=True` lets downstream analysis (explainability, audit) see what was modified. Can be turned off to keep DataFrames lean.
6. **No Pandas**: All Polars, consistent with production code style.
7. **Seasonal imputation over simple interpolation**: For stockouts, seasonal patterns matter more than linear trends. Same-week-prior-year is a better imputation strategy for weekly retail data.
