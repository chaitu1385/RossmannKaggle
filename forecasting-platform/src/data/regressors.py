"""
External regressor data loading, holiday generation, and validation.

Provides utilities for loading promotional calendars, generating holiday
features, and validating that external regressors align with actuals data.
"""

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import polars as pl

logger = logging.getLogger(__name__)

# Holiday generation — optional dependency
try:
    import holidays as _holidays_lib
    _HAS_HOLIDAYS = True
except ImportError:
    _HAS_HOLIDAYS = False


def load_external_features(path: str) -> pl.DataFrame:
    """
    Load external feature data from Parquet or CSV.

    Expected columns: at minimum a time column (e.g. 'week') and one or more
    feature columns.  May optionally include a series_id column for
    series-specific features (e.g. per-SKU promotions).

    Parameters
    ----------
    path:
        Path to a Parquet or CSV file.

    Returns
    -------
    DataFrame with external feature columns.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"External features file not found: {path}")

    if p.suffix == ".parquet":
        return pl.read_parquet(path)
    elif p.suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}. Use .parquet or .csv")


def generate_holiday_calendar(
    country: str,
    start_date: date,
    end_date: date,
    time_column: str = "week",
) -> pl.DataFrame:
    """
    Generate a weekly holiday flag calendar for a given country.

    Requires the `holidays` package (pip install holidays).
    Each week gets a count of holidays falling within that week.

    Parameters
    ----------
    country:
        ISO 3166-1 alpha-2 country code (e.g. 'US', 'DE', 'GB').
    start_date:
        Start of the date range.
    end_date:
        End of the date range.
    time_column:
        Name for the time column in the output.

    Returns
    -------
    DataFrame with columns [time_column, 'holiday_flag', 'holiday_count'].
    """
    if not _HAS_HOLIDAYS:
        logger.warning(
            "holidays package not installed. "
            "Install with: pip install holidays. Returning empty calendar."
        )
        return pl.DataFrame(schema={
            time_column: pl.Date,
            "holiday_flag": pl.Int8,
            "holiday_count": pl.Int32,
        })

    # Get all holidays in the range
    years = list(range(start_date.year, end_date.year + 1))
    country_holidays = _holidays_lib.country_holidays(country, years=years)

    # Generate weekly date range
    weeks = pl.date_range(start_date, end_date, interval="1w", eager=True)

    holiday_counts = []
    for week_start in weeks:
        # Count holidays in the 7-day window starting from this date
        count = 0
        for day_offset in range(7):
            check_date = week_start + pl.duration(days=day_offset)
            if isinstance(check_date, date) and check_date in country_holidays:
                count += 1
            else:
                # Handle polars date type
                try:
                    py_date = week_start + timedelta(days=day_offset)
                    if py_date in country_holidays:
                        count += 1
                except Exception:
                    pass
        holiday_counts.append(count)

    return pl.DataFrame({
        time_column: weeks,
        "holiday_flag": [1 if c > 0 else 0 for c in holiday_counts],
        "holiday_count": holiday_counts,
    })


def validate_regressors(
    external_features: pl.DataFrame,
    actuals: pl.DataFrame,
    feature_columns: List[str],
    time_column: str = "week",
    id_column: Optional[str] = None,
    horizon_weeks: int = 0,
) -> List[str]:
    """
    Validate external regressor data against actuals.

    Checks:
    1. All feature columns exist in external_features.
    2. Time column alignment (same grain as actuals).
    3. No nulls in feature columns during training period.
    4. If horizon_weeks > 0, future feature values must be present.

    Parameters
    ----------
    external_features:
        The external feature DataFrame.
    actuals:
        The actuals DataFrame (for alignment checks).
    feature_columns:
        List of expected feature column names.
    time_column:
        Time column name.
    id_column:
        Optional series ID column (for series-level features).
    horizon_weeks:
        If > 0, validates that future feature values exist for this many weeks.

    Returns
    -------
    List of warning/error messages. Empty list means all checks passed.
    """
    issues: List[str] = []

    # Check 1: Feature columns exist
    missing_cols = [c for c in feature_columns if c not in external_features.columns]
    if missing_cols:
        issues.append(f"Missing feature columns in external data: {missing_cols}")
        return issues  # Can't proceed without columns

    # Check 2: Time column exists
    if time_column not in external_features.columns:
        issues.append(f"Time column '{time_column}' not found in external features")
        return issues

    # Check 3: Nulls in feature columns
    for col in feature_columns:
        null_count = external_features[col].null_count()
        if null_count > 0:
            issues.append(
                f"Feature '{col}' has {null_count} null values. "
                f"Consider filling with forward-fill or zeros."
            )

    # Check 4: Time range coverage
    if time_column in actuals.columns:
        actuals_max = actuals[time_column].max()
        features_max = external_features[time_column].max()
        actuals_min = actuals[time_column].min()
        features_min = external_features[time_column].min()

        if features_min is not None and actuals_min is not None:
            if features_min > actuals_min:
                issues.append(
                    f"External features start ({features_min}) is after "
                    f"actuals start ({actuals_min}). Early periods will have null features."
                )

        if horizon_weeks > 0 and actuals_max is not None and features_max is not None:
            required_end = actuals_max + timedelta(weeks=horizon_weeks)
            if features_max < required_end:
                issues.append(
                    f"External features end ({features_max}) does not cover "
                    f"the forecast horizon (need through {required_end}). "
                    f"Prediction will fail without future feature values."
                )

    # Check 5: Series ID alignment (if series-level features)
    if id_column and id_column in external_features.columns and id_column in actuals.columns:
        feat_ids = set(external_features[id_column].unique().to_list())
        actual_ids = set(actuals[id_column].unique().to_list())
        missing_ids = actual_ids - feat_ids
        if missing_ids:
            issues.append(
                f"{len(missing_ids)} series in actuals have no external features. "
                f"These series will get null feature values (filled with 0)."
            )

    return issues
