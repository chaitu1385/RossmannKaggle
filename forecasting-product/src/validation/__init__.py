"""
4-Layer Validation Framework for the Forecasting Platform.

Layers:
    1. Structural  — schema, primary keys, completeness, date range, row count
    2. Logical     — aggregation consistency, trends, monotonicity, future dates
    3. Business    — value ranges, metric relationships, temporal spikes, segments
    4. Simpson's   — paradox detection across segment dimensions

Plus confidence scoring that produces an A-F letter grade.

Quick start::

    import polars as pl
    from src.validation import (
        run_structural_checks,
        run_logical_checks,
        validate_business_rules,
        check_simpsons_paradox,
        score_confidence,
        format_confidence_badge,
        run_full_validation,
    )

    df = pl.read_csv("data.csv")

    # Run individual layers
    s = run_structural_checks(df, config={"primary_key": ["id", "date"]})
    l = run_logical_checks(df, config={"date_column": "date", "value_column": "qty"})
    b = validate_business_rules(df, config={"non_negative_columns": ["qty"]})
    p = check_simpsons_paradox(df, "region", "qty", period_col="month")

    # Score
    badge = score_confidence(structural=s, logical=l, business=b, paradox=p)
    print(format_confidence_badge(badge))

    # Or run everything at once
    result = run_full_validation(df, config={...})
"""

from src.validation.structural_validator import (
    validate_schema,
    validate_primary_key,
    validate_completeness,
    validate_date_range,
    validate_referential_integrity,
    validate_row_count,
    run_structural_checks,
)

from src.validation.logical_validator import (
    validate_aggregation_consistency,
    validate_percentages_sum,
    validate_monotonic,
    validate_trend_consistency,
    validate_ratio_bounds,
    validate_group_balance,
    validate_no_future_dates,
    validate_forecast_vs_actual_alignment,
    run_logical_checks,
)

from src.validation.business_rules import (
    validate_ranges,
    validate_metric_relationships,
    validate_temporal_consistency,
    validate_segment_coverage,
    validate_no_negative,
    validate_cardinality,
    validate_business_rules,
    get_default_rules,
)

from src.validation.simpsons_paradox import (
    check_simpsons_paradox,
    check_simpsons_multi_segment,
    weighted_vs_unweighted,
    suggest_segments_to_check,
    generate_paradox_report,
)

from src.validation.confidence_scoring import (
    score_confidence,
    format_confidence_badge,
    merge_confidence_scores,
)

from typing import Any, Dict, Optional


def run_full_validation(
    df,
    config: Optional[Dict[str, Any]] = None,
) -> dict:
    """Run all 4 layers + confidence scoring in one call.

    Args:
        df: Polars DataFrame.
        config: Merged config dict.  Keys are routed to the appropriate layer:
            - Structural: ``expected_columns``, ``primary_key``, ``min_rows``, etc.
            - Logical: ``date_column``, ``value_column``, ``group_col``, etc.
            - Business: ``range_rules``, ``non_negative_columns``, etc.
            - Paradox: ``segment_columns``, ``period_col``, ``weight_col``.

    Returns:
        dict with ``grade``, ``score``, ``badge``, ``layers``, ``confidence``.
    """
    config = config or {}

    structural = run_structural_checks(df, config)
    logical = run_logical_checks(df, config)
    business = validate_business_rules(df, config)

    # Paradox — only if segment columns provided
    paradox = None
    seg_cols = config.get("segment_columns")
    value_col = config.get("value_column")
    if seg_cols and value_col:
        paradox = generate_paradox_report(
            df,
            segment_columns=seg_cols,
            value_col=value_col,
            weight_col=config.get("weight_col"),
            period_col=config.get("period_col"),
        )

    confidence = score_confidence(
        structural=structural,
        logical=logical,
        business=business,
        paradox=paradox,
    )

    return {
        "grade": confidence["grade"],
        "score": confidence["score"],
        "badge": format_confidence_badge(confidence),
        "layers": {
            "structural": structural,
            "logical": logical,
            "business": business,
            "paradox": paradox,
        },
        "confidence": confidence,
    }


__all__ = [
    # Layer 1
    "validate_schema", "validate_primary_key", "validate_completeness",
    "validate_date_range", "validate_referential_integrity",
    "validate_row_count", "run_structural_checks",
    # Layer 2
    "validate_aggregation_consistency", "validate_percentages_sum",
    "validate_monotonic", "validate_trend_consistency",
    "validate_ratio_bounds", "validate_group_balance",
    "validate_no_future_dates", "validate_forecast_vs_actual_alignment",
    "run_logical_checks",
    # Layer 3
    "validate_ranges", "validate_metric_relationships",
    "validate_temporal_consistency", "validate_segment_coverage",
    "validate_no_negative", "validate_cardinality",
    "validate_business_rules", "get_default_rules",
    # Layer 4
    "check_simpsons_paradox", "check_simpsons_multi_segment",
    "weighted_vs_unweighted", "suggest_segments_to_check",
    "generate_paradox_report",
    # Confidence
    "score_confidence", "format_confidence_badge", "merge_confidence_scores",
    # Convenience
    "run_full_validation",
]
