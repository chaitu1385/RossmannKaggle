"""
Source Tie-Out — dual-path data integrity verification.

Verifies that data loaded into the pipeline matches the original source
by comparing profiles computed via two independent code paths (Polars
direct-read vs DuckDB SQL).

Tiers:
    Tier 1 (Structural): row count, column count, null counts — exact match
    Tier 2 (Aggregation): numeric sums (0.01%), distinct counts, date ranges

Gate decision:
    - All PASS → PROCEED
    - Any WARN → PROCEED WITH CAUTION
    - Any FAIL → HALT pipeline

Usage::

    from src.tieout import (
        read_source_direct, profile_dataframe,
        compare_profiles, overall_status,
    )

    # Path A: read raw source with Polars
    df_source = read_source_direct("data/sales.csv")
    source_profile = profile_dataframe(df_source, label="source")

    # Path B: read from DuckDB
    import duckdb
    con = duckdb.connect("pipeline.duckdb")
    df_db = con.execute("SELECT * FROM sales").pl()
    db_profile = profile_dataframe(df_db, label="duckdb")

    # Compare
    results = compare_profiles(source_profile, db_profile)
    status = overall_status(results)
    print(format_tieout_table(results))
"""

from src.tieout.tieout_helpers import (
    read_source_direct,
    profile_dataframe,
    compare_profiles,
    format_tieout_table,
    overall_status,
    check_null_concentration,
    check_outliers,
    run_full_tieout,
)

__all__ = [
    "read_source_direct",
    "profile_dataframe",
    "compare_profiles",
    "format_tieout_table",
    "overall_status",
    "check_null_concentration",
    "check_outliers",
    "run_full_tieout",
]
