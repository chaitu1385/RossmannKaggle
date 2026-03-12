"""
Hierarchy aggregation and disaggregation operations.

Supports rolling data up from leaves to any ancestor level, and splitting
data down from an aggregate level to leaves using proportional keys.
"""

from typing import Dict, List, Optional

import polars as pl

from .tree import HierarchyTree


class HierarchyAggregator:
    """
    Aggregate and disaggregate time-series data across hierarchy levels.

    All operations are Polars-native (no Python loops over rows).
    """

    def __init__(self, tree: HierarchyTree):
        self.tree = tree

    def aggregate_to(
        self,
        df: pl.DataFrame,
        target_level: str,
        value_columns: List[str],
        time_column: str = "week",
        agg: str = "sum",
    ) -> pl.DataFrame:
        """
        Roll up data from leaf level to *target_level* by summing (or
        averaging) the value columns.

        Parameters
        ----------
        df:
            Must contain the leaf-level id column (tree.config.id_column),
            the time column, and all value columns.
        target_level:
            The hierarchy level to aggregate to.
        value_columns:
            Numeric columns to aggregate.
        time_column:
            Time column to group by.
        agg:
            "sum" or "mean".
        """
        leaf_col = self.tree.config.id_column
        parent_map = self.tree.get_parent_child_map(
            target_level, self.tree.leaf_level
        )

        # Build a mapping DataFrame: leaf_key → target_key
        mapping_rows = []
        for parent_key, child_keys in parent_map.items():
            for ck in child_keys:
                mapping_rows.append({leaf_col: ck, f"_agg_{target_level}": parent_key})

        if not mapping_rows:
            return df.head(0)

        mapping_df = pl.DataFrame(mapping_rows)

        joined = df.join(mapping_df, on=leaf_col, how="inner")

        agg_col = f"_agg_{target_level}"
        group_cols = [agg_col, time_column]

        if agg == "sum":
            agg_exprs = [pl.col(c).sum().alias(c) for c in value_columns]
        elif agg == "mean":
            agg_exprs = [pl.col(c).mean().alias(c) for c in value_columns]
        else:
            raise ValueError(f"Unsupported agg={agg!r}, use 'sum' or 'mean'")

        result = (
            joined.group_by(group_cols)
            .agg(agg_exprs)
            .rename({agg_col: target_level})
        )
        return result

    def disaggregate_to(
        self,
        df: pl.DataFrame,
        source_level: str,
        target_level: str,
        value_columns: List[str],
        proportions: Optional[pl.DataFrame] = None,
        time_column: str = "week",
    ) -> pl.DataFrame:
        """
        Split data from *source_level* down to *target_level* using
        proportional keys.

        Parameters
        ----------
        df:
            Data at *source_level*. Must contain a column named
            *source_level* with the node keys.
        source_level, target_level:
            Hierarchy levels.  target must be below source.
        value_columns:
            Columns to multiply by proportions.
        proportions:
            DataFrame with columns [source_level, target_level, "proportion"].
            If None, equal-split proportions are used.
        time_column:
            Time column to preserve.
        """
        parent_map = self.tree.get_parent_child_map(source_level, target_level)

        if proportions is None:
            # Equal split
            prop_rows = []
            for parent_key, child_keys in parent_map.items():
                n = len(child_keys)
                for ck in child_keys:
                    prop_rows.append({
                        source_level: parent_key,
                        target_level: ck,
                        "proportion": 1.0 / n,
                    })
            proportions = pl.DataFrame(prop_rows)

        joined = df.join(proportions, on=source_level, how="inner")

        for vc in value_columns:
            joined = joined.with_columns(
                (pl.col(vc) * pl.col("proportion")).alias(vc)
            )

        drop_cols = [source_level, "proportion"]
        keep_cols = [c for c in joined.columns if c not in drop_cols]
        return joined.select(keep_cols)

    def compute_historical_proportions(
        self,
        df: pl.DataFrame,
        source_level: str,
        target_level: str,
        value_column: str,
        time_column: str = "week",
        n_recent_weeks: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Compute disaggregation proportions from historical data.

        For each (source_level, target_level) pair, proportion = target's
        share of source's total over the given time window.

        Returns DataFrame with [source_level, target_level, proportion].
        """
        leaf_col = self.tree.config.id_column
        parent_map = self.tree.get_parent_child_map(source_level, target_level)

        # Build leaf → target, leaf → source mappings
        leaf_to_target = {}
        leaf_to_source = {}
        for parent_key, child_keys in parent_map.items():
            for ck in child_keys:
                leaf_to_source[ck] = parent_key
                # If target IS the leaf level, use the child key directly
                if target_level == self.tree.leaf_level:
                    leaf_to_target[ck] = ck
                else:
                    # Get target-level ancestor of this leaf
                    target_map = self.tree.get_parent_child_map(
                        target_level, self.tree.leaf_level
                    )
                    for tk, lks in target_map.items():
                        if ck in lks:
                            leaf_to_target[ck] = tk
                            break

        if not leaf_to_target:
            return pl.DataFrame(
                schema={source_level: pl.Utf8, target_level: pl.Utf8, "proportion": pl.Float64}
            )

        # Apply optional time filter
        work = df
        if n_recent_weeks is not None:
            max_week = work[time_column].max()
            min_week = max_week - pl.duration(weeks=n_recent_weeks)
            work = work.filter(pl.col(time_column) >= min_week)

        # Map each leaf to source and target
        source_map_df = pl.DataFrame({
            leaf_col: list(leaf_to_source.keys()),
            f"_src": list(leaf_to_source.values()),
        })
        target_map_df = pl.DataFrame({
            leaf_col: list(leaf_to_target.keys()),
            f"_tgt": list(leaf_to_target.values()),
        })

        enriched = (
            work.join(source_map_df, on=leaf_col, how="inner")
            .join(target_map_df, on=leaf_col, how="inner")
        )

        # Aggregate to (source, target) level
        target_totals = (
            enriched.group_by(["_src", "_tgt"])
            .agg(pl.col(value_column).sum().alias("_target_total"))
        )
        source_totals = (
            enriched.group_by("_src")
            .agg(pl.col(value_column).sum().alias("_source_total"))
        )

        props = (
            target_totals.join(source_totals, on="_src", how="left")
            .with_columns(
                (pl.col("_target_total") / pl.col("_source_total"))
                .alias("proportion")
            )
            .rename({"_src": source_level, "_tgt": target_level})
            .select([source_level, target_level, "proportion"])
        )

        return props
