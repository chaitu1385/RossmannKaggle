"""
Hierarchy forecast reconciliation.

Ensures forecasts at different hierarchy levels are consistent (coherent).
Supports Bottom-Up, Top-Down, Middle-Out, and MinT methods.

Phase 1 implements Bottom-Up, Top-Down, and Middle-Out.
Phase 2 will add MinT (Minimum Trace), OLS, and WLS.
"""

from typing import Dict, List, Optional

import polars as pl

from .tree import HierarchyTree
from .aggregator import HierarchyAggregator
from ..config.schema import ReconciliationConfig


class Reconciler:
    """
    Reconcile forecasts across hierarchy levels.

    After reconciliation, forecasts at every level of the hierarchy are
    consistent: the sum of children equals the parent at every node.
    """

    _METHODS = {"bottom_up", "top_down", "middle_out", "mint", "ols", "wls"}

    def __init__(
        self,
        trees: Dict[str, HierarchyTree],
        config: ReconciliationConfig,
    ):
        """
        Parameters
        ----------
        trees:
            Hierarchy trees by dimension name (e.g. {"product": ..., "geography": ...}).
        config:
            Reconciliation settings from platform config.
        """
        self.trees = trees
        self.config = config
        self._aggregators = {
            name: HierarchyAggregator(tree)
            for name, tree in trees.items()
        }

        if config.method not in self._METHODS:
            raise ValueError(
                f"Unknown reconciliation method {config.method!r}. "
                f"Supported: {self._METHODS}"
            )

    def reconcile(
        self,
        forecasts: Dict[str, pl.DataFrame],
        actuals: Optional[pl.DataFrame] = None,
        value_columns: Optional[List[str]] = None,
        time_column: str = "week",
    ) -> pl.DataFrame:
        """
        Reconcile forecasts produced at various hierarchy levels.

        Parameters
        ----------
        forecasts:
            Keyed by the hierarchy level name at which the forecast was
            produced.  Each DataFrame must contain the level column, the
            time column, and the value columns.
        actuals:
            Historical actuals at the leaf level (needed for top-down
            proportions and MinT).
        value_columns:
            Columns to reconcile. Defaults to ``["forecast"]``.
        time_column:
            Time column name.

        Returns
        -------
        Leaf-level reconciled forecasts.
        """
        value_columns = value_columns or ["forecast"]

        if self.config.method == "bottom_up":
            return self._bottom_up(forecasts, value_columns, time_column)
        elif self.config.method == "top_down":
            return self._top_down(forecasts, actuals, value_columns, time_column)
        elif self.config.method == "middle_out":
            return self._middle_out(forecasts, actuals, value_columns, time_column)
        else:
            raise NotImplementedError(
                f"Reconciliation method {self.config.method!r} is Phase 2. "
                f"Use bottom_up, top_down, or middle_out."
            )

    def _bottom_up(
        self,
        forecasts: Dict[str, pl.DataFrame],
        value_columns: List[str],
        time_column: str,
    ) -> pl.DataFrame:
        """
        Bottom-Up: forecasts are at leaf level, aggregate up.

        The leaf-level forecast IS the reconciled forecast — no adjustment
        needed.  We just return it as-is.  Higher-level numbers are obtained
        by summing.
        """
        # Find the leaf-level forecast
        for tree in self.trees.values():
            leaf = tree.leaf_level
            if leaf in forecasts:
                return forecasts[leaf]

        raise ValueError(
            "Bottom-Up reconciliation requires a leaf-level forecast. "
            f"Got levels: {list(forecasts.keys())}"
        )

    def _top_down(
        self,
        forecasts: Dict[str, pl.DataFrame],
        actuals: Optional[pl.DataFrame],
        value_columns: List[str],
        time_column: str,
    ) -> pl.DataFrame:
        """
        Top-Down: forecast at top level, disaggregate to leaves using
        historical proportions.
        """
        if actuals is None:
            raise ValueError(
                "Top-Down reconciliation requires historical actuals "
                "to compute disaggregation proportions."
            )

        result = None
        for dim_name, tree in self.trees.items():
            agg = self._aggregators[dim_name]
            top_level = tree.levels[0]
            leaf_level = tree.leaf_level

            if top_level not in forecasts:
                continue

            # Compute historical proportions
            props = agg.compute_historical_proportions(
                actuals,
                source_level=top_level,
                target_level=leaf_level,
                value_column=value_columns[0],
                time_column=time_column,
            )

            result = agg.disaggregate_to(
                forecasts[top_level],
                source_level=top_level,
                target_level=leaf_level,
                value_columns=value_columns,
                proportions=props,
                time_column=time_column,
            )

        if result is None:
            raise ValueError("No top-level forecasts found for Top-Down reconciliation")

        return result

    def _middle_out(
        self,
        forecasts: Dict[str, pl.DataFrame],
        actuals: Optional[pl.DataFrame],
        value_columns: List[str],
        time_column: str,
    ) -> pl.DataFrame:
        """
        Middle-Out: forecast at a mid level, then:
        - Disaggregate DOWN to leaves using historical proportions
        - Aggregate UP to higher levels by summing

        This is the recommended approach: forecast at CDS×ProductUnit,
        disaggregate to Country, aggregate to Region/Global.
        """
        result = None
        for dim_name, tree in self.trees.items():
            agg = self._aggregators[dim_name]
            mid_level = tree.config.reconciliation_level

            if mid_level is None or mid_level not in forecasts:
                continue

            leaf_level = tree.leaf_level

            if mid_level == leaf_level:
                # Already at leaf, no disaggregation needed
                result = forecasts[mid_level]
            else:
                # Disaggregate to leaf
                if actuals is None:
                    raise ValueError(
                        f"Middle-Out for {dim_name!r}: need actuals to compute "
                        f"proportions from {mid_level!r} → {leaf_level!r}."
                    )
                props = agg.compute_historical_proportions(
                    actuals,
                    source_level=mid_level,
                    target_level=leaf_level,
                    value_column=value_columns[0],
                    time_column=time_column,
                )
                result = agg.disaggregate_to(
                    forecasts[mid_level],
                    source_level=mid_level,
                    target_level=leaf_level,
                    value_columns=value_columns,
                    proportions=props,
                    time_column=time_column,
                )

        if result is None:
            raise ValueError(
                "No mid-level forecasts found for Middle-Out reconciliation"
            )

        return result
