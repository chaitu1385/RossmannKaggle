"""
Hierarchy forecast reconciliation.

Ensures forecasts at different hierarchy levels are consistent (coherent).
Supports Bottom-Up, Top-Down, Middle-Out, and MinT/OLS/WLS methods.

Phase 1: Bottom-Up, Top-Down, Middle-Out.
Phase 2: MinT (Minimum Trace), OLS, WLS.

MinT Theory
-----------
Given base (unreconciled) forecasts ŷ stacked over all hierarchy levels and a
summing matrix S (shape n_nodes × n_leaves, where S[i,j]=1 if leaf j rolls up
into node i), the MinT reconciled leaf forecasts are:

    ỹ = P ŷ     where  P = (S' W⁻¹ S)⁻¹ S' W⁻¹

W is the covariance matrix of forecast errors:
- OLS:  W = I  (identity — equal uncertainty at every level)
- WLS:  W = diag(σ²)  (per-series variance, estimated from residuals or actuals)
- MinT: W = full covariance with Ledoit-Wolf diagonal shrinkage

All-level reconciled forecasts follow from S ỹ.

References
----------
Wickramasuriya et al. (2019). "Optimal Forecast Reconciliation Using a Unifying
Framework for the ETS-ARIMA State Space Model." JASA.
"""

from typing import Dict, List, Optional

import numpy as np
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
        elif self.config.method in {"mint", "ols", "wls"}:
            return self._mint_reconcile(forecasts, actuals, value_columns, time_column, self.config.method)
        else:
            raise NotImplementedError(
                f"Unknown reconciliation method {self.config.method!r}. "
                f"Supported: {self._METHODS}"
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

    def _mint_reconcile(
        self,
        forecasts: Dict[str, pl.DataFrame],
        actuals: Optional[pl.DataFrame],
        value_columns: List[str],
        time_column: str,
        method: str,
    ) -> pl.DataFrame:
        """
        MinT / OLS / WLS reconciliation (Phase 2).

        Requires leaf-level base forecasts.  All-level base forecasts are
        derived by summing leaves through the summing matrix S, so the
        reconciliation operates on a fully coherent starting point.

        Parameters
        ----------
        forecasts:
            Must include the leaf level for each hierarchy dimension.
        actuals:
            Historical leaf-level actuals used to estimate forecast error
            variances (WLS) or the full covariance (MinT).  If None, the
            method degrades gracefully: WLS uses forecast variance; MinT
            uses the same diagonal approximation.
        value_columns:
            Forecast value columns to reconcile.
        time_column:
            Time column name.
        method:
            "ols" | "wls" | "mint"
        """
        result: Optional[pl.DataFrame] = None

        for dim_name, tree in self.trees.items():
            # ── Summing matrix ──────────────────────────────────────────────
            S_df = tree.summing_matrix()
            leaf_keys = [c for c in S_df.columns if c not in ("node_key", "node_level")]
            S_np = S_df.select(leaf_keys).to_numpy().astype(np.float64)  # (n_nodes, n_leaves)
            n_nodes, n_leaves = S_np.shape
            leaf_level = tree.leaf_level

            if leaf_level not in forecasts:
                raise ValueError(
                    f"MinT/OLS/WLS requires leaf-level ('{leaf_level}') forecasts "
                    f"for dimension '{dim_name}'. Got: {list(forecasts.keys())}"
                )

            leaf_df = forecasts[leaf_level]

            dim_result: Optional[pl.DataFrame] = None

            for vc in value_columns:
                if vc not in leaf_df.columns:
                    continue

                # ── Pivot leaf forecasts: (T, n_leaves) ────────────────────
                leaf_wide = leaf_df.pivot(values=vc, index=time_column, on=leaf_level)
                times = leaf_wide[time_column].to_list()
                T = len(times)

                Y_np = np.zeros((T, n_leaves), dtype=np.float64)
                for j, lk in enumerate(leaf_keys):
                    if lk in leaf_wide.columns:
                        col = leaf_wide[lk]
                        Y_np[:, j] = col.fill_null(0).cast(pl.Float64).to_numpy()

                # ── Base forecasts at ALL levels: (T, n_nodes) ─────────────
                # yhat_all[t, i] = sum of leaves under node i at time t
                yhat_all = Y_np @ S_np.T

                # ── W inverse ───────────────────────────────────────────────
                W_inv = self._build_W_inv(
                    method, n_nodes, n_leaves, S_np,
                    yhat_all, Y_np, actuals, leaf_keys, leaf_level, time_column, vc,
                )

                # ── MinT formula: P = (S' W⁻¹ S)⁻¹ S' W⁻¹ ─────────────────
                SWinv = S_np.T @ W_inv          # (n_leaves, n_nodes)
                SWinvS = SWinv @ S_np           # (n_leaves, n_leaves)
                P = np.linalg.pinv(SWinvS) @ SWinv  # (n_leaves, n_nodes)

                # Reconciled leaf forecasts: (T, n_leaves)
                reconciled = yhat_all @ P.T

                # ── Back to Polars long format ───────────────────────────────
                wide_out = pl.DataFrame(
                    {time_column: times,
                     **{leaf_keys[j]: reconciled[:, j].tolist() for j in range(n_leaves)}}
                )
                long_out = wide_out.unpivot(
                    on=leaf_keys,
                    index=[time_column],
                    variable_name=leaf_level,
                    value_name=vc,
                )

                dim_result = long_out if dim_result is None else dim_result.join(
                    long_out, on=[time_column, leaf_level], how="inner"
                )

            if dim_result is not None:
                result = dim_result

        if result is None:
            raise ValueError("No leaf-level forecasts found for MinT/OLS/WLS reconciliation.")

        return result

    def _build_W_inv(
        self,
        method: str,
        n_nodes: int,
        n_leaves: int,
        S_np: np.ndarray,
        yhat_all: np.ndarray,
        Y_leaves: np.ndarray,
        actuals: Optional[pl.DataFrame],
        leaf_keys: List[str],
        leaf_level: str,
        time_column: str,
        vc: str,
    ) -> np.ndarray:
        """
        Build the inverse of the weight matrix W for MinT/OLS/WLS.

        OLS
            W = I  →  W⁻¹ = I.  Equal weight everywhere.

        WLS
            W = diag(σ²_i) where σ²_i is the variance of node i's forecast
            errors.  Estimated from actuals if provided, else from the spread
            of the base forecasts over time.

        MinT
            W is the full forecast-error covariance matrix, estimated using
            Ledoit-Wolf diagonal shrinkage to ensure positive definiteness
            even when the number of nodes exceeds the number of time steps.

            Shrinkage intensity λ is set analytically as:
                λ = min(1, (n_nodes + 2) / T)
            which increases shrinkage as the problem becomes ill-conditioned.
        """
        if method == "ols":
            return np.eye(n_nodes)

        # ── Estimate residuals ───────────────────────────────────────────────
        # Prefer actuals-based residuals; fall back to centred forecasts.
        residuals: np.ndarray
        if actuals is not None and leaf_level in actuals.columns and vc in actuals.columns:
            try:
                act_wide = actuals.pivot(values=vc, index=time_column, on=leaf_level)
                T_act = len(act_wide)
                A_np = np.zeros((T_act, n_leaves), dtype=np.float64)
                for j, lk in enumerate(leaf_keys):
                    if lk in act_wide.columns:
                        A_np[:, j] = act_wide[lk].fill_null(0).cast(pl.Float64).to_numpy()
                A_all = A_np @ S_np.T   # actuals at all levels: (T_act, n_nodes)
                T_min = min(yhat_all.shape[0], A_all.shape[0])
                residuals = yhat_all[:T_min] - A_all[:T_min]
            except Exception:
                residuals = yhat_all - yhat_all.mean(axis=0, keepdims=True)
        else:
            residuals = yhat_all - yhat_all.mean(axis=0, keepdims=True)

        T_r = max(residuals.shape[0], 1)

        if method == "wls":
            # Diagonal: per-node variance
            variances = np.var(residuals, axis=0, ddof=min(1, T_r - 1))
            variances = np.where(variances < 1e-10, 1e-10, variances)
            return np.diag(1.0 / variances)

        # method == "mint": shrinkage covariance
        # Sample covariance (unbiased)
        cov = (residuals.T @ residuals) / T_r  # (n_nodes, n_nodes)

        # Ledoit-Wolf diagonal target
        diag_cov = np.diag(np.diag(cov))

        # Analytical shrinkage intensity: increases when n_nodes >> T
        shrinkage = float(np.clip((n_nodes + 2) / T_r, 0.0, 1.0))
        W = (1.0 - shrinkage) * cov + shrinkage * diag_cov

        # Regularise to guarantee invertibility
        W += np.eye(n_nodes) * 1e-8

        try:
            return np.linalg.inv(W)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(W)
