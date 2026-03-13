"""
Hierarchy forecast reconciliation.

Ensures forecasts at different hierarchy levels are consistent (coherent).

Supported methods
-----------------
bottom_up   — leaf forecasts are authoritative; aggregate by summing.
top_down    — top-level forecast disaggregated by historical proportions.
middle_out  — mid-level forecast disaggregated down and aggregated up.
ols         — Optimal reconciliation with identity error covariance (W = I).
wls         — Weighted Least Squares; uses structural weights (1/n_leaves)
              by default, or per-series residual variance when residuals are
              supplied.
mint        — Minimum Trace reconciliation (Wickramasuriya et al., 2019).
              Uses Ledoit–Wolf diagonal shrinkage covariance when residuals
              are supplied; falls back to WLS-structural otherwise.

OLS / WLS / MinT mathematics
-----------------------------
Given:
  S   — summing matrix (n_all × n_leaves);  S[i,j]=1 if leaf j ∈ subtree(i)
  P̂  — base forecast matrix (n_all × T), one column per time period
  W   — error covariance matrix (n_all × n_all)

Reconciled leaf forecasts:
  P̃_leaf = G · P̂
  G = (S′W⁻¹S)⁻¹ S′W⁻¹           (projects onto the coherent subspace)

For OLS,  W = I   → G = (S′S)⁻¹ S′
For WLS,  W = diag(w_i)
For MinT, W = Ŵ_shrink (Ledoit-Wolf diagonal shrinkage)

Base-forecast construction
--------------------------
``forecasts`` is keyed by hierarchy level name.  For nodes whose level is
not present, base forecasts are computed bottom-up (S @ P̂_leaf).  When
independent upper-level forecasters are available (e.g. a market-level
model alongside SKU models), pass them in the dict and they will be used
directly, making the reconciliation genuinely "optimal".
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

    After reconciliation, forecasts at every level are coherent:
    the sum of children equals the parent at every node.
    """

    _METHODS = {"bottom_up", "top_down", "middle_out", "mint", "ols", "wls"}

    def __init__(
        self,
        trees: Dict[str, HierarchyTree],
        config: ReconciliationConfig,
    ):
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

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def reconcile(
        self,
        forecasts: Dict[str, pl.DataFrame],
        actuals: Optional[pl.DataFrame] = None,
        residuals: Optional[pl.DataFrame] = None,
        value_columns: Optional[List[str]] = None,
        time_column: str = "week",
    ) -> pl.DataFrame:
        """
        Reconcile forecasts produced at various hierarchy levels.

        Parameters
        ----------
        forecasts:
            Keyed by hierarchy level name.  Each DataFrame must contain:
            - a column named after the level (node key),
            - ``time_column``,
            - the ``value_columns``.
        actuals:
            Historical actuals at leaf level — needed by top_down and
            middle_out to compute disaggregation proportions.
        residuals:
            In-sample forecast residuals for WLS/MinT.  Expected columns:
            ``["node_key", "node_level", time_column, "residual"]``.
            When omitted, WLS uses structural weights (1/n_leaves per node)
            and MinT falls back to the same.
        value_columns:
            Columns to reconcile.  Defaults to ``["forecast"]``.
            Pass ``["forecast", "forecast_p10", "forecast_p90"]`` to
            reconcile point and quantile forecasts simultaneously.
        time_column:
            Time column name.

        Returns
        -------
        Leaf-level reconciled DataFrame with columns
        [leaf_level_name, time_column] + value_columns.
        """
        value_columns = value_columns or ["forecast"]

        if self.config.method == "bottom_up":
            return self._bottom_up(forecasts, value_columns, time_column)
        elif self.config.method == "top_down":
            return self._top_down(forecasts, actuals, value_columns, time_column)
        elif self.config.method == "middle_out":
            return self._middle_out(forecasts, actuals, value_columns, time_column)
        elif self.config.method in ("ols", "wls", "mint"):
            return self._linear_reconcile(
                forecasts=forecasts,
                value_columns=value_columns,
                time_column=time_column,
                method=self.config.method,
                residuals=residuals,
            )
        else:
            raise ValueError(
                f"Reconciliation method {self.config.method!r} not handled."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Existing methods (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    def _bottom_up(
        self,
        forecasts: Dict[str, pl.DataFrame],
        value_columns: List[str],
        time_column: str,
    ) -> pl.DataFrame:
        """
        Bottom-Up: forecasts are at leaf level — no adjustment needed.
        Higher-level numbers are obtained by summing.
        """
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
        """Top-Down: disaggregate root-level forecast to leaves."""
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
        """Middle-Out: forecast at mid level, disaggregate down, aggregate up."""
        result = None
        for dim_name, tree in self.trees.items():
            agg = self._aggregators[dim_name]
            mid_level = tree.config.reconciliation_level

            if mid_level is None or mid_level not in forecasts:
                continue

            leaf_level = tree.leaf_level

            if mid_level == leaf_level:
                result = forecasts[mid_level]
            else:
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

    # ─────────────────────────────────────────────────────────────────────────
    # Linear reconciliation (OLS / WLS / MinT)
    # ─────────────────────────────────────────────────────────────────────────

    def _linear_reconcile(
        self,
        forecasts: Dict[str, pl.DataFrame],
        value_columns: List[str],
        time_column: str,
        method: str,
        residuals: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """
        Core linear reconciliation used by OLS, WLS, and MinT.

        For each hierarchy tree that has leaf-level forecasts, builds the
        summing matrix S, stacks base forecasts into P̂, computes
        G = (S′W⁻¹S)⁻¹S′W⁻¹, and returns P̃_leaf = G·P̂.

        Multiple value_columns (e.g. point + quantiles) are reconciled
        independently — the same G matrix applies to each because the
        reconciliation is a linear operator.
        """
        for _dim_name, tree in self.trees.items():
            leaf_level = tree.leaf_level

            if leaf_level not in forecasts:
                continue

            leaf_df = forecasts[leaf_level]

            # ── Build S matrix ────────────────────────────────────────────
            S_df = tree.summing_matrix()
            leaf_keys: List[str] = [
                c for c in S_df.columns if c not in ("node_key", "node_level")
            ]
            S: np.ndarray = S_df.select(leaf_keys).to_numpy().astype(float)
            node_keys: List[str] = S_df["node_key"].to_list()
            node_levels: List[str] = S_df["node_level"].to_list()
            n_all = len(node_keys)
            n_leaves = len(leaf_keys)

            # ── Time axis ─────────────────────────────────────────────────
            time_periods: List = sorted(leaf_df[time_column].unique().to_list())
            T = len(time_periods)
            time_to_idx = {t: i for i, t in enumerate(time_periods)}
            leaf_to_idx = {lk: i for i, lk in enumerate(leaf_keys)}
            node_to_idx = {nk: i for i, nk in enumerate(node_keys)}

            # ── Compute W⁻¹  (shared across value columns) ────────────────
            W_inv = self._build_W_inv(
                method=method,
                n_all=n_all,
                node_keys=node_keys,
                node_levels=node_levels,
                S=S,
                residuals=residuals,
                time_column=time_column,
            )

            # ── Reconciliation matrix G ────────────────────────────────────
            # G = (S′W⁻¹S)⁻¹ S′W⁻¹   shape: (n_leaves × n_all)
            StWinv = S.T @ W_inv                         # (n_leaves × n_all)
            StWinvS = StWinv @ S                         # (n_leaves × n_leaves)
            # Tikhonov regularisation for near-singular matrices
            reg = 1e-8 * np.trace(StWinvS) / n_leaves
            try:
                StWinvS_inv = np.linalg.inv(StWinvS + reg * np.eye(n_leaves))
            except np.linalg.LinAlgError:
                StWinvS_inv = np.linalg.pinv(StWinvS)
            G = StWinvS_inv @ StWinv                     # (n_leaves × n_all)

            # ── Reconcile each value column ────────────────────────────────
            reconciled: Dict[str, np.ndarray] = {}

            for vc in value_columns:
                if vc not in leaf_df.columns:
                    continue

                # Build P̂_leaf  (n_leaves × T) from leaf-level forecasts
                P_leaf = np.zeros((n_leaves, T))
                for row in leaf_df.select([leaf_level, time_column, vc]).iter_rows():
                    lk, t, val = row
                    l_idx = leaf_to_idx.get(str(lk))
                    t_idx = time_to_idx.get(t)
                    if l_idx is not None and t_idx is not None and val is not None:
                        P_leaf[l_idx, t_idx] = float(val)

                # P̂_all = S @ P̂_leaf  (bottom-up aggregates, n_all × T)
                P_all = S @ P_leaf

                # Override with any independent non-leaf level forecasts
                for level_name in tree.levels[:-1]:  # all non-leaf levels
                    if level_name not in forecasts:
                        continue
                    lv_df = forecasts[level_name]
                    if vc not in lv_df.columns:
                        continue
                    for row in lv_df.select([level_name, time_column, vc]).iter_rows():
                        nk, t, val = row
                        n_idx = node_to_idx.get(str(nk))
                        t_idx = time_to_idx.get(t)
                        if n_idx is not None and t_idx is not None and val is not None:
                            P_all[n_idx, t_idx] = float(val)

                # Reconcile: P̃_leaf = G · P̂_all  (n_leaves × T)
                P_tilde = G @ P_all
                P_tilde = np.maximum(P_tilde, 0.0)   # non-negativity constraint
                reconciled[vc] = P_tilde

            if not reconciled:
                raise ValueError(
                    f"None of the requested value_columns {value_columns} "
                    f"found in leaf-level forecast DataFrame."
                )

            # ── Build output DataFrame ─────────────────────────────────────
            rows = []
            for l_idx, lk in enumerate(leaf_keys):
                for t_idx, t in enumerate(time_periods):
                    row: Dict = {leaf_level: lk, time_column: t}
                    for vc, P_tilde in reconciled.items():
                        row[vc] = float(P_tilde[l_idx, t_idx])
                    rows.append(row)

            return pl.DataFrame(rows)

        raise ValueError(
            "No reconcilable hierarchy found with leaf-level forecasts. "
            f"Provided forecast levels: {list(forecasts.keys())}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # W-matrix construction helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_W_inv(
        self,
        method: str,
        n_all: int,
        node_keys: List[str],
        node_levels: List[str],
        S: np.ndarray,
        residuals: Optional[pl.DataFrame],
        time_column: str,
    ) -> np.ndarray:
        """
        Return W⁻¹ for the chosen reconciliation method.

        OLS   → W = I
        WLS   → W = diag(w_i) where w_i = n_leaves_under_i (structural)
                or diag(σ²_i) from residuals when provided
        MinT  → W = shrinkage covariance from residuals,
                or WLS-structural fallback when residuals are absent
        """
        if method == "ols":
            return np.eye(n_all)

        if method == "wls":
            if residuals is not None:
                diag = self._residual_variances(
                    residuals, node_keys, node_levels, time_column
                )
            else:
                # Structural weights: w_i = number of leaf descendants
                diag = np.maximum(S.sum(axis=1), 1.0)
            return np.diag(1.0 / np.maximum(diag, 1e-10))

        if method == "mint":
            if residuals is not None:
                W = self._shrinkage_cov(
                    residuals, node_keys, node_levels, time_column, n_all
                )
                try:
                    return np.linalg.inv(W)
                except np.linalg.LinAlgError:
                    return np.linalg.pinv(W)
            else:
                # No residuals supplied → fall back to WLS structural
                diag = np.maximum(S.sum(axis=1), 1.0)
                return np.diag(1.0 / np.maximum(diag, 1e-10))

        raise ValueError(f"Unknown method {method!r}")

    def _residual_variances(
        self,
        residuals: pl.DataFrame,
        node_keys: List[str],
        node_levels: List[str],
        time_column: str,
    ) -> np.ndarray:
        """
        Per-node variance of base-forecast residuals.

        ``residuals`` must have columns:
        ["node_key", "node_level", time_column, "residual"].
        """
        variances = np.ones(len(node_keys))
        for i, (nk, nl) in enumerate(zip(node_keys, node_levels)):
            match = residuals.filter(
                (pl.col("node_key") == nk) & (pl.col("node_level") == nl)
            )
            if not match.is_empty() and "residual" in match.columns:
                res = match["residual"].drop_nulls().to_numpy()
                if len(res) > 0:
                    variances[i] = max(float(np.var(res)), 1e-10)
        return variances

    def _shrinkage_cov(
        self,
        residuals: pl.DataFrame,
        node_keys: List[str],
        node_levels: List[str],
        time_column: str,
        n_all: int,
    ) -> np.ndarray:
        """
        Ledoit-Wolf diagonal shrinkage covariance estimator.

        Shrinkage target: diag(Ŵ_sample).
        Intensity λ ∈ [0,1] is set heuristically as min(1, n/T), which
        gives full shrinkage (diagonal W) when T < n and approaches the
        sample covariance as T → ∞.

        Returns a positive-definite (n_all × n_all) matrix.
        """
        all_times: List = sorted(residuals[time_column].unique().to_list())
        T = len(all_times)
        time_to_idx = {t: i for i, t in enumerate(all_times)}

        E = np.zeros((n_all, T))
        for i, (nk, nl) in enumerate(zip(node_keys, node_levels)):
            match = residuals.filter(
                (pl.col("node_key") == nk) & (pl.col("node_level") == nl)
            )
            if match.is_empty() or "residual" not in match.columns:
                continue
            for row in match.select([time_column, "residual"]).iter_rows():
                t, val = row
                t_idx = time_to_idx.get(t)
                if t_idx is not None and val is not None:
                    E[i, t_idx] = float(val)

        if T < 2:
            return np.diag(np.maximum(np.var(E, axis=1), 1e-10))

        # Sample covariance (with ddof=1)
        W_sample = np.cov(E)

        # Shrinkage intensity: aggressive when T < n, relaxed when T >> n
        lam = max(0.0, min(1.0, n_all / T))

        W_diag = np.diag(np.diag(W_sample))
        W_shrink = (1.0 - lam) * W_sample + lam * W_diag

        # Ensure strict positive definiteness
        min_eig = np.linalg.eigvalsh(W_shrink).min()
        if min_eig <= 0:
            W_shrink += (-min_eig + 1e-8) * np.eye(n_all)

        return W_shrink
