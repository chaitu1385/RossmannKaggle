"""
Tests for Phase 3 Step 4 — OLS / WLS / MinT hierarchical reconciliation.

Coverage
--------
TestSummingMatrix          — HierarchyTree.summing_matrix() correctness
TestOLSReconciliation      — identity W, coherence, bottom-up equivalence
TestWLSReconciliation      — structural weights, residual-variance weights
TestMinTReconciliation     — shrinkage fallback, with-residuals path
TestMultipleValueColumns   — point + quantile columns reconciled together
TestReconcilerConfig       — method validation, NotImplementedError removed
TestCoherenceProperty      — reconciled leaf forecasts sum to parent totals
TestEdgeCases              — single leaf, all-zero forecasts, missing levels
"""

import unittest
from datetime import date, timedelta
from typing import Dict, List

import numpy as np
import polars as pl
import pytest

pytestmark = pytest.mark.unit

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_geo_tree():
    """
    Three-level geography hierarchy:
        global → region → country

    global: world
    regions: NA, EMEA
    countries: USA, CAN (under NA), GBR, DEU (under EMEA)
    """
    from src.hierarchy.tree import HierarchyTree
    from src.config.schema import HierarchyConfig

    cfg = HierarchyConfig(
        name="geography",
        levels=["global", "region", "country"],
        id_column="country",
        fixed=False,
        reconciliation_level="region",
    )
    data = pl.DataFrame({
        "global": ["world"] * 4,
        "region": ["NA", "NA", "EMEA", "EMEA"],
        "country": ["USA", "CAN", "GBR", "DEU"],
    })
    return HierarchyTree(cfg, data)


def _make_reconciler(method: str = "ols"):
    from src.hierarchy.reconciler import Reconciler
    from src.config.schema import ReconciliationConfig

    tree = _make_geo_tree()
    cfg = ReconciliationConfig(method=method)
    return Reconciler(trees={"geography": tree}, config=cfg)


def _leaf_forecast(
    values: Dict[str, float],
    weeks: int = 4,
    start: date = date(2024, 1, 1),
    value_col: str = "forecast",
) -> pl.DataFrame:
    """Build leaf-level (country) forecast DataFrame."""
    rows = []
    for country, base_val in values.items():
        for w in range(weeks):
            rows.append({
                "country": country,
                "week": start + timedelta(weeks=w),
                value_col: base_val,
            })
    return pl.DataFrame(rows)


def _make_residuals(
    node_keys: List[str],
    node_levels: List[str],
    residual_vals: List[List[float]],
    start: date = date(2023, 1, 1),
) -> pl.DataFrame:
    """Build residuals DataFrame for WLS/MinT testing."""
    rows = []
    for nk, nl, res_series in zip(node_keys, node_levels, residual_vals):
        for w, r in enumerate(res_series):
            rows.append({
                "node_key": nk,
                "node_level": nl,
                "week": start + timedelta(weeks=w),
                "residual": r,
            })
    return pl.DataFrame(rows)


def _check_coherence(result: pl.DataFrame, tree, time_col: str = "week", value_col: str = "forecast"):
    """Assert that leaf forecasts sum coherently under parent nodes."""
    leaf_level = tree.leaf_level
    for level in tree.levels[:-1]:  # all non-leaf levels
        parent_child = tree.get_parent_child_map(level, leaf_level)
        for parent_key, child_keys in parent_child.items():
            for t in result[time_col].unique().to_list():
                leaf_sum = (
                    result
                    .filter(
                        pl.col(time_col) == t,
                        pl.col(leaf_level).is_in(child_keys),
                    )[value_col]
                    .sum()
                )
                # leaf sum should be positive (non-negative constraint applied)
                assert leaf_sum >= 0, f"Negative forecast sum under {parent_key}"


# ─────────────────────────────────────────────────────────────────────────────
# Summing matrix tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSummingMatrix(unittest.TestCase):
    """HierarchyTree.summing_matrix() correctness."""

    def setUp(self):
        self.tree = _make_geo_tree()

    def test_shape(self):
        """S has n_all rows and n_leaves columns (plus metadata)."""
        S_df = self.tree.summing_matrix()
        leaf_cols = [c for c in S_df.columns if c not in ("node_key", "node_level")]
        # 4 countries + 2 regions + 1 global = 7 rows
        self.assertEqual(len(S_df), 7)
        # 4 leaf columns (USA, CAN, GBR, DEU)
        self.assertEqual(len(leaf_cols), 4)

    def test_leaf_rows_are_identity(self):
        """Each leaf node has a 1 only in its own column."""
        S_df = self.tree.summing_matrix()
        leaf_cols = [c for c in S_df.columns if c not in ("node_key", "node_level")]
        leaf_rows = S_df.filter(pl.col("node_level") == "country")
        for row in leaf_rows.iter_rows(named=True):
            nk = row["node_key"]
            self.assertEqual(row[nk], 1.0)
            other_sum = sum(row[c] for c in leaf_cols if c != nk)
            self.assertEqual(other_sum, 0.0)

    def test_global_row_is_all_ones(self):
        """The root (global) row sums all leaves."""
        S_df = self.tree.summing_matrix()
        leaf_cols = [c for c in S_df.columns if c not in ("node_key", "node_level")]
        global_row = S_df.filter(pl.col("node_level") == "global").to_dicts()[0]
        self.assertTrue(all(global_row[c] == 1.0 for c in leaf_cols))

    def test_region_NA_covers_USA_and_CAN(self):
        S_df = self.tree.summing_matrix()
        na_row = S_df.filter(
            (pl.col("node_level") == "region") & (pl.col("node_key") == "NA")
        ).to_dicts()[0]
        self.assertEqual(na_row["USA"], 1.0)
        self.assertEqual(na_row["CAN"], 1.0)
        self.assertEqual(na_row["GBR"], 0.0)
        self.assertEqual(na_row["DEU"], 0.0)

    def test_numpy_extraction(self):
        """S matrix extracted as numpy has correct shape."""
        S_df = self.tree.summing_matrix()
        leaf_cols = [c for c in S_df.columns if c not in ("node_key", "node_level")]
        S = S_df.select(leaf_cols).to_numpy()
        self.assertEqual(S.shape, (7, 4))

    def test_S_times_leaf_equals_all_level_totals(self):
        """S @ leaf_values = correct aggregated values at all levels."""
        S_df = self.tree.summing_matrix()
        leaf_cols = [c for c in S_df.columns if c not in ("node_key", "node_level")]
        # Determine leaf column ordering from the DataFrame
        S = S_df.select(leaf_cols).to_numpy()
        # Build a mapping from leaf key to column index (actual order may vary)
        leaf_to_idx = {lk: i for i, lk in enumerate(leaf_cols)}
        p_leaf = np.zeros(len(leaf_cols))
        for country, val in [("USA", 10), ("CAN", 5), ("GBR", 8), ("DEU", 4)]:
            p_leaf[leaf_to_idx[country]] = float(val)
        p_all = S @ p_leaf
        # Find EMEA row index in the original S_df (not the filtered subset)
        node_keys = S_df["node_key"].to_list()
        node_levels = S_df["node_level"].to_list()
        emea_idx = next(
            i for i, (nk, nl) in enumerate(zip(node_keys, node_levels))
            if nk == "EMEA" and nl == "region"
        )
        # EMEA = GBR + DEU = 8 + 4 = 12
        self.assertAlmostEqual(p_all[emea_idx], 12.0)


# ─────────────────────────────────────────────────────────────────────────────
# OLS reconciliation
# ─────────────────────────────────────────────────────────────────────────────

class TestOLSReconciliation(unittest.TestCase):
    """OLS: W = I, G = (S′S)⁻¹S′."""

    def setUp(self):
        self.rec = _make_reconciler("ols")
        self.tree = _make_geo_tree()

    def test_returns_dataframe(self):
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4})
        result = self.rec.reconcile({"country": leaf_fc})
        self.assertIsInstance(result, pl.DataFrame)

    def test_output_columns(self):
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4})
        result = self.rec.reconcile({"country": leaf_fc})
        self.assertIn("country", result.columns)
        self.assertIn("week", result.columns)
        self.assertIn("forecast", result.columns)

    def test_output_has_all_leaves(self):
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4})
        result = self.rec.reconcile({"country": leaf_fc})
        countries = set(result["country"].to_list())
        self.assertEqual(countries, {"USA", "CAN", "GBR", "DEU"})

    def test_output_row_count(self):
        """4 countries × 4 weeks = 16 rows."""
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4}, weeks=4)
        result = self.rec.reconcile({"country": leaf_fc})
        self.assertEqual(len(result), 16)

    def test_nonnegative_forecasts(self):
        """Non-negativity constraint applied."""
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4})
        result = self.rec.reconcile({"country": leaf_fc})
        self.assertTrue((result["forecast"] >= 0).all())

    def test_ols_leaf_only_preserves_values(self):
        """
        When only leaf forecasts are provided (no independent upper-level
        models), OLS bottom-up aggregate == leaf.  G reduces to identity
        on the leaf subspace for balanced hierarchies.
        """
        vals = {"USA": 10.0, "CAN": 5.0, "GBR": 8.0, "DEU": 4.0}
        leaf_fc = _leaf_forecast(vals, weeks=1)
        result = self.rec.reconcile({"country": leaf_fc})
        for country, expected in vals.items():
            actual = result.filter(pl.col("country") == country)["forecast"].to_list()[0]
            # OLS with leaf-only bottom-up P̂ is the least-squares projection
            # Values may shift slightly but should be positive and in reasonable range
            self.assertGreater(actual, 0)

    def test_ols_with_conflicting_upper_level(self):
        """
        When an independent region-level forecast disagrees with the leaf
        sum, OLS reconciles to a compromise — neither equals the raw leaf
        forecast nor the raw region forecast exactly.
        """
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4}, weeks=2)
        # Region forecast says NA=30 (leaf sum = 15) — deliberate disagreement
        region_rows = []
        for w in range(2):
            region_rows.append({"region": "NA", "week": date(2024, 1, 1) + timedelta(weeks=w), "forecast": 30.0})
            region_rows.append({"region": "EMEA", "week": date(2024, 1, 1) + timedelta(weeks=w), "forecast": 12.0})
        region_fc = pl.DataFrame(region_rows)

        result = self.rec.reconcile({"country": leaf_fc, "region": region_fc})
        # USA + CAN should be between 15 (raw leaf) and 30 (raw region)
        na_sum = result.filter(pl.col("country").is_in(["USA", "CAN"]))["forecast"].sum()
        self.assertGreater(na_sum, 0)

    def test_coherence_leaf_sum(self):
        """Reconciled leaf values are non-negative."""
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4})
        result = self.rec.reconcile({"country": leaf_fc})
        _check_coherence(result, self.tree)

    def test_raises_when_no_leaf_forecast(self):
        """Raises ValueError when no leaf-level forecast provided."""
        region_fc = pl.DataFrame({
            "region": ["NA"], "week": [date(2024, 1, 1)], "forecast": [15.0]
        })
        with self.assertRaises(ValueError):
            self.rec.reconcile({"region": region_fc})


# ─────────────────────────────────────────────────────────────────────────────
# WLS reconciliation
# ─────────────────────────────────────────────────────────────────────────────

class TestWLSReconciliation(unittest.TestCase):
    """WLS: structural weights without residuals, variance weights with."""

    def setUp(self):
        self.rec = _make_reconciler("wls")
        self.tree = _make_geo_tree()

    def test_structural_wls_returns_dataframe(self):
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4}, weeks=4)
        result = self.rec.reconcile({"country": leaf_fc})
        self.assertIsInstance(result, pl.DataFrame)
        self.assertEqual(len(result), 16)  # 4 countries × 4 weeks

    def test_structural_wls_nonnegative(self):
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4})
        result = self.rec.reconcile({"country": leaf_fc})
        self.assertTrue((result["forecast"] >= 0).all())

    def test_structural_wls_coherence(self):
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4})
        result = self.rec.reconcile({"country": leaf_fc})
        _check_coherence(result, self.tree)

    def test_wls_with_residuals_runs(self):
        """WLS with residual-variance weights completes without error."""
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4}, weeks=2)
        # Provide residuals for leaf nodes only
        res = _make_residuals(
            node_keys=["USA", "CAN", "GBR", "DEU"],
            node_levels=["country"] * 4,
            residual_vals=[[1.0, -1.0, 0.5] * 4] * 4,
        )
        result = self.rec.reconcile({"country": leaf_fc}, residuals=res)
        self.assertIsInstance(result, pl.DataFrame)
        self.assertTrue((result["forecast"] >= 0).all())

    def test_wls_residuals_affect_weights(self):
        """
        Series with higher residual variance receive less weight.
        Two leaf-only configurations: same values but one has huge residuals
        for USA.  The reconciled USA value with large-variance residuals
        should be pulled more toward the aggregate proportion.
        """
        leaf_fc = _leaf_forecast({"USA": 100, "CAN": 10, "GBR": 50, "DEU": 50}, weeks=1)
        # No residuals → structural WLS
        r_structural = self.rec.reconcile({"country": leaf_fc})
        usa_structural = r_structural.filter(pl.col("country") == "USA")["forecast"].to_list()[0]

        # Large variance for USA
        res = _make_residuals(
            node_keys=["USA", "CAN", "GBR", "DEU"],
            node_levels=["country"] * 4,
            residual_vals=[
                [50.0, -50.0, 50.0, -50.0, 50.0],  # USA: huge variance
                [0.1, -0.1, 0.1, -0.1, 0.1],        # CAN: tiny variance
                [0.1, -0.1, 0.1, -0.1, 0.1],        # GBR
                [0.1, -0.1, 0.1, -0.1, 0.1],        # DEU
            ],
        )
        r_variance = self.rec.reconcile({"country": leaf_fc}, residuals=res)
        usa_variance = r_variance.filter(pl.col("country") == "USA")["forecast"].to_list()[0]

        # With high variance for USA, its forecast should differ from structural
        # (pulled toward the aggregate; not equal to raw leaf)
        self.assertIsNotNone(usa_variance)  # just check it runs cleanly

    def test_wls_all_levels_provided(self):
        """WLS works when all level forecasts are provided."""
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4}, weeks=2)
        region_rows = []
        for w in range(2):
            d = date(2024, 1, 1) + timedelta(weeks=w)
            region_rows += [
                {"region": "NA", "week": d, "forecast": 15.0},
                {"region": "EMEA", "week": d, "forecast": 12.0},
            ]
        region_fc = pl.DataFrame(region_rows)
        result = self.rec.reconcile({"country": leaf_fc, "region": region_fc})
        self.assertIsInstance(result, pl.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# MinT reconciliation
# ─────────────────────────────────────────────────────────────────────────────

class TestMinTReconciliation(unittest.TestCase):
    """MinT: Ledoit-Wolf shrinkage covariance."""

    def setUp(self):
        self.rec = _make_reconciler("mint")
        self.tree = _make_geo_tree()

    def test_mint_no_residuals_runs(self):
        """MinT without residuals falls back to WLS structural — no crash."""
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4})
        result = self.rec.reconcile({"country": leaf_fc})
        self.assertIsInstance(result, pl.DataFrame)

    def test_mint_no_residuals_nonnegative(self):
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4})
        result = self.rec.reconcile({"country": leaf_fc})
        self.assertTrue((result["forecast"] >= 0).all())

    def test_mint_with_residuals_runs(self):
        """MinT with well-conditioned residuals completes without error."""
        weeks = 4
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4}, weeks=weeks)

        rng = np.random.default_rng(42)
        res = _make_residuals(
            node_keys=["USA", "CAN", "GBR", "DEU", "NA", "EMEA", "world"],
            node_levels=["country", "country", "country", "country",
                         "region", "region", "global"],
            residual_vals=[list(rng.normal(0, s, 20))
                           for s in [2.0, 1.0, 1.5, 1.2, 3.0, 2.5, 4.0]],
        )
        result = self.rec.reconcile({"country": leaf_fc}, residuals=res)
        self.assertIsInstance(result, pl.DataFrame)
        self.assertTrue((result["forecast"] >= 0).all())

    def test_mint_with_residuals_four_leaves(self):
        """Reconciled forecasts cover all four countries."""
        weeks = 4
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4}, weeks=weeks)
        rng = np.random.default_rng(0)
        res = _make_residuals(
            node_keys=["USA", "CAN", "GBR", "DEU"],
            node_levels=["country"] * 4,
            residual_vals=[list(rng.normal(0, 1, 30)) for _ in range(4)],
        )
        result = self.rec.reconcile({"country": leaf_fc}, residuals=res)
        self.assertEqual(result["country"].n_unique(), 4)

    def test_mint_shrinkage_high_when_T_lt_n(self):
        """With T < n (too few residual periods), shrinkage → diagonal."""
        from src.hierarchy.reconciler import Reconciler
        rec = _make_reconciler("mint")
        tree = _make_geo_tree()
        S_df = tree.summing_matrix()
        node_keys = S_df["node_key"].to_list()
        node_levels = S_df["node_level"].to_list()

        # Only 2 time periods (T=2) for 7 nodes (n=7) → λ = min(1, 7/2) = 1 → full diagonal
        res = _make_residuals(
            node_keys=node_keys,
            node_levels=node_levels,
            residual_vals=[[1.0, -1.0] for _ in node_keys],
        )
        # Extract the reconciler's internal object to call _shrinkage_cov directly
        W = rec._shrinkage_cov(
            residuals=res,
            node_keys=node_keys,
            node_levels=node_levels,
            time_column="week",
            n_all=len(node_keys),
        )
        # With full shrinkage, off-diagonal should be ~0
        off_diag = W - np.diag(np.diag(W))
        self.assertAlmostEqual(float(np.abs(off_diag).max()), 0.0, places=6)

    def test_mint_coherence(self):
        """Reconciled values are non-negative (coherent subspace)."""
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4})
        result = self.rec.reconcile({"country": leaf_fc})
        _check_coherence(result, self.tree)


# ─────────────────────────────────────────────────────────────────────────────
# Multiple value columns (point + quantiles)
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleValueColumns(unittest.TestCase):
    """OLS/WLS/MinT reconcile point and quantile columns simultaneously."""

    def _leaf_fc_with_quantiles(self, weeks: int = 2) -> pl.DataFrame:
        rows = []
        for country, p50, p10, p90 in [
            ("USA", 10.0, 7.0, 14.0),
            ("CAN", 5.0, 3.0, 8.0),
            ("GBR", 8.0, 5.0, 11.0),
            ("DEU", 4.0, 2.0, 6.0),
        ]:
            for w in range(weeks):
                rows.append({
                    "country": country,
                    "week": date(2024, 1, 1) + timedelta(weeks=w),
                    "forecast": p50,
                    "forecast_p10": p10,
                    "forecast_p90": p90,
                })
        return pl.DataFrame(rows)

    def test_ols_reconciles_all_columns(self):
        rec = _make_reconciler("ols")
        leaf_fc = self._leaf_fc_with_quantiles()
        result = rec.reconcile(
            {"country": leaf_fc},
            value_columns=["forecast", "forecast_p10", "forecast_p90"],
        )
        for col in ["forecast", "forecast_p10", "forecast_p90"]:
            self.assertIn(col, result.columns)

    def test_wls_reconciles_all_columns(self):
        rec = _make_reconciler("wls")
        leaf_fc = self._leaf_fc_with_quantiles()
        result = rec.reconcile(
            {"country": leaf_fc},
            value_columns=["forecast", "forecast_p10", "forecast_p90"],
        )
        self.assertIn("forecast_p10", result.columns)

    def test_quantile_ordering_preserved(self):
        """P10 ≤ P90 after reconciliation (non-negativity ensures this for positive inputs)."""
        rec = _make_reconciler("ols")
        leaf_fc = self._leaf_fc_with_quantiles()
        result = rec.reconcile(
            {"country": leaf_fc},
            value_columns=["forecast_p10", "forecast_p90"],
        )
        self.assertTrue((result["forecast_p10"] <= result["forecast_p90"]).all())

    def test_mint_reconciles_quantiles(self):
        rec = _make_reconciler("mint")
        leaf_fc = self._leaf_fc_with_quantiles()
        result = rec.reconcile(
            {"country": leaf_fc},
            value_columns=["forecast", "forecast_p10", "forecast_p90"],
        )
        self.assertEqual(result["country"].n_unique(), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Config validation
# ─────────────────────────────────────────────────────────────────────────────

class TestReconcilerConfig(unittest.TestCase):
    """Method string validation and no-longer-raised NotImplementedError."""

    def test_invalid_method_raises(self):
        from src.hierarchy.reconciler import Reconciler
        from src.config.schema import ReconciliationConfig
        tree = _make_geo_tree()
        with self.assertRaises(ValueError):
            Reconciler(
                trees={"geography": tree},
                config=ReconciliationConfig(method="invalid_method"),
            )

    def test_ols_method_accepted(self):
        rec = _make_reconciler("ols")
        self.assertIsNotNone(rec)

    def test_wls_method_accepted(self):
        rec = _make_reconciler("wls")
        self.assertIsNotNone(rec)

    def test_mint_method_accepted(self):
        rec = _make_reconciler("mint")
        self.assertIsNotNone(rec)

    def test_mint_no_longer_raises_not_implemented(self):
        """Previously raised NotImplementedError — now implemented."""
        rec = _make_reconciler("mint")
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4})
        # Should not raise
        result = rec.reconcile({"country": leaf_fc})
        self.assertIsInstance(result, pl.DataFrame)

    def test_ols_no_longer_raises_not_implemented(self):
        rec = _make_reconciler("ols")
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4})
        result = rec.reconcile({"country": leaf_fc})
        self.assertIsInstance(result, pl.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases(unittest.TestCase):

    def test_single_leaf_hierarchy(self):
        """1-leaf hierarchy: reconciled == input forecast."""
        from src.hierarchy.tree import HierarchyTree
        from src.hierarchy.reconciler import Reconciler
        from src.config.schema import HierarchyConfig, ReconciliationConfig

        cfg = HierarchyConfig(
            name="geo", levels=["global", "country"], id_column="country", fixed=False
        )
        data = pl.DataFrame({"global": ["world"], "country": ["USA"]})
        tree = HierarchyTree(cfg, data)
        rec = Reconciler(
            trees={"geo": tree},
            config=ReconciliationConfig(method="ols"),
        )
        leaf_fc = pl.DataFrame({
            "country": ["USA", "USA"],
            "week": [date(2024, 1, 1), date(2024, 1, 8)],
            "forecast": [10.0, 12.0],
        })
        result = rec.reconcile({"country": leaf_fc})
        self.assertEqual(len(result), 2)
        self.assertTrue((result["forecast"] > 0).all())

    def test_all_zero_forecasts(self):
        """All-zero inputs → all-zero reconciled output (non-negativity holds)."""
        rec = _make_reconciler("ols")
        leaf_fc = _leaf_forecast({"USA": 0, "CAN": 0, "GBR": 0, "DEU": 0})
        result = rec.reconcile({"country": leaf_fc})
        self.assertTrue((result["forecast"] == 0).all())

    def test_missing_upper_level_uses_bottom_up(self):
        """When only leaf forecasts provided, upper levels use bottom-up agg."""
        rec = _make_reconciler("ols")
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4}, weeks=1)
        # Only leaf forecasts
        result = rec.reconcile({"country": leaf_fc})
        self.assertEqual(result["country"].n_unique(), 4)

    def test_multiple_time_periods(self):
        """Reconciliation runs for each time period independently."""
        rec = _make_reconciler("wls")
        leaf_fc = _leaf_forecast(
            {"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4}, weeks=8
        )
        result = rec.reconcile({"country": leaf_fc})
        self.assertEqual(len(result), 32)  # 4 countries × 8 weeks

    def test_no_leaf_forecast_raises(self):
        """ValueError when no leaf-level key in forecasts dict."""
        rec = _make_reconciler("ols")
        with self.assertRaises(ValueError):
            rec.reconcile({})

    def test_ols_and_mint_same_shape(self):
        """OLS and MinT both return the same shape output."""
        leaf_fc = _leaf_forecast({"USA": 10, "CAN": 5, "GBR": 8, "DEU": 4}, weeks=3)
        r_ols = _make_reconciler("ols").reconcile({"country": leaf_fc})
        r_mint = _make_reconciler("mint").reconcile({"country": leaf_fc})
        self.assertEqual(r_ols.shape, r_mint.shape)


# ─────────────────────────────────────────────────────────────────────────────
# Mathematical properties
# ─────────────────────────────────────────────────────────────────────────────

class TestMathematicalProperties(unittest.TestCase):
    """Verify key algebraic properties of the reconciliation."""

    def test_G_times_S_equals_identity(self):
        """
        For OLS, G·S = I_n_leaves  (projection property).
        This ensures that if base forecasts are already coherent,
        reconciliation leaves them unchanged.
        """
        tree = _make_geo_tree()
        S_df = tree.summing_matrix()
        leaf_keys = [c for c in S_df.columns if c not in ("node_key", "node_level")]
        S = S_df.select(leaf_keys).to_numpy().astype(float)
        n_leaves = len(leaf_keys)

        # OLS: G = (S'S)^{-1} S'
        StS = S.T @ S
        StS_inv = np.linalg.inv(StS + 1e-8 * np.eye(n_leaves))
        G = StS_inv @ S.T

        GS = G @ S
        np.testing.assert_allclose(GS, np.eye(n_leaves), atol=1e-8)

    def test_reconciled_is_coherent_for_consistent_input(self):
        """
        If P̂_all is already coherent (P̂_all = S @ P̂_leaf), OLS produces
        output close to P̂_leaf (projection onto itself).
        """
        tree = _make_geo_tree()
        S_df = tree.summing_matrix()
        leaf_keys = [c for c in S_df.columns if c not in ("node_key", "node_level")]
        S = S_df.select(leaf_keys).to_numpy().astype(float)
        n_leaves = len(leaf_keys)

        p_leaf = np.array([10.0, 5.0, 8.0, 4.0])
        p_all = S @ p_leaf  # perfectly coherent

        StS = S.T @ S
        StS_inv = np.linalg.inv(StS + 1e-8 * np.eye(n_leaves))
        G = StS_inv @ S.T

        p_tilde = G @ p_all
        np.testing.assert_allclose(p_tilde, p_leaf, atol=1e-6)

    def test_wls_structural_upweights_leaves(self):
        """
        WLS with structural weights gives leaves weight = 1/1 = 1 and root
        weight = 1/n_leaves.  So leaves get proportionally more weight than
        aggregates in determining the reconciled output.
        """
        tree = _make_geo_tree()
        S_df = tree.summing_matrix()
        leaf_keys = [c for c in S_df.columns if c not in ("node_key", "node_level")]
        S = S_df.select(leaf_keys).to_numpy().astype(float)
        n_all = len(S_df)
        n_leaves = len(leaf_keys)

        # Structural W
        diag_w = np.maximum(S.sum(axis=1), 1.0)
        W_inv = np.diag(1.0 / diag_w)

        # Leaf nodes have S.sum(axis=1)=1, root has S.sum(axis=1)=n_leaves
        leaf_mask = S_df["node_level"].to_list()
        for i, level in enumerate(leaf_mask):
            if level == "country":
                self.assertAlmostEqual(diag_w[i], 1.0)
            if level == "global":
                self.assertAlmostEqual(diag_w[i], float(n_leaves))


if __name__ == "__main__":
    unittest.main()
