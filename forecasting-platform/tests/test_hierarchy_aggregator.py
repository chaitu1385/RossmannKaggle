"""
Tests for HierarchyAggregator — src/hierarchy/aggregator.py.

Covers:
  - aggregate_to with sum and mean
  - disaggregate_to with equal split and explicit proportions
  - round-trip aggregate → disaggregate preserves totals
  - compute_historical_proportions sums to 1.0 per parent
  - n_recent_weeks filtering
  - invalid agg raises ValueError
"""

import unittest
from datetime import date, timedelta

import polars as pl
import pytest


def _make_geo_tree():
    """
    Three-level geography hierarchy:
        global → region → country
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


def _make_leaf_data(weeks: int = 8):
    """Build country-level weekly data."""
    values = {"USA": 100.0, "CAN": 50.0, "GBR": 80.0, "DEU": 60.0}
    rows = []
    for w in range(weeks):
        for country, val in values.items():
            rows.append({
                "country": country,
                "week": date(2024, 1, 1) + timedelta(weeks=w),
                "quantity": val + w * 2,
            })
    return pl.DataFrame(rows).with_columns(pl.col("quantity").cast(pl.Float64))


class TestAggregateSum(unittest.TestCase):

    def test_aggregate_to_region_sum(self):
        from src.hierarchy.aggregator import HierarchyAggregator
        tree = _make_geo_tree()
        agg = HierarchyAggregator(tree)
        df = _make_leaf_data(weeks=4)

        result = agg.aggregate_to(df, "region", ["quantity"])
        self.assertIn("region", result.columns)
        self.assertIn("quantity", result.columns)
        # 2 regions x 4 weeks = 8 rows
        self.assertEqual(result.shape[0], 8)

        # Check NA sum for week 0: USA(100) + CAN(50) = 150
        na_w0 = result.filter(
            (pl.col("region") == "NA")
            & (pl.col("week") == date(2024, 1, 1))
        )
        self.assertAlmostEqual(na_w0["quantity"][0], 150.0)

    def test_aggregate_to_global_sum(self):
        from src.hierarchy.aggregator import HierarchyAggregator
        tree = _make_geo_tree()
        agg = HierarchyAggregator(tree)
        df = _make_leaf_data(weeks=2)

        result = agg.aggregate_to(df, "global", ["quantity"])
        self.assertEqual(result.shape[0], 2)  # 1 global x 2 weeks
        # week 0: 100+50+80+60 = 290
        w0 = result.filter(pl.col("week") == date(2024, 1, 1))
        self.assertAlmostEqual(w0["quantity"][0], 290.0)


class TestAggregateMean(unittest.TestCase):

    def test_aggregate_mean(self):
        from src.hierarchy.aggregator import HierarchyAggregator
        tree = _make_geo_tree()
        agg = HierarchyAggregator(tree)
        df = _make_leaf_data(weeks=2)

        result = agg.aggregate_to(df, "region", ["quantity"], agg="mean")
        na_w0 = result.filter(
            (pl.col("region") == "NA")
            & (pl.col("week") == date(2024, 1, 1))
        )
        # mean of USA(100) + CAN(50) = 75
        self.assertAlmostEqual(na_w0["quantity"][0], 75.0)


class TestAggregateInvalidAgg(unittest.TestCase):

    def test_invalid_agg_raises(self):
        from src.hierarchy.aggregator import HierarchyAggregator
        tree = _make_geo_tree()
        agg = HierarchyAggregator(tree)
        df = _make_leaf_data(weeks=2)
        with self.assertRaises(ValueError):
            agg.aggregate_to(df, "region", ["quantity"], agg="median")


class TestDisaggregate(unittest.TestCase):

    def test_equal_split(self):
        from src.hierarchy.aggregator import HierarchyAggregator
        tree = _make_geo_tree()
        agg = HierarchyAggregator(tree)

        region_data = pl.DataFrame({
            "region": ["NA", "EMEA"],
            "week": [date(2024, 1, 1)] * 2,
            "quantity": [200.0, 100.0],
        })
        result = agg.disaggregate_to(
            region_data, "region", "country", ["quantity"]
        )
        self.assertIn("country", result.columns)
        # NA → USA, CAN equally: 100 each
        usa = result.filter(pl.col("country") == "USA")
        self.assertAlmostEqual(usa["quantity"][0], 100.0)

    def test_with_proportions(self):
        from src.hierarchy.aggregator import HierarchyAggregator
        tree = _make_geo_tree()
        agg = HierarchyAggregator(tree)

        region_data = pl.DataFrame({
            "region": ["NA"],
            "week": [date(2024, 1, 1)],
            "quantity": [300.0],
        })
        props = pl.DataFrame({
            "region": ["NA", "NA"],
            "country": ["USA", "CAN"],
            "proportion": [0.7, 0.3],
        })
        result = agg.disaggregate_to(
            region_data, "region", "country", ["quantity"],
            proportions=props,
        )
        usa = result.filter(pl.col("country") == "USA")
        can = result.filter(pl.col("country") == "CAN")
        self.assertAlmostEqual(usa["quantity"][0], 210.0)
        self.assertAlmostEqual(can["quantity"][0], 90.0)

    def test_disaggregate_preserves_total(self):
        from src.hierarchy.aggregator import HierarchyAggregator
        tree = _make_geo_tree()
        agg = HierarchyAggregator(tree)

        region_data = pl.DataFrame({
            "region": ["NA", "EMEA"],
            "week": [date(2024, 1, 1)] * 2,
            "quantity": [200.0, 100.0],
        })
        result = agg.disaggregate_to(
            region_data, "region", "country", ["quantity"]
        )
        total = result["quantity"].sum()
        self.assertAlmostEqual(total, 300.0)


class TestRoundTrip(unittest.TestCase):

    def test_aggregate_then_disaggregate(self):
        from src.hierarchy.aggregator import HierarchyAggregator
        tree = _make_geo_tree()
        agg_obj = HierarchyAggregator(tree)
        leaf_data = _make_leaf_data(weeks=2)

        # Aggregate to region
        region_agg = agg_obj.aggregate_to(leaf_data, "region", ["quantity"])
        # Disaggregate back with equal split
        back = agg_obj.disaggregate_to(
            region_agg, "region", "country", ["quantity"]
        )
        # Total should be preserved
        original_total = leaf_data["quantity"].sum()
        round_trip_total = back["quantity"].sum()
        self.assertAlmostEqual(round_trip_total, original_total, places=2)


class TestComputeHistoricalProportions(unittest.TestCase):

    def test_proportions_sum_to_one(self):
        from src.hierarchy.aggregator import HierarchyAggregator
        tree = _make_geo_tree()
        agg = HierarchyAggregator(tree)
        df = _make_leaf_data(weeks=8)

        props = agg.compute_historical_proportions(
            df, "region", "country", "quantity"
        )
        self.assertIn("proportion", props.columns)
        # Proportions per region sum to 1.0
        for region in ["NA", "EMEA"]:
            region_props = props.filter(pl.col("region") == region)
            total = region_props["proportion"].sum()
            self.assertAlmostEqual(total, 1.0, places=5)

    def test_proportions_reflect_data(self):
        from src.hierarchy.aggregator import HierarchyAggregator
        tree = _make_geo_tree()
        agg = HierarchyAggregator(tree)
        df = _make_leaf_data(weeks=1)  # USA=100, CAN=50

        props = agg.compute_historical_proportions(
            df, "region", "country", "quantity"
        )
        na = props.filter(pl.col("region") == "NA")
        usa_prop = na.filter(pl.col("country") == "USA")["proportion"][0]
        # USA / (USA+CAN) = 100/150 ≈ 0.667
        self.assertAlmostEqual(usa_prop, 100.0 / 150.0, places=3)

    def test_n_recent_weeks_filter(self):
        from src.hierarchy.aggregator import HierarchyAggregator
        tree = _make_geo_tree()
        agg = HierarchyAggregator(tree)
        # Make data with dates as Datetime to support duration arithmetic
        df = _make_leaf_data(weeks=8)
        df = df.with_columns(pl.col("week").cast(pl.Datetime))

        props = agg.compute_historical_proportions(
            df, "region", "country", "quantity",
            time_column="week",
            n_recent_weeks=4,
        )
        # Should still produce proportions
        self.assertGreater(props.shape[0], 0)
        for region in ["NA", "EMEA"]:
            region_props = props.filter(pl.col("region") == region)
            total = region_props["proportion"].sum()
            self.assertAlmostEqual(total, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
