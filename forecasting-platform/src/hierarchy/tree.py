"""
Generic hierarchy tree built from data + config.

The tree is constructed from a Polars DataFrame containing hierarchy columns.
It supports any number of levels and any naming — the level names come from
config, the actual node values come from data.

Example
-------
Given config levels ``["region", "subregion", "country"]`` and data::

    region   | subregion | country
    Americas | NA        | USA
    Americas | NA        | CAN
    EMEA     | WE        | GBR
    EMEA     | WE        | DEU
    EMEA     | NE        | NOR

The tree looks like::

    __root__
    ├── Americas
    │   └── NA
    │       ├── USA
    │       └── CAN
    └── EMEA
        ├── WE
        │   ├── GBR
        │   └── DEU
        └── NE
            └── NOR
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import polars as pl

from ..config.schema import HierarchyConfig


@dataclass
class HierarchyNode:
    """A single node in the hierarchy tree."""
    key: str
    level: str
    parent: Optional[HierarchyNode] = field(default=None, repr=False)
    children: List[HierarchyNode] = field(default_factory=list, repr=False)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def descendants(self, level: Optional[str] = None) -> List[HierarchyNode]:
        """All descendants, optionally filtered to a specific level."""
        result: List[HierarchyNode] = []
        for child in self.children:
            if level is None or child.level == level:
                result.append(child)
            result.extend(child.descendants(level))
        return result

    def ancestors(self) -> List[HierarchyNode]:
        """Walk up to root, excluding self."""
        chain: List[HierarchyNode] = []
        current = self.parent
        while current is not None:
            chain.append(current)
            current = current.parent
        return chain

    def leaf_descendants(self) -> List[HierarchyNode]:
        """All leaf nodes under this node."""
        if self.is_leaf:
            return [self]
        leaves: List[HierarchyNode] = []
        for child in self.children:
            leaves.extend(child.leaf_descendants())
        return leaves


class HierarchyTree:
    """
    A single hierarchy dimension (product, geography, or channel).

    Built from a ``HierarchyConfig`` and the actual data.  The tree is
    immutable once constructed.  All query operations are O(tree size) or
    better — there is no database.
    """

    def __init__(self, config: HierarchyConfig, data: pl.DataFrame):
        """
        Parameters
        ----------
        config:
            Hierarchy definition from platform config.
        data:
            DataFrame that contains at least the columns named in
            ``config.levels``.  Rows are distinct combinations observed in
            the data.
        """
        self.config = config
        self.name = config.name
        self.levels: List[str] = list(config.levels)

        # Validate that all level columns exist
        missing = set(self.levels) - set(data.columns)
        if missing:
            raise ValueError(
                f"Hierarchy {self.name!r}: columns {missing} not found in data. "
                f"Available: {data.columns}"
            )

        # Build the tree
        self._root = HierarchyNode(key="__root__", level="__root__")
        self._nodes: Dict[str, Dict[str, HierarchyNode]] = {}  # level → {key → node}

        self._build(data)

    @property
    def root(self) -> HierarchyNode:
        return self._root

    @property
    def leaf_level(self) -> str:
        return self.levels[-1]

    def get_nodes(self, level: str) -> List[HierarchyNode]:
        """All nodes at a given level."""
        if level not in self._nodes:
            raise KeyError(
                f"Level {level!r} not in hierarchy {self.name!r}. "
                f"Available: {list(self._nodes.keys())}"
            )
        return list(self._nodes[level].values())

    def get_node(self, level: str, key: str) -> HierarchyNode:
        """Look up a specific node by level and key."""
        return self._nodes[level][key]

    def get_leaves(self) -> List[HierarchyNode]:
        """All leaf-level nodes."""
        return self.get_nodes(self.leaf_level)

    def get_parent_child_map(
        self, parent_level: str, child_level: str
    ) -> Dict[str, List[str]]:
        """
        Return a mapping from parent keys to child keys.

        The parent and child levels do not need to be adjacent — this
        supports skip-level relationships (e.g. region → country, skipping
        subregion).
        """
        parent_idx = self.levels.index(parent_level)
        child_idx = self.levels.index(child_level)
        if child_idx <= parent_idx:
            raise ValueError(
                f"Child level {child_level!r} must be below parent "
                f"level {parent_level!r} in hierarchy {self.name!r}"
            )

        result: Dict[str, List[str]] = {}
        for parent_node in self.get_nodes(parent_level):
            children = parent_node.descendants(level=child_level)
            result[parent_node.key] = [c.key for c in children]
        return result

    def summing_matrix(self) -> pl.DataFrame:
        """
        Build the S (summing) matrix for reconciliation.

        Rows = all nodes at all levels (top → bottom).
        Columns = leaf-level nodes.
        S[i, j] = 1 if leaf j aggregates into node i, else 0.

        Returns a Polars DataFrame with columns: node_key, node_level,
        plus one boolean column per leaf node.
        """
        leaves = self.get_leaves()
        leaf_keys = [n.key for n in leaves]

        rows = []
        for level in self.levels:
            for node in self.get_nodes(level):
                leaf_set: Set[str] = {
                    n.key for n in node.leaf_descendants()
                }
                row = {
                    "node_key": node.key,
                    "node_level": level,
                }
                for lk in leaf_keys:
                    row[lk] = 1.0 if lk in leaf_set else 0.0
                rows.append(row)

        return pl.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build(self, data: pl.DataFrame) -> None:
        """Construct the tree from the data."""
        # De-duplicate to get distinct hierarchy paths
        paths = data.select(self.levels).unique()

        # Initialise level dicts
        for level in self.levels:
            self._nodes[level] = {}

        for row in paths.iter_rows(named=True):
            parent_node = self._root
            for level in self.levels:
                key = str(row[level])
                if key not in self._nodes[level]:
                    node = HierarchyNode(
                        key=key, level=level, parent=parent_node
                    )
                    self._nodes[level][key] = node
                    parent_node.children.append(node)

                # Ensure parent linkage is consistent
                node = self._nodes[level][key]
                parent_node = node

    def __repr__(self) -> str:
        n_nodes = sum(len(v) for v in self._nodes.values())
        return (
            f"HierarchyTree(name={self.name!r}, levels={self.levels}, "
            f"nodes={n_nodes})"
        )
