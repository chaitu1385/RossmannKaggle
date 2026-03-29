"use client";

import dynamic from "next/dynamic";
import type { HierarchyTreeNode } from "@/lib/types";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface Props {
  nodes: HierarchyTreeNode[];
  height?: number;
}

export function HierarchySunburst({ nodes, height = 420 }: Props) {
  if (nodes.length === 0) return null;

  // Plotly sunburst expects: labels[], parents[], values[]
  const labels = nodes.map((n) => n.key);
  const parents = nodes.map((n) => n.parent);

  // Count children per node to size non-leaf nodes
  const childCount = new Map<string, number>();
  for (const n of nodes) {
    if (n.is_leaf) {
      // Walk up the chain incrementing counts
      let current = n.parent;
      while (current) {
        childCount.set(current, (childCount.get(current) || 0) + 1);
        const parentNode = nodes.find((p) => p.key === current);
        current = parentNode?.parent || "";
        if (!current) break;
      }
    }
  }

  // Leaf nodes get value 1, non-leaf nodes are sized by number of leaves below
  const values = nodes.map((n) =>
    n.is_leaf ? 1 : (childCount.get(n.key) || 1)
  );

  return (
    <Plot
      data={[
        {
          type: "sunburst",
          labels,
          parents,
          values,
          branchvalues: "total",
          textinfo: "label",
          hovertemplate: "<b>%{label}</b><br>Children: %{value}<extra></extra>",
        } as Plotly.Data,
      ]}
      layout={{
        height,
        margin: { t: 10, r: 10, b: 10, l: 10 },
        paper_bgcolor: "transparent",
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: "100%" }}
    />
  );
}
