"use client";

import { ComingSoon } from "@/components/shared/coming-soon";
import { MetricCard } from "@/components/shared/metric-card";

const RECONCILIATION_METHODS = [
  { value: "bottom_up", label: "Bottom-Up", description: "Aggregate leaf forecasts upward" },
  { value: "top_down", label: "Top-Down", description: "Disaggregate top-level forecast using proportions" },
  { value: "middle_out", label: "Middle-Out", description: "Reconcile from a chosen middle level" },
  { value: "ols", label: "OLS", description: "Ordinary Least Squares regression-based reconciliation" },
  { value: "wls", label: "WLS", description: "Weighted Least Squares using variance scaling" },
  { value: "mint", label: "MinT", description: "Minimum Trace — optimal reconciliation using error covariance" },
];

export default function HierarchyPage() {
  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Hierarchy Manager</h1>
        <p className="text-sm text-muted-foreground">
          Visualize hierarchy structure, aggregate data across levels, and run reconciliation.
        </p>
      </div>

      {/* Demo overview */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricCard label="Levels" value="4" />
        <MetricCard label="Total Nodes" value="156" />
        <MetricCard label="Leaf Nodes" value="120" />
        <MetricCard label="Method" value="MinT" />
      </div>

      {/* Hierarchy Tree */}
      <ComingSoon
        feature="Hierarchy Sunburst Visualization"
        description="Interactive sunburst chart showing hierarchy tree with drill-down, node count per level, and leaf count"
      />

      {/* Reconciliation Methods Reference */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Reconciliation Methods</h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {RECONCILIATION_METHODS.map((m) => (
            <div key={m.value} className="rounded-lg border p-4">
              <h3 className="font-semibold text-sm">{m.label}</h3>
              <p className="mt-1 text-xs text-muted-foreground">{m.description}</p>
              <code className="mt-2 block text-xs text-primary">{m.value}</code>
            </div>
          ))}
        </div>
      </section>

      {/* Placeholders */}
      <div className="grid gap-4 sm:grid-cols-2">
        <ComingSoon
          feature="Aggregation Explorer"
          description="Select target level, view aggregated time series for top-N nodes with drill-down"
        />
        <ComingSoon
          feature="Reconciliation Execution"
          description="Run reconciliation with selected method, view before/after comparison and coherence check"
        />
      </div>
    </div>
  );
}
