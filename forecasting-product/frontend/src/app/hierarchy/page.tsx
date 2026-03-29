"use client";

import { useState, useRef } from "react";
import { api } from "@/lib/api-client";
import { MetricCard } from "@/components/shared/metric-card";
import { HierarchySunburst } from "@/components/charts/hierarchy-sunburst";
import type {
  HierarchyBuildResponse,
  HierarchyAggregateResponse,
  HierarchyReconcileResponse,
} from "@/lib/types";

const RECONCILIATION_METHODS = [
  { value: "bottom_up", label: "Bottom-Up", description: "Aggregate leaf forecasts upward" },
  { value: "top_down", label: "Top-Down", description: "Disaggregate top-level forecast using proportions" },
  { value: "middle_out", label: "Middle-Out", description: "Reconcile from a chosen middle level" },
  { value: "ols", label: "OLS", description: "Ordinary Least Squares regression-based reconciliation" },
  { value: "wls", label: "WLS", description: "Weighted Least Squares using variance scaling" },
  { value: "mint", label: "MinT", description: "Minimum Trace — optimal reconciliation using error covariance" },
];

function GenericTable({ data }: { data: Record<string, unknown>[] }) {
  if (data.length === 0) return <p className="text-sm text-muted-foreground">No data.</p>;
  const cols = Object.keys(data[0]);
  return (
    <div className="overflow-x-auto rounded-md border">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b bg-muted/50 text-left text-xs text-muted-foreground">
            {cols.map((c) => (
              <th key={c} className="px-3 py-2">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i} className="border-b last:border-0">
              {cols.map((c) => (
                <td key={c} className="px-3 py-2">{String(row[c] ?? "")}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function HierarchyPage() {
  // Shared state
  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [levels, setLevels] = useState("");

  // Build hierarchy state
  const [buildResult, setBuildResult] = useState<HierarchyBuildResponse | null>(null);
  const [buildLoading, setBuildLoading] = useState(false);
  const [buildError, setBuildError] = useState<string | null>(null);

  // Aggregation state
  const [targetLevel, setTargetLevel] = useState("");
  const [aggResult, setAggResult] = useState<HierarchyAggregateResponse | null>(null);
  const [aggLoading, setAggLoading] = useState(false);
  const [aggError, setAggError] = useState<string | null>(null);

  // Reconciliation state
  const [reconMethod, setReconMethod] = useState("bottom_up");
  const [reconResult, setReconResult] = useState<HierarchyReconcileResponse | null>(null);
  const [reconLoading, setReconLoading] = useState(false);
  const [reconError, setReconError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0] ?? null;
    setFile(selected);
    // Reset downstream results when file changes
    setBuildResult(null);
    setAggResult(null);
    setReconResult(null);
  };

  const handleBuild = async () => {
    if (!file || !levels.trim()) return;
    setBuildLoading(true);
    setBuildError(null);
    setBuildResult(null);
    try {
      const result = await api.buildHierarchy(file, levels.trim());
      setBuildResult(result);
      // Default target level to first level if available
      if (result.levels.length > 0 && !targetLevel) {
        setTargetLevel(result.levels[0]);
      }
    } catch (err) {
      setBuildError(err instanceof Error ? err.message : "Failed to build hierarchy");
    } finally {
      setBuildLoading(false);
    }
  };

  const handleAggregate = async () => {
    if (!file || !levels.trim() || !targetLevel) return;
    setAggLoading(true);
    setAggError(null);
    setAggResult(null);
    try {
      const result = await api.aggregateHierarchy(file, levels.trim(), targetLevel);
      setAggResult(result);
    } catch (err) {
      setAggError(err instanceof Error ? err.message : "Failed to aggregate hierarchy");
    } finally {
      setAggLoading(false);
    }
  };

  const handleReconcile = async () => {
    if (!file || !levels.trim()) return;
    setReconLoading(true);
    setReconError(null);
    setReconResult(null);
    try {
      const result = await api.reconcileHierarchy(file, levels.trim(), reconMethod);
      setReconResult(result);
    } catch (err) {
      setReconError(err instanceof Error ? err.message : "Failed to reconcile hierarchy");
    } finally {
      setReconLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Hierarchy Manager</h1>
        <p className="text-sm text-muted-foreground">
          Visualize hierarchy structure, aggregate data across levels, and run reconciliation.
        </p>
      </div>

      {/* Overview metrics */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricCard label="Levels" value={buildResult ? String(buildResult.levels.length) : "--"} />
        <MetricCard label="Total Nodes" value={buildResult ? String(buildResult.total_nodes) : "--"} />
        <MetricCard label="Leaf Nodes" value={buildResult ? String(buildResult.leaf_count) : "--"} />
        <MetricCard label="S-Matrix" value={buildResult ? `${buildResult.s_matrix_shape[0]}x${buildResult.s_matrix_shape[1]}` : "--"} />
      </div>

      {/* Shared file + levels input */}
      <section className="space-y-3 rounded-lg border p-4">
        <h2 className="text-lg font-semibold">Data Input</h2>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
          <div className="flex-1">
            <label className="mb-1 block text-sm font-medium">Upload CSV</label>
            <input
              ref={fileRef}
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="block w-full rounded-md border px-3 py-2 text-sm file:mr-3 file:rounded file:border-0 file:bg-primary/10 file:px-3 file:py-1 file:text-sm file:font-medium"
            />
          </div>
          <div className="flex-1">
            <label className="mb-1 block text-sm font-medium">Hierarchy Levels (comma-separated)</label>
            <input
              type="text"
              value={levels}
              onChange={(e) => setLevels(e.target.value)}
              placeholder="e.g. category,subcategory,brand"
              className="block w-full rounded-md border px-3 py-2 text-sm"
            />
          </div>
        </div>
      </section>

      {/* Build Hierarchy Panel */}
      <section className="space-y-3 rounded-lg border p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Build Hierarchy</h2>
          <button
            onClick={handleBuild}
            disabled={buildLoading || !file || !levels.trim()}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
          >
            {buildLoading ? "Building..." : "Build Hierarchy"}
          </button>
        </div>
        {buildError && (
          <div className="rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">{buildError}</div>
        )}
        {buildResult && (
          <div className="space-y-3">
            <h3 className="text-sm font-semibold">Level Stats</h3>
            <GenericTable
              data={buildResult.level_stats.map((ls) => ({
                Level: ls.level,
                "Node Count": ls.node_count,
              }))}
            />
            <div className="grid grid-cols-3 gap-3 text-sm">
              <div className="rounded-md border p-3">
                <span className="text-muted-foreground">Total Nodes</span>
                <p className="text-lg font-semibold">{buildResult.total_nodes}</p>
              </div>
              <div className="rounded-md border p-3">
                <span className="text-muted-foreground">Leaf Count</span>
                <p className="text-lg font-semibold">{buildResult.leaf_count}</p>
              </div>
              <div className="rounded-md border p-3">
                <span className="text-muted-foreground">S-Matrix Shape</span>
                <p className="text-lg font-semibold">{buildResult.s_matrix_shape[0]} x {buildResult.s_matrix_shape[1]}</p>
              </div>
            </div>
            {/* Sunburst visualization */}
            {buildResult.tree_nodes && buildResult.tree_nodes.length > 0 && (
              <div>
                <h3 className="text-sm font-semibold mb-2">Hierarchy Tree</h3>
                <div className="rounded-lg border p-2">
                  <HierarchySunburst nodes={buildResult.tree_nodes} height={400} />
                </div>
              </div>
            )}
          </div>
        )}
      </section>

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

      {/* Aggregation + Reconciliation panels side by side */}
      <div className="grid gap-4 sm:grid-cols-2">
        {/* Aggregation Panel */}
        <section className="space-y-3 rounded-lg border p-4">
          <h2 className="text-lg font-semibold">Aggregation Explorer</h2>
          <div className="space-y-2">
            <label className="block text-sm font-medium">Target Level</label>
            <select
              value={targetLevel}
              onChange={(e) => setTargetLevel(e.target.value)}
              className="block w-full rounded-md border px-3 py-2 text-sm"
            >
              <option value="" disabled>
                {buildResult ? "Select a level" : "Build hierarchy first"}
              </option>
              {(buildResult?.levels ?? []).map((lvl) => (
                <option key={lvl} value={lvl}>{lvl}</option>
              ))}
            </select>
          </div>
          <button
            onClick={handleAggregate}
            disabled={aggLoading || !file || !levels.trim() || !targetLevel}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
          >
            {aggLoading ? "Aggregating..." : "Aggregate"}
          </button>
          {aggError && (
            <div className="rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">{aggError}</div>
          )}
          {aggResult && (
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">
                Level: <strong>{aggResult.target_level}</strong> | Rows: {aggResult.total_rows} | Unique Nodes: {aggResult.unique_nodes}
              </p>
              <GenericTable data={aggResult.top_n_data} />
            </div>
          )}
        </section>

        {/* Reconciliation Panel */}
        <section className="space-y-3 rounded-lg border p-4">
          <h2 className="text-lg font-semibold">Reconciliation Execution</h2>
          <div className="space-y-2">
            <label className="block text-sm font-medium">Method</label>
            <select
              value={reconMethod}
              onChange={(e) => setReconMethod(e.target.value)}
              className="block w-full rounded-md border px-3 py-2 text-sm"
            >
              {RECONCILIATION_METHODS.filter((m) => m.value !== "middle_out").map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
          </div>
          <button
            onClick={handleReconcile}
            disabled={reconLoading || !file || !levels.trim()}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
          >
            {reconLoading ? "Reconciling..." : "Reconcile"}
          </button>
          {reconError && (
            <div className="rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">{reconError}</div>
          )}
          {reconResult && (
            <div className="space-y-2">
              <div className="flex gap-3 text-sm">
                <div className="rounded-md border p-2 flex-1">
                  <span className="text-muted-foreground">Before Total</span>
                  <p className="font-semibold">{reconResult.before_total.toLocaleString()}</p>
                </div>
                <div className="rounded-md border p-2 flex-1">
                  <span className="text-muted-foreground">After Total</span>
                  <p className="font-semibold">{reconResult.after_total.toLocaleString()}</p>
                </div>
                <div className="rounded-md border p-2 flex-1">
                  <span className="text-muted-foreground">Rows</span>
                  <p className="font-semibold">{reconResult.rows}</p>
                </div>
              </div>
              <GenericTable data={reconResult.reconciled_preview} />
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
