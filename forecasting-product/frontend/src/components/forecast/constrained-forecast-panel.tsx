"use client";

import { useState } from "react";
import { FileUpload } from "@/components/data/file-upload";
import { DataTable } from "@/components/data/data-table";
import { MetricCard } from "@/components/shared/metric-card";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { useAsyncOperation } from "@/hooks/use-async-operation";
import { api } from "@/lib/api-client";
import { formatNumber } from "@/lib/utils";
import type { ConstrainResponse } from "@/lib/types";

export function ConstrainedForecastPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [minDemand, setMinDemand] = useState<string>("0");
  const [maxCapacity, setMaxCapacity] = useState<string>("");
  const [aggregateMax, setAggregateMax] = useState<string>("");
  const [proportional, setProportional] = useState(false);
  const { result, loading, error, run, setError } = useAsyncOperation<ConstrainResponse>();

  const handleConstrain = () => {
    if (!file) return;
    const params: { min_demand?: number; max_capacity?: number; aggregate_max?: number; proportional?: boolean } = {};
    if (minDemand) params.min_demand = Number(minDemand);
    if (maxCapacity) params.max_capacity = Number(maxCapacity);
    if (aggregateMax) params.aggregate_max = Number(aggregateMax);
    params.proportional = proportional;
    run(() => api.constrainForecast(file, params));
  };

  return (
    <section className="space-y-4 rounded-lg border p-6">
      <h2 className="text-lg font-semibold">Constrained Forecast</h2>
      <p className="text-sm text-muted-foreground">
        Apply capacity and budget constraints to forecasts.
      </p>
      <FileUpload
        accept=".csv"
        onFileSelect={(f) => setFile(f)}
        label="Forecast File"
      />
      <div className="flex flex-wrap items-end gap-4">
        <div className="space-y-1">
          <label className="text-xs font-medium">Min Demand</label>
          <input
            type="number"
            value={minDemand}
            onChange={(e) => setMinDemand(e.target.value)}
            className="rounded-md border bg-background px-3 py-1.5 text-sm w-28"
          />
        </div>
        <div className="space-y-1">
          <label className="text-xs font-medium">Max Capacity</label>
          <input
            type="number"
            value={maxCapacity}
            onChange={(e) => setMaxCapacity(e.target.value)}
            placeholder="Optional"
            className="rounded-md border bg-background px-3 py-1.5 text-sm w-28"
          />
        </div>
        <div className="space-y-1">
          <label className="text-xs font-medium">Aggregate Max</label>
          <input
            type="number"
            value={aggregateMax}
            onChange={(e) => setAggregateMax(e.target.value)}
            placeholder="Optional"
            className="rounded-md border bg-background px-3 py-1.5 text-sm w-28"
          />
        </div>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={proportional}
            onChange={(e) => setProportional(e.target.checked)}
            className="rounded"
          />
          Proportional
        </label>
      </div>
      <button
        onClick={handleConstrain}
        disabled={!file || loading}
        className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
      >
        {loading ? "Applying Constraints..." : "Apply Constraints"}
      </button>
      {loading && <ChartSkeleton />}
      {error && <ErrorDisplay message={error} onRetry={() => setError(null)} />}
      {result && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricCard label="Before Total" value={formatNumber(result.before_total)} />
            <MetricCard label="After Total" value={formatNumber(result.after_total)} />
            <MetricCard label="Rows Modified" value={formatNumber(result.rows_modified)} />
            <MetricCard
              label="Change"
              value={result.before_total > 0 ? `${(((result.after_total - result.before_total) / result.before_total) * 100).toFixed(1)}%` : "N/A"}
            />
          </div>
          {result.constrained_preview.length > 0 && (
            <DataTable
              columns={Object.keys(result.constrained_preview[0]).map((k) => ({
                key: k,
                label: k,
                sortable: true,
              }))}
              data={result.constrained_preview}
              pageSize={10}
            />
          )}
        </div>
      )}
    </section>
  );
}
