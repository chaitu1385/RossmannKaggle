"use client";

import { useEffect } from "react";
import { DataTable } from "@/components/data/data-table";
import { MetricCard } from "@/components/shared/metric-card";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { useAsyncOperation } from "@/hooks/use-async-operation";
import { api } from "@/lib/api-client";
import type { LineageResponse } from "@/lib/types";

export function LineagePanel({ lob }: { lob: string }) {
  const { result: data, loading, error, run } = useAsyncOperation<LineageResponse>();

  useEffect(() => {
    run(() => api.getLineage(lob));
  }, [lob, run]);

  return (
    <section className="space-y-3 rounded-lg border p-6">
      <h2 className="text-lg font-semibold">Forecast Lineage</h2>
      <p className="text-sm text-muted-foreground">
        Trace forecast provenance: data source, preprocessing, model, postprocessing, output.
      </p>
      {loading && <ChartSkeleton height="h-32" />}
      {error && (
        <ErrorDisplay
          message={error}
          onRetry={() => run(() => api.getLineage(lob))}
        />
      )}
      {data && (
        <>
          <MetricCard label="Lineage Entries" value={data.count} className="w-fit" />
          {data.lineage.length > 0 ? (
            <DataTable
              columns={[
                { key: "timestamp", label: "Timestamp", sortable: true },
                { key: "model_id", label: "Model ID", sortable: true },
                { key: "lob", label: "LOB", sortable: true },
                { key: "n_series", label: "Series", sortable: true },
                { key: "horizon_weeks", label: "Horizon", sortable: true },
                { key: "selection_strategy", label: "Strategy", sortable: true },
                { key: "run_id", label: "Run ID", sortable: true },
                { key: "user_id", label: "User", sortable: true },
                { key: "notes", label: "Notes" },
              ]}
              data={data.lineage as unknown as Record<string, unknown>[]}
              pageSize={10}
            />
          ) : (
            <p className="text-sm text-muted-foreground text-center py-4">No lineage data available for this LOB.</p>
          )}
        </>
      )}
    </section>
  );
}
