"use client";

import { useState, useEffect } from "react";
import { DataTable } from "@/components/data/data-table";
import { MetricCard } from "@/components/shared/metric-card";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { useAsyncOperation } from "@/hooks/use-async-operation";
import { api } from "@/lib/api-client";
import type { ModelCardListResponse } from "@/lib/types";

export function ModelCardsPanel() {
  const { result: data, loading, error, run, setError } = useAsyncOperation<ModelCardListResponse>();

  useEffect(() => {
    run(() => api.listModelCards());
  }, [run]);

  return (
    <section className="space-y-3 rounded-lg border p-6">
      <h2 className="text-lg font-semibold">Model Cards</h2>
      <p className="text-sm text-muted-foreground">
        Model card details: version, training window, series count, metrics, features, config hash.
      </p>
      {loading && <ChartSkeleton height="h-32" />}
      {error && (
        <ErrorDisplay
          message={error}
          onRetry={() => run(() => api.listModelCards())}
        />
      )}
      {data && (
        <>
          <MetricCard label="Total Model Cards" value={data.count} className="w-fit" />
          {data.model_cards.length > 0 ? (
            <DataTable
              columns={[
                { key: "model_name", label: "Model", sortable: true },
                { key: "lob", label: "LOB", sortable: true },
                { key: "n_series", label: "Series", sortable: true },
                { key: "n_observations", label: "Observations", sortable: true },
                { key: "backtest_wmape", label: "WMAPE", sortable: true,
                  render: (v) => v != null ? `${((v as number) * 100).toFixed(1)}%` : "N/A",
                },
                { key: "backtest_bias", label: "Bias", sortable: true,
                  render: (v) => v != null ? `${((v as number) * 100).toFixed(1)}%` : "N/A",
                },
                { key: "champion_since", label: "Champion Since", sortable: true },
                { key: "config_hash", label: "Config Hash", sortable: true,
                  render: (v) => (
                    <span className="font-mono text-xs">{String(v).slice(0, 8)}</span>
                  ),
                },
                { key: "notes", label: "Notes" },
              ]}
              data={data.model_cards as unknown as Record<string, unknown>[]}
              pageSize={10}
            />
          ) : (
            <p className="text-sm text-muted-foreground text-center py-4">No model cards available.</p>
          )}
        </>
      )}
    </section>
  );
}
