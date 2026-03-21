"use client";

import { useState } from "react";
import { FileUpload } from "@/components/data/file-upload";
import { DataTable } from "@/components/data/data-table";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { useAsyncOperation } from "@/hooks/use-async-operation";
import { api } from "@/lib/api-client";
import type { ForecastCompareResponse } from "@/lib/types";

export function CrossRunComparisonPanel() {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [externalFile, setExternalFile] = useState<File | null>(null);
  const { result, loading, error, run, setError } = useAsyncOperation<ForecastCompareResponse>();

  const handleCompare = () => {
    if (!modelFile || !externalFile) return;
    run(() => api.compareForecasts(modelFile, externalFile));
  };

  return (
    <section className="space-y-4 rounded-lg border p-6">
      <h2 className="text-lg font-semibold">Cross-Run Comparison</h2>
      <p className="text-sm text-muted-foreground">
        Upload prior run forecasts for overlay comparison with gap analysis and summary metrics.
      </p>
      <div className="grid gap-4 sm:grid-cols-2">
        <FileUpload
          accept=".csv"
          onFileSelect={(file) => setModelFile(file)}
          label="Current Run Forecast"
        />
        <FileUpload
          accept=".csv"
          onFileSelect={(file) => setExternalFile(file)}
          label="Prior Run Forecast"
        />
      </div>
      <button
        onClick={handleCompare}
        disabled={!modelFile || !externalFile || loading}
        className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
      >
        {loading ? "Comparing..." : "Compare Runs"}
      </button>
      {loading && <ChartSkeleton />}
      {error && <ErrorDisplay message={error} onRetry={() => setError(null)} />}
      {result && (
        <div className="space-y-3">
          {result.summary.length > 0 && (
            <div>
              <h3 className="text-sm font-medium mb-2">Summary</h3>
              <DataTable
                columns={Object.keys(result.summary[0]).map((k) => ({
                  key: k,
                  label: k,
                  sortable: true,
                }))}
                data={result.summary}
                pageSize={10}
              />
            </div>
          )}
          {result.comparison.length > 0 && (
            <div>
              <h3 className="text-sm font-medium mb-2">Comparison Detail</h3>
              <DataTable
                columns={Object.keys(result.comparison[0]).map((k) => ({
                  key: k,
                  label: k,
                  sortable: true,
                }))}
                data={result.comparison}
                pageSize={10}
              />
            </div>
          )}
        </div>
      )}
    </section>
  );
}
