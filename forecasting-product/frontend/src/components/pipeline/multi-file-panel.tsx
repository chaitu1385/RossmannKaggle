"use client";

import { useState } from "react";
import { DataTable } from "@/components/data/data-table";
import { MetricCard } from "@/components/shared/metric-card";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { useAsyncOperation } from "@/hooks/use-async-operation";
import { api } from "@/lib/api-client";
import { formatNumber } from "@/lib/utils";
import type { MultiFileAnalysisResponse } from "@/lib/types";

export function MultiFilePanel({ lobName }: { lobName: string }) {
  const [files, setFiles] = useState<File[]>([]);
  const { result, loading, error, run, setError } = useAsyncOperation<MultiFileAnalysisResponse>();

  const handleFilesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));
    }
  };

  const handleAnalyze = () => {
    if (files.length === 0) return;
    run(() => api.analyzeMultiFile(files, lobName));
  };

  return (
    <section className="space-y-4 rounded-lg border p-6">
      <h2 className="text-lg font-semibold">Multi-File Classification &amp; Merge</h2>
      <p className="text-sm text-muted-foreground">
        Upload multiple files for auto-detection of roles (time_series, dimension, regressor) and merge preview.
      </p>
      <div className="flex flex-wrap items-end gap-4">
        <div className="space-y-1">
          <label className="text-xs font-medium">Select Files</label>
          <input
            type="file"
            accept=".csv,.parquet"
            multiple
            onChange={handleFilesChange}
            className="rounded-md border bg-background px-3 py-1.5 text-sm"
          />
        </div>
        <button
          onClick={handleAnalyze}
          disabled={files.length === 0 || loading}
          className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
        >
          {loading ? "Analyzing..." : "Analyze Files"}
        </button>
      </div>
      {files.length > 0 && (
        <p className="text-xs text-muted-foreground">{files.length} file(s) selected: {files.map(f => f.name).join(", ")}</p>
      )}
      {loading && <ChartSkeleton height="h-32" />}
      {error && <ErrorDisplay message={error} onRetry={() => setError(null)} />}
      {result && (
        <div className="space-y-4">
          <h3 className="text-sm font-medium">File Profiles</h3>
          <DataTable
            columns={[
              { key: "filename", label: "File", sortable: true },
              { key: "role", label: "Role", sortable: true },
              { key: "confidence", label: "Confidence", sortable: true, render: (v) => `${((v as number) * 100).toFixed(0)}%` },
              { key: "n_rows", label: "Rows", sortable: true },
              { key: "n_columns", label: "Columns", sortable: true },
              { key: "time_column", label: "Time Col" },
            ]}
            data={result.profiles as unknown as Record<string, unknown>[]}
            pageSize={10}
          />
          {result.primary_file && (
            <p className="text-sm">Primary file: <span className="font-mono font-medium">{result.primary_file}</span></p>
          )}
          {result.warnings.length > 0 && (
            <div className="rounded-md bg-yellow-50 dark:bg-yellow-950 p-3">
              <h4 className="text-xs font-medium text-yellow-800 dark:text-yellow-200">Warnings</h4>
              <ul className="mt-1 text-xs text-yellow-700 dark:text-yellow-300 space-y-0.5">
                {result.warnings.map((w, i) => <li key={i}>&bull; {w}</li>)}
              </ul>
            </div>
          )}
          {result.merge_preview && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium">Merge Preview</h3>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <MetricCard label="Total Rows" value={formatNumber(result.merge_preview.total_rows)} />
                <MetricCard label="Total Columns" value={result.merge_preview.total_columns} />
                <MetricCard label="Matched Rows" value={formatNumber(result.merge_preview.matched_rows)} />
                <MetricCard label="Unmatched Keys" value={result.merge_preview.unmatched_primary_keys} />
              </div>
              {result.merge_preview.sample_rows.length > 0 && (
                <DataTable
                  columns={Object.keys(result.merge_preview.sample_rows[0]).map(k => ({ key: k, label: k, sortable: true }))}
                  data={result.merge_preview.sample_rows}
                  pageSize={5}
                />
              )}
            </div>
          )}
          {result.merge_error && (
            <ErrorDisplay message={result.merge_error} />
          )}
        </div>
      )}
    </section>
  );
}
