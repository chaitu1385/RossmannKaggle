"use client";

import { useState } from "react";
import { FileUpload } from "@/components/data/file-upload";
import { DataTable } from "@/components/data/data-table";
import { MetricCard } from "@/components/shared/metric-card";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { useAsyncOperation } from "@/hooks/use-async-operation";
import { api } from "@/lib/api-client";
import type { PipelineRunResponse } from "@/lib/types";

export function PipelineExecutionPanel({ lobName }: { lobName: string }) {
  const [pipelineFile, setPipelineFile] = useState<File | null>(null);
  const backtest = useAsyncOperation<PipelineRunResponse>();
  const forecast = useAsyncOperation<PipelineRunResponse>();
  const [error, setError] = useState<string | null>(null);

  const handleRunBacktest = () => {
    if (!pipelineFile) return;
    setError(null);
    backtest.run(() => api.runBacktest(pipelineFile, lobName)).catch((err) =>
      setError(err instanceof Error ? err.message : "Backtest failed")
    );
  };

  const handleRunForecast = () => {
    if (!pipelineFile) return;
    setError(null);
    forecast.run(() => api.runForecast(pipelineFile, lobName)).catch((err) =>
      setError(err instanceof Error ? err.message : "Forecast failed")
    );
  };

  const combinedError = error || backtest.error || forecast.error;

  return (
    <section className="space-y-4 rounded-lg border p-6">
      <h2 className="text-lg font-semibold">Pipeline Execution</h2>
      <p className="text-sm text-muted-foreground">
        Run Backtest and Forecast pipelines directly from the UI.
      </p>
      <FileUpload
        accept=".csv"
        onFileSelect={(file) => setPipelineFile(file)}
        label="Pipeline Data File"
      />
      <div className="flex flex-wrap gap-3">
        <button
          onClick={handleRunBacktest}
          disabled={!pipelineFile || backtest.loading}
          className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
        >
          {backtest.loading ? "Running Backtest..." : "Run Backtest"}
        </button>
        <button
          onClick={handleRunForecast}
          disabled={!pipelineFile || forecast.loading}
          className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
        >
          {forecast.loading ? "Running Forecast..." : "Run Forecast"}
        </button>
      </div>
      {(backtest.loading || forecast.loading) && <ChartSkeleton height="h-24" />}
      {combinedError && <ErrorDisplay message={combinedError} onRetry={() => setError(null)} />}
      {backtest.result && (
        <div className="space-y-2 rounded-md bg-green-50 dark:bg-green-950 p-4">
          <h3 className="text-sm font-medium">Backtest Result</h3>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricCard label="Status" value={backtest.result.status} />
            <MetricCard label="Champion" value={backtest.result.champion_model || "N/A"} />
            <MetricCard label="Best WMAPE" value={backtest.result.best_wmape != null ? `${(backtest.result.best_wmape * 100).toFixed(1)}%` : "N/A"} />
            <MetricCard label="Series" value={backtest.result.series_count ?? "N/A"} />
          </div>
          {backtest.result.leaderboard && backtest.result.leaderboard.length > 0 && (
            <DataTable
              columns={Object.keys(backtest.result.leaderboard[0]).map(k => ({ key: k, label: k, sortable: true }))}
              data={backtest.result.leaderboard}
              pageSize={10}
            />
          )}
        </div>
      )}
      {forecast.result && (
        <div className="space-y-2 rounded-md bg-blue-50 dark:bg-blue-950 p-4">
          <h3 className="text-sm font-medium">Forecast Result</h3>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricCard label="Status" value={forecast.result.status} />
            <MetricCard label="Rows" value={forecast.result.forecast_rows ?? "N/A"} />
            <MetricCard label="Series" value={forecast.result.series_count ?? "N/A"} />
            <MetricCard label="LOB" value={forecast.result.lob} />
          </div>
          {forecast.result.forecast_preview && forecast.result.forecast_preview.length > 0 && (
            <DataTable
              columns={Object.keys(forecast.result.forecast_preview[0]).map(k => ({ key: k, label: k, sortable: true }))}
              data={forecast.result.forecast_preview}
              pageSize={10}
            />
          )}
        </div>
      )}
    </section>
  );
}
