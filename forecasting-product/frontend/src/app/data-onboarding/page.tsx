"use client";

import { useState } from "react";
import { FileUpload } from "@/components/data/file-upload";
import { ConfigViewer } from "@/components/data/config-viewer";
import { ForecastabilityGauge } from "@/components/charts/forecastability-gauge";
import { DemandClassDonut } from "@/components/charts/demand-class-donut";
import { DataTable } from "@/components/data/data-table";
import { MetricCard } from "@/components/shared/metric-card";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { useAnalyze } from "@/hooks/use-analyze";
import { api } from "@/lib/api-client";
import { formatNumber } from "@/lib/utils";
import type { AnalysisResponse, MultiFileAnalysisResponse, PipelineRunResponse } from "@/lib/types";

function MultiFilePanel({ lobName }: { lobName: string }) {
  const [files, setFiles] = useState<File[]>([]);
  const [result, setResult] = useState<MultiFileAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFilesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));
    }
  };

  const handleAnalyze = async () => {
    if (files.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.analyzeMultiFile(files, lobName);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Multi-file analysis failed");
    } finally {
      setLoading(false);
    }
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

function PipelineExecutionPanel({ lobName }: { lobName: string }) {
  const [pipelineFile, setPipelineFile] = useState<File | null>(null);
  const [backtestResult, setBacktestResult] = useState<PipelineRunResponse | null>(null);
  const [forecastResult, setForecastResult] = useState<PipelineRunResponse | null>(null);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRunBacktest = async () => {
    if (!pipelineFile) return;
    setBacktestLoading(true);
    setError(null);
    try {
      const data = await api.runBacktest(pipelineFile, lobName);
      setBacktestResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Backtest failed");
    } finally {
      setBacktestLoading(false);
    }
  };

  const handleRunForecast = async () => {
    if (!pipelineFile) return;
    setForecastLoading(true);
    setError(null);
    try {
      const data = await api.runForecast(pipelineFile, lobName);
      setForecastResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Forecast failed");
    } finally {
      setForecastLoading(false);
    }
  };

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
          disabled={!pipelineFile || backtestLoading}
          className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
        >
          {backtestLoading ? "Running Backtest..." : "Run Backtest"}
        </button>
        <button
          onClick={handleRunForecast}
          disabled={!pipelineFile || forecastLoading}
          className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
        >
          {forecastLoading ? "Running Forecast..." : "Run Forecast"}
        </button>
      </div>
      {(backtestLoading || forecastLoading) && <ChartSkeleton height="h-24" />}
      {error && <ErrorDisplay message={error} onRetry={() => setError(null)} />}
      {backtestResult && (
        <div className="space-y-2 rounded-md bg-green-50 dark:bg-green-950 p-4">
          <h3 className="text-sm font-medium">Backtest Result</h3>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricCard label="Status" value={backtestResult.status} />
            <MetricCard label="Champion" value={backtestResult.champion_model || "N/A"} />
            <MetricCard label="Best WMAPE" value={backtestResult.best_wmape != null ? `${(backtestResult.best_wmape * 100).toFixed(1)}%` : "N/A"} />
            <MetricCard label="Series" value={backtestResult.series_count ?? "N/A"} />
          </div>
          {backtestResult.leaderboard && backtestResult.leaderboard.length > 0 && (
            <DataTable
              columns={Object.keys(backtestResult.leaderboard[0]).map(k => ({ key: k, label: k, sortable: true }))}
              data={backtestResult.leaderboard}
              pageSize={10}
            />
          )}
        </div>
      )}
      {forecastResult && (
        <div className="space-y-2 rounded-md bg-blue-50 dark:bg-blue-950 p-4">
          <h3 className="text-sm font-medium">Forecast Result</h3>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricCard label="Status" value={forecastResult.status} />
            <MetricCard label="Rows" value={forecastResult.forecast_rows ?? "N/A"} />
            <MetricCard label="Series" value={forecastResult.series_count ?? "N/A"} />
            <MetricCard label="LOB" value={forecastResult.lob} />
          </div>
          {forecastResult.forecast_preview && forecastResult.forecast_preview.length > 0 && (
            <DataTable
              columns={Object.keys(forecastResult.forecast_preview[0]).map(k => ({ key: k, label: k, sortable: true }))}
              data={forecastResult.forecast_preview}
              pageSize={10}
            />
          )}
        </div>
      )}
    </section>
  );
}

export default function DataOnboardingPage() {
  const [lobName, setLobName] = useState("retail");
  const [llmEnabled, setLlmEnabled] = useState(false);
  const analyzeMutation = useAnalyze();
  const result: AnalysisResponse | undefined = analyzeMutation.data;

  const handleFileSelect = (file: File) => {
    analyzeMutation.mutate({ file, lobName, llmEnabled });
  };

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Data Onboarding</h1>
        <p className="text-sm text-muted-foreground">
          Upload your time series CSV for automated analysis, schema detection, and config generation.
        </p>
      </div>

      {/* Upload Section */}
      <section className="space-y-4 rounded-lg border p-6">
        <h2 className="text-lg font-semibold">Upload Data</h2>

        <div className="flex flex-wrap items-end gap-4">
          <div className="space-y-1">
            <label className="text-xs font-medium">LOB Name</label>
            <input
              type="text"
              value={lobName}
              onChange={(e) => setLobName(e.target.value)}
              className="rounded-md border bg-background px-3 py-1.5 text-sm w-40"
            />
          </div>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={llmEnabled}
              onChange={(e) => setLlmEnabled(e.target.checked)}
              className="rounded"
            />
            Enable AI Interpretation
          </label>
        </div>

        <FileUpload
          accept=".csv"
          onFileSelect={handleFileSelect}
          label="Time Series Data"
        />

        {analyzeMutation.isPending && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              Analyzing data...
            </div>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <ChartSkeleton height="h-24" />
              <ChartSkeleton height="h-24" />
              <ChartSkeleton height="h-24" />
              <ChartSkeleton height="h-24" />
            </div>
          </div>
        )}

        {analyzeMutation.isError && (
          <ErrorDisplay
            message={analyzeMutation.error.message}
            onRetry={() => analyzeMutation.reset()}
          />
        )}
      </section>

      {/* Results */}
      {result && (
        <>
          {/* Schema Detection */}
          <section className="space-y-4">
            <h2 className="text-lg font-semibold">Schema Detection</h2>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <MetricCard label="Series" value={formatNumber(result.n_series)} />
              <MetricCard label="Rows" value={formatNumber(result.n_rows)} />
              <MetricCard label="Frequency" value={result.frequency} />
              <MetricCard
                label="Date Range"
                value={`${result.date_range_start} \u2192 ${result.date_range_end}`}
              />
            </div>
            <div className="grid grid-cols-3 gap-3">
              <div className="rounded-md bg-muted/50 p-3">
                <p className="text-xs text-muted-foreground">Time Column</p>
                <p className="font-mono text-sm font-medium">{result.time_column}</p>
              </div>
              <div className="rounded-md bg-muted/50 p-3">
                <p className="text-xs text-muted-foreground">Target Column</p>
                <p className="font-mono text-sm font-medium">{result.target_column}</p>
              </div>
              <div className="rounded-md bg-muted/50 p-3">
                <p className="text-xs text-muted-foreground">ID Columns</p>
                <p className="font-mono text-sm font-medium">{result.id_columns.join(", ")}</p>
              </div>
            </div>
          </section>

          {/* Forecastability Assessment */}
          <section className="space-y-4">
            <h2 className="text-lg font-semibold">Forecastability Assessment</h2>
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              <div className="flex flex-col items-center rounded-lg border p-4">
                <ForecastabilityGauge value={result.overall_forecastability} />
              </div>
              <div className="rounded-lg border p-4">
                <h3 className="mb-2 text-sm font-medium">Demand Classification</h3>
                <DemandClassDonut data={result.demand_classes} />
              </div>
            </div>
          </section>

          {/* Config Reasoning */}
          {result.config_reasoning.length > 0 && (
            <section className="space-y-3">
              <h2 className="text-lg font-semibold">Configuration Reasoning</h2>
              <ul className="space-y-1">
                {result.config_reasoning.map((r, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm">
                    <span className="mt-0.5 text-primary">&bull;</span> {r}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* Hypotheses */}
          {result.hypotheses.length > 0 && (
            <section className="space-y-3">
              <h2 className="text-lg font-semibold">Hypotheses</h2>
              <div className="grid gap-2 sm:grid-cols-2">
                {result.hypotheses.map((h, i) => (
                  <div key={i} className="rounded-md bg-blue-50 dark:bg-blue-950 p-3 text-sm">
                    {h}
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* AI Narrative */}
          {result.llm_narrative && (
            <section className="space-y-3">
              <h2 className="text-lg font-semibold">AI Interpretation</h2>
              <div className="rounded-md bg-purple-50 dark:bg-purple-950 p-4 text-sm leading-relaxed">
                {result.llm_narrative}
              </div>
              {result.llm_risk_factors && result.llm_risk_factors.length > 0 && (
                <div>
                  <h3 className="text-sm font-medium text-destructive">Risk Factors</h3>
                  <ul className="mt-1 space-y-1">
                    {result.llm_risk_factors.map((r, i) => (
                      <li key={i} className="text-sm text-muted-foreground">
                        &bull; {r}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </section>
          )}

          {/* Config YAML */}
          <section className="space-y-3">
            <h2 className="text-lg font-semibold">Recommended Configuration</h2>
            <ConfigViewer yaml={result.recommended_config_yaml} />
          </section>

          {/* Multi-File Classification & Merge */}
          <MultiFilePanel lobName={lobName} />

          {/* Pipeline Execution */}
          <PipelineExecutionPanel lobName={lobName} />
        </>
      )}
    </div>
  );
}
