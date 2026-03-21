"use client";

import { useState, useEffect } from "react";
import { CommentaryPanel } from "@/components/ai/commentary-panel";
import { FileUpload } from "@/components/data/file-upload";
import { DataTable } from "@/components/data/data-table";
import { MetricCard } from "@/components/shared/metric-card";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { api } from "@/lib/api-client";
import type { ForecastCompareResponse, ModelCardListResponse, LineageResponse, BIExportResponse } from "@/lib/types";

function CrossRunComparisonPanel() {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [externalFile, setExternalFile] = useState<File | null>(null);
  const [result, setResult] = useState<ForecastCompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCompare = async () => {
    if (!modelFile || !externalFile) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.compareForecasts(modelFile, externalFile);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Comparison failed");
    } finally {
      setLoading(false);
    }
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

function ModelCardsPanel() {
  const [data, setData] = useState<ModelCardListResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api.listModelCards()
      .then(setData)
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load model cards"))
      .finally(() => setLoading(false));
  }, []);

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
          onRetry={() => {
            setLoading(true);
            setError(null);
            api.listModelCards()
              .then(setData)
              .catch((err) => setError(err instanceof Error ? err.message : "Failed"))
              .finally(() => setLoading(false));
          }}
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

function LineagePanel({ lob }: { lob: string }) {
  const [data, setData] = useState<LineageResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api.getLineage(lob)
      .then(setData)
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load lineage"))
      .finally(() => setLoading(false));
  }, [lob]);

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
          onRetry={() => {
            setLoading(true);
            setError(null);
            api.getLineage(lob)
              .then(setData)
              .catch((err) => setError(err instanceof Error ? err.message : "Failed"))
              .finally(() => setLoading(false));
          }}
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

function BIExportPanel({ lob }: { lob: string }) {
  const [exportLoading, setExportLoading] = useState<string | null>(null);
  const [exportResult, setExportResult] = useState<BIExportResponse | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);

  const handleExport = async (reportType: string) => {
    setExportLoading(reportType);
    setExportError(null);
    setExportResult(null);
    try {
      const data = await api.biExport(reportType, lob);
      setExportResult(data);
    } catch (err) {
      setExportError(err instanceof Error ? err.message : `Export "${reportType}" failed`);
    } finally {
      setExportLoading(null);
    }
  };

  return (
    <section className="space-y-3">
      <h2 className="text-lg font-semibold">BI Export</h2>
      <div className="flex flex-wrap gap-3">
        <button
          onClick={() => handleExport("forecast_vs_actual")}
          disabled={exportLoading !== null}
          className="rounded-md border px-4 py-2 text-sm font-medium hover:bg-muted transition-colors disabled:opacity-50"
        >
          {exportLoading === "forecast_vs_actual" ? "Exporting..." : "Export Forecast vs Actual"}
        </button>
        <button
          onClick={() => handleExport("leaderboard")}
          disabled={exportLoading !== null}
          className="rounded-md border px-4 py-2 text-sm font-medium hover:bg-muted transition-colors disabled:opacity-50"
        >
          {exportLoading === "leaderboard" ? "Exporting..." : "Export Leaderboard"}
        </button>
        <button
          onClick={() => handleExport("bias_report")}
          disabled={exportLoading !== null}
          className="rounded-md border px-4 py-2 text-sm font-medium hover:bg-muted transition-colors disabled:opacity-50"
        >
          {exportLoading === "bias_report" ? "Exporting..." : "Export Bias Report"}
        </button>
      </div>
      {exportError && <ErrorDisplay message={exportError} onRetry={() => setExportError(null)} />}
      {exportResult && (
        <div className="rounded-md bg-green-50 dark:bg-green-950 p-3">
          <p className="text-sm">
            Export <span className="font-medium">{exportResult.report_type}</span> completed.
            Status: <span className="font-medium">{exportResult.status}</span>.
            Path: <span className="font-mono text-xs">{exportResult.export_path}</span>
          </p>
        </div>
      )}
    </section>
  );
}

export default function SOPPage() {
  const [lob, setLob] = useState("retail");

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold">S&amp;OP Meeting Prep</h1>
        <p className="text-sm text-muted-foreground">
          AI executive commentary, cross-run comparison, model governance, and BI exports.
        </p>
      </div>

      {/* LOB control */}
      <div className="space-y-1">
        <label className="text-xs font-medium">LOB</label>
        <input
          type="text"
          value={lob}
          onChange={(e) => setLob(e.target.value)}
          className="rounded-md border bg-background px-3 py-1.5 text-sm w-32"
        />
      </div>

      {/* Quick summary cards (demo) */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricCard label="Overall WMAPE" value="16.2%" trend="improving" />
        <MetricCard label="Bias" value="-2.1%" trend="stable" />
        <MetricCard label="Series at Risk" value="12" trend="degrading" />
        <MetricCard label="Champion Models" value="5" />
      </div>

      {/* AI Commentary — live endpoint */}
      <section>
        <h2 className="text-lg font-semibold mb-3">Executive Commentary</h2>
        <CommentaryPanel lob={lob} />
      </section>

      {/* Cross-Run Comparison */}
      <CrossRunComparisonPanel />

      {/* Model Governance */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Model Governance</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          <ModelCardsPanel />
          <LineagePanel lob={lob} />
        </div>
      </section>

      {/* BI Export */}
      <BIExportPanel lob={lob} />
    </div>
  );
}
