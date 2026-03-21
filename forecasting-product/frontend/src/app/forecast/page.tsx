"use client";

import { useState, useMemo } from "react";
import { TimeSeriesLine } from "@/components/charts/time-series-line";
import { NLQueryPanel } from "@/components/ai/nl-query-panel";
import { FileUpload } from "@/components/data/file-upload";
import { DataTable } from "@/components/data/data-table";
import { MetricCard } from "@/components/shared/metric-card";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { useForecast } from "@/hooks/use-forecast";
import { api } from "@/lib/api-client";
import { formatNumber } from "@/lib/utils";
import { COLORS } from "@/lib/constants";
import type { DecomposeResponse, ForecastCompareResponse, ConstrainResponse } from "@/lib/types";

function DecompositionPanel() {
  const [historyFile, setHistoryFile] = useState<File | null>(null);
  const [forecastFile, setForecastFile] = useState<File | null>(null);
  const [result, setResult] = useState<DecomposeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDecompose = async () => {
    if (!historyFile || !forecastFile) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.decomposeForecast(historyFile, forecastFile);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Decomposition failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="space-y-4 rounded-lg border p-6">
      <h2 className="text-lg font-semibold">Seasonal Decomposition</h2>
      <p className="text-sm text-muted-foreground">
        STL decomposition showing trend, seasonal, and residual components.
      </p>
      <div className="grid gap-4 sm:grid-cols-2">
        <FileUpload
          accept=".csv"
          onFileSelect={(file) => setHistoryFile(file)}
          label="History File"
        />
        <FileUpload
          accept=".csv"
          onFileSelect={(file) => setForecastFile(file)}
          label="Forecast File"
        />
      </div>
      <button
        onClick={handleDecompose}
        disabled={!historyFile || !forecastFile || loading}
        className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
      >
        {loading ? "Decomposing..." : "Run Decomposition"}
      </button>
      {loading && <ChartSkeleton />}
      {error && <ErrorDisplay message={error} onRetry={() => setError(null)} />}
      {result && (
        <div className="space-y-3">
          <MetricCard label="Series Count" value={result.series_count} />
          {result.decomposition.length > 0 && (
            <DataTable
              columns={Object.keys(result.decomposition[0]).map((k) => ({
                key: k,
                label: k,
                sortable: true,
              }))}
              data={result.decomposition}
              pageSize={10}
            />
          )}
          {Object.keys(result.narratives).length > 0 && (
            <div className="space-y-2">
              <h3 className="text-sm font-medium">Narratives</h3>
              {Object.entries(result.narratives).map(([series, narrative]) => (
                <div key={series} className="rounded-md bg-muted/50 p-3">
                  <p className="text-xs font-medium font-mono">{series}</p>
                  <p className="text-sm text-muted-foreground mt-1">{narrative}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function ComparisonPanel() {
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
      <h2 className="text-lg font-semibold">Forecast Comparison</h2>
      <p className="text-sm text-muted-foreground">
        Upload external forecast for overlay comparison with gap analysis.
      </p>
      <FileUpload
        accept=".csv"
        onFileSelect={(file) => setModelFile(file)}
        label="Model Forecast File"
      />
      <FileUpload
        accept=".csv"
        onFileSelect={(file) => setExternalFile(file)}
        label="External Forecast File"
      />
      <button
        onClick={handleCompare}
        disabled={!modelFile || !externalFile || loading}
        className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
      >
        {loading ? "Comparing..." : "Compare Forecasts"}
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

function ConstrainedForecastPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [minDemand, setMinDemand] = useState<string>("0");
  const [maxCapacity, setMaxCapacity] = useState<string>("");
  const [aggregateMax, setAggregateMax] = useState<string>("");
  const [proportional, setProportional] = useState(false);
  const [result, setResult] = useState<ConstrainResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleConstrain = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const params: { min_demand?: number; max_capacity?: number; aggregate_max?: number; proportional?: boolean } = {};
      if (minDemand) params.min_demand = Number(minDemand);
      if (maxCapacity) params.max_capacity = Number(maxCapacity);
      if (aggregateMax) params.aggregate_max = Number(aggregateMax);
      params.proportional = proportional;
      const data = await api.constrainForecast(file, params);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Constraint application failed");
    } finally {
      setLoading(false);
    }
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

export default function ForecastPage() {
  const [lob, setLob] = useState("retail");
  const [selectedSeries, setSelectedSeries] = useState("");
  const { data, isLoading, error, refetch } = useForecast(lob);

  // Extract unique series IDs
  const seriesIds = useMemo(() => {
    if (!data) return [];
    return [...new Set(data.points.map((p) => p.series_id))];
  }, [data]);

  // Filter points for selected series
  const seriesPoints = useMemo(() => {
    if (!data || !selectedSeries) return [];
    return data.points
      .filter((p) => p.series_id === selectedSeries)
      .sort((a, b) => a.week.localeCompare(b.week))
      .map((p) => ({
        week: p.week,
        forecast: p.forecast,
        model: p.model || "unknown",
      }));
  }, [data, selectedSeries]);

  // Auto-select first series
  if (seriesIds.length > 0 && !selectedSeries) {
    setSelectedSeries(seriesIds[0]);
  }

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Forecast Viewer</h1>
        <p className="text-sm text-muted-foreground">
          View forecasts with fan charts, seasonal decomposition, AI Q&amp;A, and constrained forecasting.
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-end gap-4">
        <div className="space-y-1">
          <label className="text-xs font-medium">LOB</label>
          <input
            type="text"
            value={lob}
            onChange={(e) => setLob(e.target.value)}
            className="rounded-md border bg-background px-3 py-1.5 text-sm w-32"
          />
        </div>
        {seriesIds.length > 0 && (
          <div className="space-y-1">
            <label className="text-xs font-medium">Series</label>
            <select
              value={selectedSeries}
              onChange={(e) => setSelectedSeries(e.target.value)}
              className="rounded-md border bg-background px-3 py-1.5 text-sm max-w-xs"
            >
              {seriesIds.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
          </div>
        )}
        <button
          onClick={() => refetch()}
          className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          Load Forecasts
        </button>
      </div>

      {isLoading && <ChartSkeleton />}
      {error && <ErrorDisplay message={error.message} onRetry={() => refetch()} />}

      {data && (
        <>
          {/* Summary */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricCard label="Series Count" value={formatNumber(data.series_count)} />
            <MetricCard label="Forecast Origin" value={data.forecast_origin || "N/A"} />
            <MetricCard label="Points" value={formatNumber(data.points.length)} />
            <MetricCard
              label="Horizon"
              value={selectedSeries ? `${seriesPoints.length} periods` : "N/A"}
            />
          </div>

          {/* Forecast Chart */}
          {seriesPoints.length > 0 && (
            <section className="space-y-3">
              <h2 className="text-lg font-semibold">
                Forecast: {selectedSeries}
              </h2>
              <div className="rounded-lg border p-4">
                <TimeSeriesLine
                  data={seriesPoints}
                  xKey="week"
                  lines={[
                    {
                      dataKey: "forecast",
                      name: "Forecast",
                      color: COLORS.primary,
                    },
                  ]}
                  height={350}
                />
                <p className="mt-2 text-xs text-muted-foreground">
                  Model: {seriesPoints[0]?.model}
                </p>
              </div>
            </section>
          )}

          {/* Fan Chart note */}
          <section className="rounded-lg border p-6">
            <h2 className="text-lg font-semibold">Fan Chart (P10-P90)</h2>
            <p className="text-sm text-muted-foreground mt-2">
              Fan chart requires quantile columns (p10, p25, p50, p75, p90) in forecast data. Available when prediction intervals are enabled in config.
            </p>
          </section>

          {/* Decomposition */}
          <DecompositionPanel />

          {/* AI Q&A — live */}
          {selectedSeries && (
            <section>
              <h2 className="text-lg font-semibold mb-3">Ask About This Forecast</h2>
              <NLQueryPanel
                seriesId={selectedSeries}
                lob={lob}
                suggestions={[
                  "Why this trend?",
                  "What drives seasonality?",
                  "How does this compare to last year?",
                  "Is this forecast reliable?",
                ]}
              />
            </section>
          )}

          {/* Comparison & Constraints */}
          <ComparisonPanel />
          <ConstrainedForecastPanel />
        </>
      )}
    </div>
  );
}
