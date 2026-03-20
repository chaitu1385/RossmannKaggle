"use client";

import { useState, useMemo } from "react";
import { TimeSeriesLine } from "@/components/charts/time-series-line";
import { NLQueryPanel } from "@/components/ai/nl-query-panel";
import { ComingSoon } from "@/components/shared/coming-soon";
import { MetricCard } from "@/components/shared/metric-card";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { useForecast } from "@/hooks/use-forecast";
import { formatNumber } from "@/lib/utils";
import { COLORS } from "@/lib/constants";

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

          {/* Fan chart placeholder */}
          <ComingSoon
            feature="Fan Chart (P10-P90)"
            description="Prediction interval bands with P10, P25, P50, P75, P90 percentiles and actuals overlay"
          />

          {/* Decomposition placeholder */}
          <ComingSoon
            feature="Seasonal Decomposition"
            description="STL decomposition showing trend, seasonal, and residual components"
          />

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
          <div className="grid gap-4 sm:grid-cols-2">
            <ComingSoon
              feature="Forecast Comparison"
              description="Upload external forecast for overlay comparison with gap analysis"
            />
            <ComingSoon
              feature="Constrained Forecast"
              description="Apply capacity and budget constraints to forecasts"
            />
          </div>
        </>
      )}
    </div>
  );
}
