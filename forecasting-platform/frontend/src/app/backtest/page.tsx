"use client";

import { useState } from "react";
import { LeaderboardBar } from "@/components/charts/leaderboard-bar";
import { FVACascade } from "@/components/charts/fva-cascade";
import { CalibrationPlot } from "@/components/charts/calibration-plot";
import { ConfigTunerPanel } from "@/components/ai/config-tuner-panel";
import { ComingSoon } from "@/components/shared/coming-soon";
import { MetricCard } from "@/components/shared/metric-card";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { useLeaderboard } from "@/hooks/use-leaderboard";
import { formatPct } from "@/lib/utils";

// Demo FVA data
const DEMO_FVA = [
  { layer: "Naive", wmape: 0.28 },
  { layer: "Statistical", wmape: 0.22 },
  { layer: "ML", wmape: 0.18 },
  { layer: "Neural", wmape: 0.19 },
  { layer: "Ensemble", wmape: 0.16 },
];

// Demo calibration data
const DEMO_CALIBRATION = [
  { nominal: 0.1, empirical: 0.12 },
  { nominal: 0.2, empirical: 0.18 },
  { nominal: 0.3, empirical: 0.28 },
  { nominal: 0.5, empirical: 0.48 },
  { nominal: 0.7, empirical: 0.65 },
  { nominal: 0.8, empirical: 0.78 },
  { nominal: 0.9, empirical: 0.88 },
  { nominal: 0.95, empirical: 0.93 },
];

export default function BacktestPage() {
  const [lob, setLob] = useState("retail");
  const [runType, setRunType] = useState("backtest");
  const { data, isLoading, error, refetch } = useLeaderboard(lob, runType);

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Backtest Results</h1>
        <p className="text-sm text-muted-foreground">
          Model leaderboard, FVA analysis, calibration, feature attribution, and AI config tuning.
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
        <div className="space-y-1">
          <label className="text-xs font-medium">Run Type</label>
          <select
            value={runType}
            onChange={(e) => setRunType(e.target.value)}
            className="rounded-md border bg-background px-3 py-1.5 text-sm"
          >
            <option value="backtest">Backtest</option>
            <option value="live">Live</option>
          </select>
        </div>
        <button
          onClick={() => refetch()}
          className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          Load
        </button>
      </div>

      {/* Leaderboard */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Model Leaderboard</h2>
        {isLoading && <ChartSkeleton />}
        {error && <ErrorDisplay message={error.message} onRetry={() => refetch()} />}
        {data && (
          <>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <MetricCard label="Models" value={data.entries.length} />
              <MetricCard
                label="Best WMAPE"
                value={data.entries.length > 0 ? formatPct(Math.min(...data.entries.map((e) => e.wmape))) : "N/A"}
              />
              <MetricCard
                label="Best Model"
                value={data.entries.length > 0 ? data.entries.sort((a, b) => a.wmape - b.wmape)[0].model : "N/A"}
              />
              <MetricCard label="Run Type" value={data.run_type} />
            </div>
            <div className="rounded-lg border p-4">
              <LeaderboardBar entries={data.entries} />
            </div>

            {/* Leaderboard table */}
            <div className="overflow-x-auto rounded-md border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50 text-left text-xs text-muted-foreground">
                    <th className="px-3 py-2">Rank</th>
                    <th className="px-3 py-2">Model</th>
                    <th className="px-3 py-2">WMAPE</th>
                    <th className="px-3 py-2">Bias</th>
                    <th className="px-3 py-2">Series</th>
                  </tr>
                </thead>
                <tbody>
                  {data.entries
                    .sort((a, b) => a.rank - b.rank)
                    .map((e) => (
                      <tr key={e.model} className="border-b last:border-0">
                        <td className="px-3 py-2 font-medium">{e.rank}</td>
                        <td className="px-3 py-2">{e.model}</td>
                        <td className="px-3 py-2">{formatPct(e.wmape)}</td>
                        <td className="px-3 py-2">{formatPct(e.normalized_bias)}</td>
                        <td className="px-3 py-2">{e.n_series}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>

      {/* FVA Cascade (demo) */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Forecast Value Added (FVA)</h2>
        <div className="rounded-lg border p-4">
          <FVACascade data={DEMO_FVA} />
          <p className="mt-2 text-xs text-center text-muted-foreground">
            Demo data — live FVA requires metric store access
          </p>
        </div>
      </section>

      {/* Calibration (demo) */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Prediction Interval Calibration</h2>
        <div className="rounded-lg border p-4">
          <CalibrationPlot data={DEMO_CALIBRATION} />
          <p className="mt-2 text-xs text-center text-muted-foreground">
            Demo data — live calibration requires metric store access
          </p>
        </div>
      </section>

      {/* SHAP */}
      <ComingSoon
        feature="SHAP Feature Attribution"
        description="Top features by mean |SHAP| for LightGBM/XGBoost models with per-series waterfall"
      />

      {/* AI Config Tuner — live endpoint */}
      <section>
        <h2 className="text-lg font-semibold mb-3">AI Configuration Tuner</h2>
        <ConfigTunerPanel lob={lob} runType={runType} />
      </section>
    </div>
  );
}
