"use client";

import { useState } from "react";
import { NLQueryPanel } from "@/components/ai/nl-query-panel";
import { ComingSoon } from "@/components/shared/coming-soon";
import { MetricCard } from "@/components/shared/metric-card";
import { DEMAND_CLASS_COLORS } from "@/lib/constants";

// Demo data for SBC scatter (shown until API endpoint exists)
const DEMO_SERIES = [
  { id: "SKU_001", adi: 1.0, cv2: 0.15, class: "Smooth" },
  { id: "SKU_002", adi: 1.8, cv2: 0.30, class: "Intermittent" },
  { id: "SKU_003", adi: 0.8, cv2: 0.65, class: "Erratic" },
  { id: "SKU_004", adi: 2.1, cv2: 0.70, class: "Lumpy" },
  { id: "SKU_005", adi: 1.1, cv2: 0.20, class: "Smooth" },
  { id: "SKU_006", adi: 0.5, cv2: 0.10, class: "Smooth" },
  { id: "SKU_007", adi: 1.5, cv2: 0.55, class: "Lumpy" },
  { id: "SKU_008", adi: 2.0, cv2: 0.25, class: "Intermittent" },
];

export default function SeriesExplorerPage() {
  const [selectedSeries, setSelectedSeries] = useState(DEMO_SERIES[0].id);
  const [lob, setLob] = useState("retail");

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Series Explorer</h1>
        <p className="text-sm text-muted-foreground">
          Deep-dive into series-level demand patterns, structural breaks, quality, and cleansing.
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
          <label className="text-xs font-medium">Series</label>
          <select
            value={selectedSeries}
            onChange={(e) => setSelectedSeries(e.target.value)}
            className="rounded-md border bg-background px-3 py-1.5 text-sm"
          >
            {DEMO_SERIES.map((s) => (
              <option key={s.id} value={s.id}>
                {s.id} ({s.class})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Series Overview Cards */}
      <section className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricCard label="Series Length" value="104" unit="periods" />
        <MetricCard label="Mean Demand" value="1,245" />
        <MetricCard label="Zero Periods" value="5%" />
        <MetricCard
          label="Demand Class"
          value={DEMO_SERIES.find((s) => s.id === selectedSeries)?.class || "Smooth"}
        />
      </section>

      {/* SBC Scatter (demo) */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">SBC Demand Classification</h2>
        <div className="rounded-lg border p-4">
          <div className="relative h-64 w-full">
            {/* Simple SVG scatter plot */}
            <svg viewBox="0 0 400 250" className="h-full w-full">
              {/* Grid lines */}
              <line x1="50" y1="0" x2="50" y2="220" stroke="hsl(var(--border))" strokeDasharray="2" />
              <line x1="50" y1="220" x2="380" y2="220" stroke="hsl(var(--border))" strokeDasharray="2" />
              {/* Quadrant lines at ADI=1.32 and CV²=0.49 */}
              <line x1={50 + (1.32 / 3) * 330} y1="0" x2={50 + (1.32 / 3) * 330} y2="220" stroke="hsl(var(--muted-foreground))" strokeDasharray="4" opacity={0.5} />
              <line x1="50" y1={220 - (0.49 / 1) * 220} x2="380" y2={220 - (0.49 / 1) * 220} stroke="hsl(var(--muted-foreground))" strokeDasharray="4" opacity={0.5} />
              {/* Axis labels */}
              <text x="215" y="245" textAnchor="middle" className="fill-muted-foreground text-[10px]">ADI</text>
              <text x="15" y="110" textAnchor="middle" transform="rotate(-90,15,110)" className="fill-muted-foreground text-[10px]">CV²</text>
              {/* Quadrant labels */}
              <text x="120" y="30" className="fill-muted-foreground text-[9px]" opacity={0.6}>Smooth</text>
              <text x="300" y="30" className="fill-muted-foreground text-[9px]" opacity={0.6}>Intermittent</text>
              <text x="120" y="200" className="fill-muted-foreground text-[9px]" opacity={0.6}>Erratic</text>
              <text x="300" y="200" className="fill-muted-foreground text-[9px]" opacity={0.6}>Lumpy</text>
              {/* Points */}
              {DEMO_SERIES.map((s) => (
                <circle
                  key={s.id}
                  cx={50 + (s.adi / 3) * 330}
                  cy={220 - (s.cv2 / 1) * 220}
                  r={s.id === selectedSeries ? 8 : 6}
                  fill={DEMAND_CLASS_COLORS[s.class]}
                  stroke={s.id === selectedSeries ? "hsl(var(--foreground))" : "none"}
                  strokeWidth={2}
                  opacity={0.8}
                >
                  <title>{s.id}: ADI={s.adi}, CV²={s.cv2}</title>
                </circle>
              ))}
            </svg>
          </div>
          <p className="mt-2 text-xs text-muted-foreground text-center">
            Demo data — live SBC scatter requires API endpoint
          </p>
        </div>
      </section>

      {/* Placeholder sections */}
      <div className="grid gap-4 sm:grid-cols-2">
        <ComingSoon feature="Structural Break Detection" description="CUSUM/PELT changepoint detection with configurable method and penalty" />
        <ComingSoon feature="Demand Cleansing Audit" description="Before/after cleansing chart with highlighted outliers" />
      </div>
      <ComingSoon feature="Regressor Screening" description="Correlation heatmap, mutual information scores, kept/dropped table" />

      {/* AI Q&A — this is live (uses /ai/explain endpoint) */}
      <section>
        <h2 className="text-lg font-semibold mb-3">AI Series Analysis</h2>
        <NLQueryPanel
          seriesId={selectedSeries}
          lob={lob}
          suggestions={[
            "Why is demand volatile?",
            "Is there a trend?",
            "Any anomalies?",
            "What seasonal pattern exists?",
          ]}
        />
      </section>
    </div>
  );
}
