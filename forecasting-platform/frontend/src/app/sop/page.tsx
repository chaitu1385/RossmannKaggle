"use client";

import { useState } from "react";
import { CommentaryPanel } from "@/components/ai/commentary-panel";
import { ComingSoon } from "@/components/shared/coming-soon";
import { MetricCard } from "@/components/shared/metric-card";

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
      <ComingSoon
        feature="Cross-Run Comparison"
        description="Upload prior run forecasts for overlay comparison with gap analysis and summary metrics"
      />

      {/* Model Governance */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Model Governance</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          <ComingSoon
            feature="Model Cards"
            description="View model card details: version, training window, series count, metrics, features, config hash"
          />
          <ComingSoon
            feature="Forecast Lineage"
            description="Trace forecast provenance: data source → preprocessing → model → postprocessing → output"
          />
        </div>
      </section>

      {/* BI Export */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">BI Export</h2>
        <div className="flex flex-wrap gap-3">
          <button
            disabled
            className="rounded-md border px-4 py-2 text-sm text-muted-foreground opacity-50 cursor-not-allowed"
          >
            Export Forecast vs Actual
          </button>
          <button
            disabled
            className="rounded-md border px-4 py-2 text-sm text-muted-foreground opacity-50 cursor-not-allowed"
          >
            Export Leaderboard
          </button>
          <button
            disabled
            className="rounded-md border px-4 py-2 text-sm text-muted-foreground opacity-50 cursor-not-allowed"
          >
            Export Bias Report
          </button>
        </div>
        <p className="text-xs text-muted-foreground">
          BI export requires API endpoints — coming in a future release
        </p>
      </section>
    </div>
  );
}
