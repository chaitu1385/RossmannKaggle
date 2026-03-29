"use client";

import { CommentaryPanel } from "@/components/ai/commentary-panel";
import { useLob } from "@/providers/lob-provider";
import { MetricCard } from "@/components/shared/metric-card";
import { CrossRunComparisonPanel } from "@/components/forecast/cross-run-comparison-panel";
import { ModelCardsPanel } from "@/components/governance/model-cards-panel";
import { LineagePanel } from "@/components/governance/lineage-panel";
import { BIExportPanel } from "@/components/governance/bi-export-panel";
import { WorkflowNav } from "@/components/shared/workflow-nav";

export default function SOPPage() {
  const { lob, setLob } = useLob();

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

      <WorkflowNav currentStep="/sop" />
    </div>
  );
}
