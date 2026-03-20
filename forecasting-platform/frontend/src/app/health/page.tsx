"use client";

import { useState } from "react";
import { DriftHistogram } from "@/components/charts/drift-histogram";
import { TriagePanel } from "@/components/ai/triage-panel";
import { DataTable } from "@/components/data/data-table";
import { ComingSoon } from "@/components/shared/coming-soon";
import { MetricCard } from "@/components/shared/metric-card";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { ChartSkeleton } from "@/components/shared/loading-skeleton";
import { useDrift } from "@/hooks/use-drift";
import { useAudit } from "@/hooks/use-audit";
import { SEVERITY_COLORS } from "@/lib/constants";

export default function HealthPage() {
  const [lob, setLob] = useState("retail");
  const [activeTab, setActiveTab] = useState<"drift" | "audit" | "manifests">("drift");
  const [auditAction, setAuditAction] = useState("");
  const [auditLimit, setAuditLimit] = useState(100);

  const drift = useDrift(lob);
  const audit = useAudit({
    action: auditAction || undefined,
    limit: auditLimit,
  });

  const tabs = [
    { id: "drift" as const, label: "Drift & Triage" },
    { id: "audit" as const, label: "Audit Log" },
    { id: "manifests" as const, label: "Pipeline Manifests" },
  ];

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Platform Health</h1>
        <p className="text-sm text-muted-foreground">
          Monitor drift, triage anomalies, review audit log, and track pipeline manifests.
        </p>
      </div>

      {/* LOB Control */}
      <div className="flex items-end gap-4">
        <div className="space-y-1">
          <label className="text-xs font-medium">LOB</label>
          <input
            type="text"
            value={lob}
            onChange={(e) => setLob(e.target.value)}
            className="rounded-md border bg-background px-3 py-1.5 text-sm w-32"
          />
        </div>
      </div>

      {/* Tab navigation */}
      <div className="flex gap-1 border-b">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === tab.id
                ? "border-primary text-primary"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Drift & Triage Tab */}
      {activeTab === "drift" && (
        <div className="space-y-6">
          {drift.isLoading && <ChartSkeleton />}
          {drift.error && (
            <ErrorDisplay message={drift.error.message} onRetry={() => drift.refetch()} />
          )}
          {drift.data && (
            <>
              <div className="grid grid-cols-3 gap-3">
                <MetricCard label="Total Alerts" value={drift.data.alerts.length} />
                <MetricCard label="Critical" value={drift.data.n_critical} />
                <MetricCard label="Warning" value={drift.data.n_warning} />
              </div>

              {/* Alert histogram */}
              {drift.data.alerts.length > 0 && (
                <section className="space-y-3">
                  <h2 className="text-lg font-semibold">Alert Distribution</h2>
                  <div className="rounded-lg border p-4">
                    <DriftHistogram alerts={drift.data.alerts} />
                  </div>
                </section>
              )}

              {/* Alert table */}
              {drift.data.alerts.length > 0 && (
                <section className="space-y-3">
                  <h2 className="text-lg font-semibold">Drift Alerts</h2>
                  <div className="overflow-x-auto rounded-md border">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b bg-muted/50 text-left text-xs text-muted-foreground">
                          <th className="px-3 py-2">Series</th>
                          <th className="px-3 py-2">Metric</th>
                          <th className="px-3 py-2">Severity</th>
                          <th className="px-3 py-2">Current</th>
                          <th className="px-3 py-2">Baseline</th>
                          <th className="px-3 py-2">Message</th>
                        </tr>
                      </thead>
                      <tbody>
                        {drift.data.alerts.map((a, i) => (
                          <tr key={i} className="border-b last:border-0">
                            <td className="px-3 py-2 font-medium">{a.series_id}</td>
                            <td className="px-3 py-2">{a.metric}</td>
                            <td className="px-3 py-2">
                              <span
                                className="rounded px-1.5 py-0.5 text-xs font-medium text-white"
                                style={{ backgroundColor: SEVERITY_COLORS[a.severity] }}
                              >
                                {a.severity}
                              </span>
                            </td>
                            <td className="px-3 py-2">{a.current_value.toFixed(3)}</td>
                            <td className="px-3 py-2">{a.baseline_value.toFixed(3)}</td>
                            <td className="px-3 py-2 text-xs text-muted-foreground max-w-xs truncate">
                              {a.message}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              )}

              {/* AI Triage */}
              <section>
                <h2 className="text-lg font-semibold mb-3">AI Anomaly Triage</h2>
                <TriagePanel lob={lob} />
              </section>
            </>
          )}

          {drift.data && drift.data.alerts.length === 0 && (
            <div className="rounded-lg border p-8 text-center">
              <p className="text-sm text-muted-foreground">No drift alerts detected for this LOB.</p>
            </div>
          )}
        </div>
      )}

      {/* Audit Log Tab */}
      {activeTab === "audit" && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-end gap-4">
            <div className="space-y-1">
              <label className="text-xs font-medium">Action Filter</label>
              <input
                type="text"
                value={auditAction}
                onChange={(e) => setAuditAction(e.target.value)}
                placeholder="e.g. create_override"
                className="rounded-md border bg-background px-3 py-1.5 text-sm w-48"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium">Max Entries</label>
              <input
                type="number"
                value={auditLimit}
                onChange={(e) => setAuditLimit(Number(e.target.value))}
                min={10}
                max={1000}
                className="rounded-md border bg-background px-3 py-1.5 text-sm w-24"
              />
            </div>
          </div>

          {audit.isLoading && <ChartSkeleton />}
          {audit.error && (
            <ErrorDisplay message={audit.error.message} onRetry={() => audit.refetch()} />
          )}
          {audit.data && (
            <>
              <MetricCard label="Events Returned" value={audit.data.count} className="w-fit" />
              <DataTable
                columns={[
                  { key: "timestamp", label: "Timestamp", sortable: true },
                  { key: "user_id", label: "User", sortable: true },
                  { key: "action", label: "Action", sortable: true },
                  { key: "resource_type", label: "Resource", sortable: true },
                  { key: "status", label: "Status", sortable: true,
                    render: (v) => (
                      <span className={`rounded px-1.5 py-0.5 text-xs font-medium ${
                        v === "success"
                          ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                          : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                      }`}>
                        {String(v)}
                      </span>
                    ),
                  },
                ]}
                data={audit.data.events as unknown as Record<string, unknown>[]}
                pageSize={20}
              />
            </>
          )}
        </div>
      )}

      {/* Manifests Tab */}
      {activeTab === "manifests" && (
        <div className="space-y-4">
          <ComingSoon
            feature="Pipeline Manifests"
            description="View recent pipeline manifests with run ID, timestamp, LOB, series count, champion model, WMAPE, and validation status"
          />
          <ComingSoon
            feature="Cost Tracking"
            description="Per-model compute time, cost-per-series metrics, and cost trend visualization"
          />
        </div>
      )}
    </div>
  );
}
