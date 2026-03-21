"use client";

import { useState } from "react";
import { ErrorDisplay } from "@/components/shared/error-boundary";
import { api } from "@/lib/api-client";
import type { BIExportResponse } from "@/lib/types";

const EXPORT_TYPES = [
  { type: "forecast_vs_actual", label: "Export Forecast vs Actual" },
  { type: "leaderboard", label: "Export Leaderboard" },
  { type: "bias_report", label: "Export Bias Report" },
] as const;

export function BIExportPanel({ lob }: { lob: string }) {
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
        {EXPORT_TYPES.map(({ type, label }) => (
          <button
            key={type}
            onClick={() => handleExport(type)}
            disabled={exportLoading !== null}
            className="rounded-md border px-4 py-2 text-sm font-medium hover:bg-muted transition-colors disabled:opacity-50"
          >
            {exportLoading === type ? "Exporting..." : label}
          </button>
        ))}
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
