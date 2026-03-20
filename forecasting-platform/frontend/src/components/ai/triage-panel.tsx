"use client";

import { useState } from "react";
import { ShieldAlert, Sparkles } from "lucide-react";
import { api } from "@/lib/api-client";
import type { TriageResponse } from "@/lib/types";
import { SEVERITY_COLORS } from "@/lib/constants";

interface Props {
  lob: string;
  runType?: string;
}

export function TriagePanel({ lob, runType = "backtest" }: Props) {
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<TriageResponse | null>(null);
  const [error, setError] = useState("");

  const handleTriage = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await api.aiTriage({ lob, run_type: runType, max_alerts: 50 });
      setResponse(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Triage failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4 rounded-lg border p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ShieldAlert className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold">AI Anomaly Triage</h3>
        </div>
        <button
          onClick={handleTriage}
          disabled={loading}
          className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors"
        >
          <Sparkles className="h-3 w-3" />
          {loading ? "Triaging..." : "Triage Alerts with AI"}
        </button>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      {response && !loading && (
        <div className="space-y-4">
          {/* Executive summary */}
          <div className="rounded-md bg-blue-50 dark:bg-blue-950 p-3">
            <p className="text-sm">{response.executive_summary}</p>
          </div>

          {/* Counts */}
          <div className="flex gap-4 text-sm">
            <span>Total: <strong>{response.total_alerts}</strong></span>
            <span style={{ color: SEVERITY_COLORS.critical }}>
              Critical: <strong>{response.critical_count}</strong>
            </span>
            <span style={{ color: SEVERITY_COLORS.warning }}>
              Warning: <strong>{response.warning_count}</strong>
            </span>
          </div>

          {/* Ranked alerts table */}
          {response.ranked_alerts.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-xs text-muted-foreground">
                    <th className="pb-2 pr-4">Series</th>
                    <th className="pb-2 pr-4">Metric</th>
                    <th className="pb-2 pr-4">Severity</th>
                    <th className="pb-2 pr-4">Impact</th>
                    <th className="pb-2">Suggested Action</th>
                  </tr>
                </thead>
                <tbody>
                  {response.ranked_alerts.map((alert, i) => (
                    <tr key={i} className="border-b last:border-0">
                      <td className="py-2 pr-4 font-medium">{alert.series_id}</td>
                      <td className="py-2 pr-4">{alert.metric}</td>
                      <td className="py-2 pr-4">
                        <span
                          className="rounded px-1.5 py-0.5 text-xs font-medium text-white"
                          style={{ backgroundColor: SEVERITY_COLORS[alert.severity as keyof typeof SEVERITY_COLORS] || SEVERITY_COLORS.info }}
                        >
                          {alert.severity}
                        </span>
                      </td>
                      <td className="py-2 pr-4">{alert.business_impact_score.toFixed(1)}</td>
                      <td className="py-2 text-xs text-muted-foreground">{alert.suggested_action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
