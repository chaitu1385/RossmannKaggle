"use client";

import { useState } from "react";
import { FileText, Sparkles } from "lucide-react";
import { api } from "@/lib/api-client";
import type { CommentaryResponse } from "@/lib/types";
import { MetricCard } from "@/components/shared/metric-card";

interface Props {
  lob: string;
  runType?: string;
}

export function CommentaryPanel({ lob, runType = "backtest" }: Props) {
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<CommentaryResponse | null>(null);
  const [error, setError] = useState("");
  const [periodStart, setPeriodStart] = useState("");
  const [periodEnd, setPeriodEnd] = useState("");

  const handleGenerate = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await api.aiCommentary({
        lob,
        run_type: runType,
        period_start: periodStart || undefined,
        period_end: periodEnd || undefined,
      });
      setResponse(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate commentary");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4 rounded-lg border p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold">AI Executive Commentary</h3>
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-end gap-3">
        <div className="space-y-1">
          <label className="text-xs text-muted-foreground">Period Start</label>
          <input
            type="date"
            value={periodStart}
            onChange={(e) => setPeriodStart(e.target.value)}
            className="rounded-md border bg-background px-2 py-1.5 text-sm"
          />
        </div>
        <div className="space-y-1">
          <label className="text-xs text-muted-foreground">Period End</label>
          <input
            type="date"
            value={periodEnd}
            onChange={(e) => setPeriodEnd(e.target.value)}
            className="rounded-md border bg-background px-2 py-1.5 text-sm"
          />
        </div>
        <button
          onClick={handleGenerate}
          disabled={loading}
          className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors"
        >
          <Sparkles className="h-3 w-3" />
          {loading ? "Generating..." : "Generate Commentary"}
        </button>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      {response && !loading && (
        <div className="space-y-4">
          {/* Executive summary */}
          <div className="rounded-md bg-blue-50 dark:bg-blue-950 p-4">
            <p className="text-sm leading-relaxed">{response.executive_summary}</p>
          </div>

          {/* Key metrics */}
          {response.key_metrics.length > 0 && (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {response.key_metrics.map((m) => (
                <MetricCard
                  key={m.name}
                  label={m.name}
                  value={m.unit === "%" ? `${(m.value * 100).toFixed(1)}%` : m.value.toFixed(2)}
                  trend={m.trend}
                />
              ))}
            </div>
          )}

          {/* Exceptions */}
          {response.exceptions.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-muted-foreground uppercase mb-2">
                Exceptions
              </h4>
              <ul className="space-y-1">
                {response.exceptions.map((e, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm">
                    <span className="text-warning">!</span> {e}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Action items */}
          {response.action_items.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-muted-foreground uppercase mb-2">
                Action Items
              </h4>
              <ul className="space-y-1">
                {response.action_items.map((a, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm">
                    <input type="checkbox" className="mt-1 rounded" /> {a}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
