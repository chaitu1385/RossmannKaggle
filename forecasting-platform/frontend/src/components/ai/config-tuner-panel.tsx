"use client";

import { useState } from "react";
import { Settings, Sparkles } from "lucide-react";
import { api } from "@/lib/api-client";
import type { ConfigTuneResponse } from "@/lib/types";
import { RISK_COLORS } from "@/lib/constants";

interface Props {
  lob: string;
  runType?: string;
}

export function ConfigTunerPanel({ lob, runType = "backtest" }: Props) {
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<ConfigTuneResponse | null>(null);
  const [error, setError] = useState("");

  const handleRecommend = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await api.aiRecommendConfig({ lob, run_type: runType });
      setResponse(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to get recommendations");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4 rounded-lg border p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Settings className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold">AI Config Tuner</h3>
        </div>
        <button
          onClick={handleRecommend}
          disabled={loading}
          className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors"
        >
          <Sparkles className="h-3 w-3" />
          {loading ? "Analyzing..." : "Get AI Recommendations"}
        </button>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      {response && !loading && (
        <div className="space-y-4">
          {/* Overall assessment */}
          <div className="rounded-md bg-muted/50 p-3">
            <p className="text-sm">{response.overall_assessment}</p>
            {response.risk_summary && (
              <p className="mt-1 text-xs text-muted-foreground">{response.risk_summary}</p>
            )}
          </div>

          {/* Recommendation cards */}
          {response.recommendations.map((rec, i) => (
            <div key={i} className="rounded-md border p-3 space-y-2">
              <div className="flex items-center justify-between">
                <code className="text-xs font-medium text-primary">{rec.field_path}</code>
                <span
                  className="rounded px-1.5 py-0.5 text-xs font-medium text-white"
                  style={{ backgroundColor: RISK_COLORS[rec.risk] }}
                >
                  {rec.risk} risk
                </span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <span className="rounded bg-red-100 dark:bg-red-900 px-2 py-0.5 text-xs">
                  {String(rec.current_value)}
                </span>
                <span className="text-muted-foreground">&rarr;</span>
                <span className="rounded bg-green-100 dark:bg-green-900 px-2 py-0.5 text-xs">
                  {String(rec.recommended_value)}
                </span>
              </div>
              <p className="text-xs text-muted-foreground">{rec.reasoning}</p>
              {rec.expected_impact && (
                <p className="text-xs font-medium text-primary">{rec.expected_impact}</p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
