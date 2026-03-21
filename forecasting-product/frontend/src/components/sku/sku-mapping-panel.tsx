"use client";

import { useState } from "react";
import { useAsyncOperation } from "@/hooks/use-async-operation";
import { api } from "@/lib/api-client";
import type { SKUMappingResponse } from "@/lib/types";

export function SKUMappingPanel() {
  const [phase, setPhase] = useState<1 | 2>(1);
  const [productFile, setProductFile] = useState<File | null>(null);
  const [salesFile, setSalesFile] = useState<File | null>(null);
  const [launchWindowDays, setLaunchWindowDays] = useState(180);
  const [minBaseSimilarity, setMinBaseSimilarity] = useState(0.6);
  const [minConfidence, setMinConfidence] = useState("medium");
  const { result, loading, error, run, setError } = useAsyncOperation<SKUMappingResponse>();

  const handleRun = () => {
    if (!productFile) {
      setError("Please upload a product master CSV file.");
      return;
    }
    run(() => {
      if (phase === 1) {
        return api.skuMappingPhase1(productFile, {
          launch_window_days: launchWindowDays,
          min_base_similarity: minBaseSimilarity,
          min_confidence: minConfidence,
        });
      } else {
        return api.skuMappingPhase2(
          productFile,
          salesFile ?? undefined,
          {
            launch_window_days: String(launchWindowDays),
            min_base_similarity: String(minBaseSimilarity),
            min_confidence: minConfidence,
          },
        );
      }
    });
  };

  return (
    <section className="space-y-4">
      <h2 className="text-lg font-semibold">SKU Mapping Pipeline</h2>
      <div className="rounded-lg border p-4 space-y-4">
        {/* Phase selector */}
        <div className="flex flex-wrap items-end gap-4">
          <div className="space-y-1">
            <label className="text-xs font-medium">Phase</label>
            <select
              value={phase}
              onChange={(e) => setPhase(Number(e.target.value) as 1 | 2)}
              className="rounded-md border bg-background px-3 py-1.5 text-sm"
            >
              <option value={1}>Phase 1 (Attribute + Naming)</option>
              <option value={2}>Phase 2 (Full)</option>
            </select>
          </div>
        </div>

        {/* File uploads */}
        <div className="flex flex-wrap items-end gap-4">
          <div className="space-y-1">
            <label className="text-xs font-medium">Product Master CSV</label>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setProductFile(e.target.files?.[0] ?? null)}
              className="block text-sm file:mr-2 file:rounded-md file:border-0 file:bg-primary file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-primary-foreground hover:file:bg-primary/90"
            />
          </div>
          {phase === 2 && (
            <div className="space-y-1">
              <label className="text-xs font-medium">Sales History CSV (optional)</label>
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setSalesFile(e.target.files?.[0] ?? null)}
                className="block text-sm file:mr-2 file:rounded-md file:border-0 file:bg-primary file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-primary-foreground hover:file:bg-primary/90"
              />
            </div>
          )}
        </div>

        {/* Parameters */}
        <div className="flex flex-wrap items-end gap-4">
          <div className="space-y-1">
            <label className="text-xs font-medium">Launch Window (days)</label>
            <input
              type="number"
              min={1}
              value={launchWindowDays}
              onChange={(e) => setLaunchWindowDays(Number(e.target.value))}
              className="w-28 rounded-md border bg-background px-3 py-1.5 text-sm"
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs font-medium">Min Base Similarity</label>
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={minBaseSimilarity}
              onChange={(e) => setMinBaseSimilarity(Number(e.target.value))}
              className="w-28 rounded-md border bg-background px-3 py-1.5 text-sm"
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs font-medium">Min Confidence</label>
            <select
              value={minConfidence}
              onChange={(e) => setMinConfidence(e.target.value)}
              className="rounded-md border bg-background px-3 py-1.5 text-sm"
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>
        </div>

        {/* Run button */}
        <button
          onClick={handleRun}
          disabled={loading || !productFile}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {loading ? "Running..." : `Run Phase ${phase}`}
        </button>

        {/* Error */}
        {error && (
          <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <span className="text-sm font-medium">
                Phase {result.phase} &mdash; Total Mappings: {result.total_mappings}
              </span>
            </div>
            {result.mappings.length > 0 ? (
              <div className="overflow-x-auto rounded-md border">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b bg-muted/50">
                      {Object.keys(result.mappings[0]).map((col) => (
                        <th key={col} className="px-3 py-2 text-left font-medium">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.mappings.map((row, idx) => (
                      <tr key={idx} className="border-b last:border-0">
                        {Object.values(row).map((val, ci) => (
                          <td key={ci} className="px-3 py-1.5">
                            {val == null ? "" : String(val)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">No mappings found.</p>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
