"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { RampShape } from "@/components/charts/ramp-shape";
import { MetricCard } from "@/components/shared/metric-card";
import { api } from "@/lib/api-client";
import type { SKUMappingResponse, OverrideItem, CreateOverrideRequest } from "@/lib/types";

// ── SKU Mapping Pipeline Panel ──────────────────────────────────────────────

function SKUMappingPanel() {
  const [phase, setPhase] = useState<1 | 2>(1);
  const [productFile, setProductFile] = useState<File | null>(null);
  const [salesFile, setSalesFile] = useState<File | null>(null);
  const [launchWindowDays, setLaunchWindowDays] = useState(180);
  const [minBaseSimilarity, setMinBaseSimilarity] = useState(0.6);
  const [minConfidence, setMinConfidence] = useState("medium");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SKUMappingResponse | null>(null);

  const handleRun = async () => {
    if (!productFile) {
      setError("Please upload a product master CSV file.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      let response: SKUMappingResponse;
      if (phase === 1) {
        response = await api.skuMappingPhase1(productFile, {
          launch_window_days: launchWindowDays,
          min_base_similarity: minBaseSimilarity,
          min_confidence: minConfidence,
        });
      } else {
        response = await api.skuMappingPhase2(
          productFile,
          salesFile ?? undefined,
          {
            launch_window_days: String(launchWindowDays),
            min_base_similarity: String(minBaseSimilarity),
            min_confidence: minConfidence,
          },
        );
      }
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "SKU mapping request failed.");
    } finally {
      setLoading(false);
    }
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

// ── Planner Override Management Panel ────────────────────────────────────────

const EMPTY_OVERRIDE_FORM: CreateOverrideRequest = {
  old_sku: "",
  new_sku: "",
  proportion: 1.0,
  scenario: "base",
  ramp_shape: "linear",
  effective_date: "",
  notes: "",
};

function OverrideManagementPanel() {
  const [overrides, setOverrides] = useState<OverrideItem[]>([]);
  const [overrideCount, setOverrideCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState<CreateOverrideRequest>({ ...EMPTY_OVERRIDE_FORM });
  const [submitting, setSubmitting] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const mounted = useRef(true);

  const fetchOverrides = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.listOverrides();
      if (mounted.current) {
        setOverrides(res.overrides);
        setOverrideCount(res.count);
      }
    } catch (err) {
      if (mounted.current) {
        setError(err instanceof Error ? err.message : "Failed to load overrides.");
      }
    } finally {
      if (mounted.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    fetchOverrides();
    return () => { mounted.current = false; };
  }, [fetchOverrides]);

  const handleCreate = async () => {
    if (!form.old_sku || !form.new_sku) {
      setError("old_sku and new_sku are required.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await api.createOverride(form);
      setForm({ ...EMPTY_OVERRIDE_FORM });
      setShowForm(false);
      await fetchOverrides();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create override.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleDelete = async (id: string) => {
    setDeletingId(id);
    setError(null);
    try {
      await api.deleteOverride(id);
      await fetchOverrides();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete override.");
    } finally {
      setDeletingId(null);
    }
  };

  const updateField = <K extends keyof CreateOverrideRequest>(key: K, value: CreateOverrideRequest[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <section className="space-y-4">
      <h2 className="text-lg font-semibold">Planner Override Management</h2>
      <div className="rounded-lg border p-4 space-y-4">
        {/* Header row */}
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">
            {loading ? "Loading overrides..." : `${overrideCount} override(s)`}
          </span>
          <button
            onClick={() => setShowForm(!showForm)}
            className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
          >
            {showForm ? "Cancel" : "Add Override"}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        )}

        {/* Add Override Form */}
        {showForm && (
          <div className="rounded-md border bg-muted/30 p-4 space-y-3">
            <h3 className="text-sm font-medium">New Override</h3>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              <div className="space-y-1">
                <label className="text-xs font-medium">Old SKU *</label>
                <input
                  type="text"
                  value={form.old_sku}
                  onChange={(e) => updateField("old_sku", e.target.value)}
                  className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                  placeholder="e.g. SKU-001"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium">New SKU *</label>
                <input
                  type="text"
                  value={form.new_sku}
                  onChange={(e) => updateField("new_sku", e.target.value)}
                  className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                  placeholder="e.g. SKU-002"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium">Proportion</label>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={form.proportion}
                  onChange={(e) => updateField("proportion", Number(e.target.value))}
                  className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium">Scenario</label>
                <input
                  type="text"
                  value={form.scenario ?? ""}
                  onChange={(e) => updateField("scenario", e.target.value)}
                  className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                  placeholder="base"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium">Ramp Shape</label>
                <select
                  value={form.ramp_shape ?? "linear"}
                  onChange={(e) => updateField("ramp_shape", e.target.value)}
                  className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                >
                  <option value="linear">Linear</option>
                  <option value="step">Step</option>
                  <option value="exponential">Exponential</option>
                </select>
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium">Effective Date</label>
                <input
                  type="date"
                  value={form.effective_date ?? ""}
                  onChange={(e) => updateField("effective_date", e.target.value)}
                  className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                />
              </div>
              <div className="space-y-1 sm:col-span-2 lg:col-span-3">
                <label className="text-xs font-medium">Notes</label>
                <input
                  type="text"
                  value={form.notes ?? ""}
                  onChange={(e) => updateField("notes", e.target.value)}
                  className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                  placeholder="Optional notes"
                />
              </div>
            </div>
            <button
              onClick={handleCreate}
              disabled={submitting}
              className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
            >
              {submitting ? "Creating..." : "Create Override"}
            </button>
          </div>
        )}

        {/* Overrides table */}
        {loading ? (
          <div className="space-y-2">
            <div className="h-8 animate-pulse rounded bg-muted" />
            <div className="h-8 animate-pulse rounded bg-muted" />
            <div className="h-8 animate-pulse rounded bg-muted" />
          </div>
        ) : overrides.length === 0 ? (
          <p className="text-sm text-muted-foreground">No overrides found.</p>
        ) : (
          <div className="overflow-x-auto rounded-md border">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="px-3 py-2 text-left font-medium">Old SKU</th>
                  <th className="px-3 py-2 text-left font-medium">New SKU</th>
                  <th className="px-3 py-2 text-left font-medium">Proportion</th>
                  <th className="px-3 py-2 text-left font-medium">Scenario</th>
                  <th className="px-3 py-2 text-left font-medium">Ramp Shape</th>
                  <th className="px-3 py-2 text-left font-medium">Effective Date</th>
                  <th className="px-3 py-2 text-left font-medium">Notes</th>
                  <th className="px-3 py-2 text-left font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {overrides.map((o) => (
                  <tr key={o.override_id ?? `${o.old_sku}-${o.new_sku}`} className="border-b last:border-0">
                    <td className="px-3 py-1.5">{o.old_sku}</td>
                    <td className="px-3 py-1.5">{o.new_sku}</td>
                    <td className="px-3 py-1.5">{o.proportion}</td>
                    <td className="px-3 py-1.5">{o.scenario}</td>
                    <td className="px-3 py-1.5">{o.ramp_shape}</td>
                    <td className="px-3 py-1.5">{o.effective_date ?? ""}</td>
                    <td className="px-3 py-1.5">{o.notes ?? ""}</td>
                    <td className="px-3 py-1.5">
                      {o.override_id && (
                        <button
                          onClick={() => handleDelete(o.override_id!)}
                          disabled={deletingId === o.override_id}
                          className="rounded px-2 py-1 text-xs font-medium text-destructive hover:bg-destructive/10 disabled:opacity-50"
                        >
                          {deletingId === o.override_id ? "Deleting..." : "Delete"}
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}

// ── Main Page ────────────────────────────────────────────────────────────────

export default function SKUTransitionsPage() {
  const [rampShape, setRampShape] = useState<"linear" | "step" | "exponential">("linear");
  const [periods, setPeriods] = useState(12);
  const [proportion, setProportion] = useState(0.8);

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold">SKU Transitions</h1>
        <p className="text-sm text-muted-foreground">
          Map old-to-new SKUs, manage planner overrides, and configure transition ramp shapes.
        </p>
      </div>

      {/* Demo overview cards */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricCard label="Total SKUs" value="1,250" />
        <MetricCard label="Active Transitions" value="34" />
        <MetricCard label="Pending Overrides" value="8" />
        <MetricCard label="Avg Confidence" value="0.82" />
      </div>

      {/* SKU Mapping Pipeline */}
      <SKUMappingPanel />

      {/* Planner Overrides */}
      <OverrideManagementPanel />

      {/* Ramp Shape Preview — this works client-side */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Transition Ramp Shape Preview</h2>
        <div className="rounded-lg border p-4 space-y-4">
          <div className="flex flex-wrap items-end gap-4">
            <div className="space-y-1">
              <label className="text-xs font-medium">Shape</label>
              <select
                value={rampShape}
                onChange={(e) => setRampShape(e.target.value as "linear" | "step" | "exponential")}
                className="rounded-md border bg-background px-3 py-1.5 text-sm"
              >
                <option value="linear">Linear</option>
                <option value="step">Step</option>
                <option value="exponential">Exponential</option>
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium">Transition Periods: {periods}</label>
              <input
                type="range"
                min={1}
                max={52}
                value={periods}
                onChange={(e) => setPeriods(Number(e.target.value))}
                className="w-40"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium">
                Target Proportion: {(proportion * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min={10}
                max={100}
                value={proportion * 100}
                onChange={(e) => setProportion(Number(e.target.value) / 100)}
                className="w-40"
              />
            </div>
          </div>
          <RampShape shape={rampShape} periods={periods} targetProportion={proportion} />
        </div>
      </section>
    </div>
  );
}
