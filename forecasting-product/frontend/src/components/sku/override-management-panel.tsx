"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { useAsyncOperation } from "@/hooks/use-async-operation";
import { api } from "@/lib/api-client";
import type { OverrideItem, CreateOverrideRequest } from "@/lib/types";

const EMPTY_OVERRIDE_FORM: CreateOverrideRequest = {
  old_sku: "",
  new_sku: "",
  proportion: 1.0,
  scenario: "base",
  ramp_shape: "linear",
  effective_date: "",
  notes: "",
};

export function OverrideManagementPanel() {
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
                  onChange={(e) => setForm((prev) => ({ ...prev, old_sku: e.target.value }))}
                  className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                  placeholder="e.g. SKU-001"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium">New SKU *</label>
                <input
                  type="text"
                  value={form.new_sku}
                  onChange={(e) => setForm((prev) => ({ ...prev, new_sku: e.target.value }))}
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
                  onChange={(e) => setForm((prev) => ({ ...prev, proportion: Number(e.target.value) }))}
                  className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium">Scenario</label>
                <input
                  type="text"
                  value={form.scenario ?? ""}
                  onChange={(e) => setForm((prev) => ({ ...prev, scenario: e.target.value }))}
                  className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                  placeholder="base"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium">Ramp Shape</label>
                <select
                  value={form.ramp_shape ?? "linear"}
                  onChange={(e) => setForm((prev) => ({ ...prev, ramp_shape: e.target.value }))}
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
                  onChange={(e) => setForm((prev) => ({ ...prev, effective_date: e.target.value }))}
                  className="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                />
              </div>
              <div className="space-y-1 sm:col-span-2 lg:col-span-3">
                <label className="text-xs font-medium">Notes</label>
                <input
                  type="text"
                  value={form.notes ?? ""}
                  onChange={(e) => setForm((prev) => ({ ...prev, notes: e.target.value }))}
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
