"use client";

import { useState } from "react";
import { RampShape } from "@/components/charts/ramp-shape";
import { MetricCard } from "@/components/shared/metric-card";
import { SKUMappingPanel } from "@/components/sku/sku-mapping-panel";
import { OverrideManagementPanel } from "@/components/sku/override-management-panel";
import { WorkflowNav } from "@/components/shared/workflow-nav";

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

      <WorkflowNav currentStep="/sku-transitions" />
    </div>
  );
}
