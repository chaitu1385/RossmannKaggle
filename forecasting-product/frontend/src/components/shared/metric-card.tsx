"use client";

import { cn } from "@/lib/utils";
import { TREND_ICONS } from "@/lib/constants";

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  trend?: "improving" | "stable" | "degrading";
  className?: string;
}

const TREND_COLORS: Record<string, string> = {
  improving: "text-green-600 dark:text-green-400",
  stable: "text-muted-foreground",
  degrading: "text-red-600 dark:text-red-400",
};

export function MetricCard({ label, value, unit, trend, className }: MetricCardProps) {
  return (
    <div className={cn("rounded-lg border bg-card p-4", className)}>
      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
        {label}
      </p>
      <div className="mt-1 flex items-baseline gap-1.5">
        <span className="text-2xl font-bold">{value}</span>
        {unit && <span className="text-sm text-muted-foreground">{unit}</span>}
        {trend && (
          <span className={cn("ml-auto text-lg", TREND_COLORS[trend])}>
            {TREND_ICONS[trend]}
          </span>
        )}
      </div>
    </div>
  );
}
