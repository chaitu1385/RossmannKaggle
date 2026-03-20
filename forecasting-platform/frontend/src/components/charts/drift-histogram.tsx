"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { SEVERITY_COLORS } from "@/lib/constants";
import type { DriftAlertItem } from "@/lib/types";

interface Props {
  alerts: DriftAlertItem[];
}

export function DriftHistogram({ alerts }: Props) {
  // Group by metric, count by severity
  const grouped: Record<string, { metric: string; critical: number; warning: number }> = {};
  for (const a of alerts) {
    if (!grouped[a.metric]) {
      grouped[a.metric] = { metric: a.metric, critical: 0, warning: 0 };
    }
    grouped[a.metric][a.severity]++;
  }
  const data = Object.values(grouped);

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="metric" tick={{ fontSize: 12 }} />
        <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
        <Tooltip />
        <Legend />
        <Bar dataKey="critical" name="Critical" fill={SEVERITY_COLORS.critical} stackId="stack" />
        <Bar dataKey="warning" name="Warning" fill={SEVERITY_COLORS.warning} stackId="stack" />
      </BarChart>
    </ResponsiveContainer>
  );
}
