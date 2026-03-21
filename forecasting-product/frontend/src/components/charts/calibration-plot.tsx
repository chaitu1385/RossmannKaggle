"use client";

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { COLORS } from "@/lib/constants";
import { formatPct } from "@/lib/utils";

interface CalibrationPoint {
  nominal: number;
  empirical: number;
  model?: string;
}

interface Props {
  data: CalibrationPoint[];
}

export function CalibrationPlot({ data }: Props) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="nominal"
          type="number"
          domain={[0, 1]}
          name="Nominal Coverage"
          tickFormatter={(v: number) => formatPct(v)}
          tick={{ fontSize: 11 }}
        />
        <YAxis
          dataKey="empirical"
          type="number"
          domain={[0, 1]}
          name="Empirical Coverage"
          tickFormatter={(v: number) => formatPct(v)}
          tick={{ fontSize: 11 }}
        />
        <Tooltip formatter={(v: number) => formatPct(v)} />
        <ReferenceLine
          segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
          stroke={COLORS.neutral}
          strokeDasharray="5 5"
        />
        <Scatter data={data} fill={COLORS.primary} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
