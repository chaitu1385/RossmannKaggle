"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { FVA_COLORS } from "@/lib/constants";
import { formatPct } from "@/lib/utils";

interface FVAEntry {
  layer: string;
  wmape: number;
}

interface Props {
  data: FVAEntry[];
}

export function FVACascade({ data }: Props) {
  const chartData = data.map((d, i) => ({
    ...d,
    color: i === 0 ? FVA_COLORS.BASELINE : d.wmape < data[0].wmape ? FVA_COLORS.ADDS_VALUE : FVA_COLORS.DESTROYS_VALUE,
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="layer" tick={{ fontSize: 12 }} />
        <YAxis tickFormatter={(v: number) => formatPct(v)} tick={{ fontSize: 12 }} />
        <Tooltip formatter={(v: number) => formatPct(v)} />
        <Bar dataKey="wmape" name="WMAPE" radius={[4, 4, 0, 0]}>
          {chartData.map((entry) => (
            <Cell key={entry.layer} fill={entry.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
