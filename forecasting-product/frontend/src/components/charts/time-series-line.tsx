"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { COLORS } from "@/lib/constants";

interface SeriesLine {
  dataKey: string;
  name: string;
  color?: string;
  dashed?: boolean;
}

interface Props {
  data: Record<string, unknown>[];
  xKey: string;
  lines: SeriesLine[];
  height?: number;
}

export function TimeSeriesLine({ data, xKey, lines, height = 300 }: Props) {
  return (
    <div style={{ width: '100%', height }}>
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={xKey} tick={{ fontSize: 11 }} />
        <YAxis tick={{ fontSize: 11 }} />
        <Tooltip />
        <Legend />
        {lines.map((line) => (
          <Line
            key={line.dataKey}
            type="monotone"
            dataKey={line.dataKey}
            name={line.name}
            stroke={line.color || COLORS.primary}
            strokeDasharray={line.dashed ? "5 5" : undefined}
            dot={false}
            strokeWidth={2}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
    </div>
  );
}
