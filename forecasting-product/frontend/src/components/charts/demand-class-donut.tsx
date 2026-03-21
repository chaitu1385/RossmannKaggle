"use client";

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from "recharts";
import { DEMAND_CLASS_COLORS } from "@/lib/constants";

interface Props {
  data: Record<string, number>;
}

export function DemandClassDonut({ data }: Props) {
  const chartData = Object.entries(data).map(([name, value]) => ({ name, value }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <PieChart>
        <Pie
          data={chartData}
          cx="50%"
          cy="50%"
          innerRadius={60}
          outerRadius={100}
          dataKey="value"
          nameKey="name"
          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
          labelLine={false}
        >
          {chartData.map((entry) => (
            <Cell
              key={entry.name}
              fill={DEMAND_CLASS_COLORS[entry.name] || "#7f7f7f"}
            />
          ))}
        </Pie>
        <Tooltip />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
}
