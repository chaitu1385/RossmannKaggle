"use client";

import { RadialBarChart, RadialBar, ResponsiveContainer } from "recharts";

interface Props {
  value: number; // 0-1
}

function getColor(v: number): string {
  if (v >= 0.7) return "#2ca02c";
  if (v >= 0.4) return "#ff7f0e";
  return "#d62728";
}

function getLabel(v: number): string {
  if (v >= 0.7) return "High";
  if (v >= 0.4) return "Medium";
  return "Low";
}

export function ForecastabilityGauge({ value }: Props) {
  const color = getColor(value);
  const data = [{ value: value * 100, fill: color }];

  return (
    <div className="flex flex-col items-center">
      <div style={{ width: 200, height: 160 }}>
      <ResponsiveContainer width={200} height={160}>
        <RadialBarChart
          cx="50%"
          cy="100%"
          innerRadius="70%"
          outerRadius="100%"
          startAngle={180}
          endAngle={0}
          data={data}
          barSize={16}
        >
          <RadialBar
            dataKey="value"
            cornerRadius={8}
            background={{ fill: "hsl(var(--muted))" }}
          />
        </RadialBarChart>
      </ResponsiveContainer>
      </div>
      <div className="mt-[-2rem] text-center">
        <p className="text-2xl font-bold" style={{ color }}>
          {(value * 100).toFixed(0)}%
        </p>
        <p className="text-xs text-muted-foreground">{getLabel(value)} Forecastability</p>
      </div>
    </div>
  );
}
