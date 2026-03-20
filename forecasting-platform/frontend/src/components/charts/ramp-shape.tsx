"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { COLORS } from "@/lib/constants";
import { formatPct } from "@/lib/utils";

interface Props {
  shape: "linear" | "step" | "exponential";
  periods: number;
  targetProportion: number;
}

function generateRampData(shape: string, periods: number, target: number) {
  const data = [];
  for (let i = 0; i <= periods; i++) {
    const t = i / periods;
    let newShare: number;
    switch (shape) {
      case "step":
        newShare = i >= Math.floor(periods / 2) ? target : 0;
        break;
      case "exponential":
        newShare = target * (1 - Math.exp(-3 * t)) / (1 - Math.exp(-3));
        break;
      default: // linear
        newShare = target * t;
    }
    data.push({
      period: i,
      newSKU: Math.min(newShare, target),
      oldSKU: Math.max(target - newShare, 0),
    });
  }
  return data;
}

export function RampShape({ shape, periods, targetProportion }: Props) {
  const data = generateRampData(shape, periods, targetProportion);

  return (
    <ResponsiveContainer width="100%" height={250}>
      <AreaChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="period" label={{ value: "Period", position: "insideBottomRight" }} />
        <YAxis tickFormatter={(v: number) => formatPct(v)} />
        <Tooltip formatter={(v: number) => formatPct(v)} />
        <Area type="monotone" dataKey="oldSKU" name="Old SKU" stackId="1" fill={COLORS.neutral} stroke={COLORS.neutral} />
        <Area type="monotone" dataKey="newSKU" name="New SKU" stackId="1" fill={COLORS.primary} stroke={COLORS.primary} />
      </AreaChart>
    </ResponsiveContainer>
  );
}
