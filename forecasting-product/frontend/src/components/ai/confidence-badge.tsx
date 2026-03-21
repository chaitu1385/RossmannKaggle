"use client";

import { cn } from "@/lib/utils";

interface Props {
  confidence: "high" | "medium" | "low";
}

const STYLES: Record<string, string> = {
  high: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
  medium: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
  low: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
};

export function ConfidenceBadge({ confidence }: Props) {
  return (
    <span className={cn("inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium", STYLES[confidence])}>
      {confidence}
    </span>
  );
}
