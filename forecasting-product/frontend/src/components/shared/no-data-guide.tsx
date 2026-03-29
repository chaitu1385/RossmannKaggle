"use client";

import { Info } from "lucide-react";
import Link from "next/link";

interface NoDataGuideProps {
  lob: string;
  dataType: "forecast" | "backtest" | "series" | "metrics" | "drift";
}

const GUIDES: Record<
  NoDataGuideProps["dataType"],
  { title: string; message: string; nextStep: string; nextHref: string }
> = {
  forecast: {
    title: "No forecast data yet",
    message: "Run a forecast pipeline first to generate predictions for this LOB.",
    nextStep: "Go to Data Onboarding",
    nextHref: "/data-onboarding",
  },
  backtest: {
    title: "No backtest results yet",
    message: "Run a backtest to evaluate models and generate a leaderboard for this LOB.",
    nextStep: "Go to Data Onboarding",
    nextHref: "/data-onboarding",
  },
  series: {
    title: "No series data found",
    message: "Upload actuals data to start exploring series for this LOB.",
    nextStep: "Go to Data Onboarding",
    nextHref: "/data-onboarding",
  },
  metrics: {
    title: "No metric data available",
    message: "Metrics are generated when you run a backtest. Run one first for this LOB.",
    nextStep: "Go to Data Onboarding",
    nextHref: "/data-onboarding",
  },
  drift: {
    title: "No drift alerts",
    message: "Drift detection requires backtest metrics. Run a backtest first for this LOB.",
    nextStep: "Go to Data Onboarding",
    nextHref: "/data-onboarding",
  },
};

export function NoDataGuide({ lob, dataType }: NoDataGuideProps) {
  const guide = GUIDES[dataType];

  return (
    <div className="rounded-lg border border-blue-200 bg-blue-50/50 p-6 text-center dark:border-blue-900 dark:bg-blue-950/20">
      <Info className="mx-auto mb-3 h-8 w-8 text-blue-500" />
      <h3 className="font-semibold text-blue-700 dark:text-blue-400">{guide.title}</h3>
      <p className="mt-1 text-sm text-muted-foreground">
        {guide.message}
      </p>
      <p className="mt-1 text-xs text-muted-foreground">
        LOB: <span className="font-mono">{lob}</span>
      </p>
      <Link
        href={guide.nextHref}
        className="mt-4 inline-block rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90 transition-colors"
      >
        {guide.nextStep}
      </Link>
    </div>
  );
}
