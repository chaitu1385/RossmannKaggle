"use client";

import Link from "next/link";

const WORKFLOW_STEPS = [
  { href: "/data-onboarding", label: "Data Onboarding" },
  { href: "/series-explorer", label: "Series Explorer" },
  { href: "/hierarchy", label: "Hierarchy" },
  { href: "/sku-transitions", label: "SKU Transitions" },
  { href: "/backtest", label: "Backtest" },
  { href: "/forecast", label: "Forecast" },
  { href: "/sop", label: "S&OP" },
  { href: "/health", label: "Health" },
] as const;

interface WorkflowNavProps {
  currentStep: (typeof WORKFLOW_STEPS)[number]["href"];
}

export function WorkflowNav({ currentStep }: WorkflowNavProps) {
  const currentIndex = WORKFLOW_STEPS.findIndex((s) => s.href === currentStep);
  const prev = currentIndex > 0 ? WORKFLOW_STEPS[currentIndex - 1] : null;
  const next = currentIndex < WORKFLOW_STEPS.length - 1 ? WORKFLOW_STEPS[currentIndex + 1] : null;

  if (!prev && !next) return null;

  return (
    <div className="flex items-center justify-between border-t pt-6 mt-8">
      <div className="flex-1">
        {prev && (
          <Link
            href={prev.href}
            className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <span aria-hidden="true">&larr;</span>
            <span>
              <span className="font-medium">Previous:</span> {prev.label}
            </span>
          </Link>
        )}
      </div>
      <div className="flex-1 text-right">
        {next && (
          <Link
            href={next.href}
            className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <span>
              <span className="font-medium">Next:</span> {next.label}
            </span>
            <span aria-hidden="true">&rarr;</span>
          </Link>
        )}
      </div>
    </div>
  );
}
