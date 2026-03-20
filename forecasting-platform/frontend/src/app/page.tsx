"use client";

import Link from "next/link";
import {
  Upload,
  Search,
  ArrowLeftRight,
  GitBranch,
  Trophy,
  TrendingUp,
  Activity,
  Presentation,
  FlaskConical,
  ClipboardList,
  BarChart3,
  Settings,
  ChevronDown,
} from "lucide-react";
import { NAV_ITEMS, GLOSSARY } from "@/lib/constants";
import { useState } from "react";

const ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  Upload, Search, ArrowLeftRight, GitBranch,
  Trophy, TrendingUp, Activity, Presentation,
};

const PERSONAS = [
  {
    title: "Data Scientist",
    icon: FlaskConical,
    description: "Onboard data, explore series, backtest models, analyze forecasts",
    pages: ["/data-onboarding", "/series-explorer", "/backtest", "/forecast"],
    color: "bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800",
  },
  {
    title: "Demand Planner",
    icon: ClipboardList,
    description: "View forecasts, manage SKU transitions, apply overrides",
    pages: ["/forecast", "/sku-transitions", "/hierarchy"],
    color: "bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800",
  },
  {
    title: "S&OP Leader",
    icon: BarChart3,
    description: "Review executive summaries, governance, BI exports",
    pages: ["/forecast", "/health", "/sop"],
    color: "bg-purple-50 dark:bg-purple-950 border-purple-200 dark:border-purple-800",
  },
  {
    title: "Platform Engineer",
    icon: Settings,
    description: "Monitor platform health, drift, audit log, cost tracking",
    pages: ["/data-onboarding", "/health"],
    color: "bg-orange-50 dark:bg-orange-950 border-orange-200 dark:border-orange-800",
  },
];

export default function HomePage() {
  const [glossaryOpen, setGlossaryOpen] = useState(false);

  return (
    <div className="mx-auto max-w-6xl space-y-12">
      {/* Hero */}
      <section className="space-y-4 text-center">
        <h1 className="text-4xl font-bold tracking-tight">
          Sales Forecasting Platform
        </h1>
        <p className="mx-auto max-w-2xl text-lg text-muted-foreground">
          Multi-frequency forecasting for retail S&amp;OP. Combines statistical,
          ML, neural, and foundation models with hierarchical reconciliation and
          AI-powered insights.
        </p>
      </section>

      {/* Persona Quick-Start */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Quick Start by Role</h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {PERSONAS.map((persona) => (
            <div
              key={persona.title}
              className={`rounded-lg border p-5 transition-shadow hover:shadow-md ${persona.color}`}
            >
              <persona.icon className="mb-3 h-8 w-8 text-primary" />
              <h3 className="font-semibold">{persona.title}</h3>
              <p className="mt-1 text-sm text-muted-foreground">
                {persona.description}
              </p>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {persona.pages.map((href) => {
                  const nav = NAV_ITEMS.find((n) => n.href === href);
                  return nav ? (
                    <Link
                      key={href}
                      href={href}
                      className="rounded bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary hover:bg-primary/20 transition-colors"
                    >
                      {nav.title}
                    </Link>
                  ) : null;
                })}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Workflow Overview */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Workflow</h2>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {NAV_ITEMS.map((item, idx) => {
            const Icon = ICONS[item.icon];
            return (
              <Link
                key={item.href}
                href={item.href}
                className="group flex items-start gap-3 rounded-lg border p-4 transition-all hover:border-primary hover:shadow-sm"
              >
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                  {Icon ? <Icon className="h-4 w-4" /> : <span className="text-xs font-bold">{idx + 1}</span>}
                </div>
                <div>
                  <p className="text-sm font-medium">
                    <span className="text-muted-foreground">{idx + 1}.</span>{" "}
                    {item.title}
                  </p>
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    {item.description}
                  </p>
                </div>
              </Link>
            );
          })}
        </div>
      </section>

      {/* Glossary */}
      <section className="space-y-2">
        <button
          onClick={() => setGlossaryOpen(!glossaryOpen)}
          className="flex items-center gap-2 text-xl font-semibold hover:text-primary transition-colors"
        >
          Glossary
          <ChevronDown
            className={`h-5 w-5 transition-transform ${glossaryOpen ? "rotate-180" : ""}`}
          />
        </button>
        {glossaryOpen && (
          <div className="grid grid-cols-1 gap-3 rounded-lg border p-4 sm:grid-cols-2">
            {GLOSSARY.map((item) => (
              <div key={item.term}>
                <dt className="text-sm font-semibold text-primary">
                  {item.term}
                </dt>
                <dd className="text-sm text-muted-foreground">
                  {item.definition}
                </dd>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
