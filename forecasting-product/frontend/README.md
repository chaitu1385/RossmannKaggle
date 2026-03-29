# Forecasting Product Frontend

Next.js 15 application (App Router) providing the UI for the Forecasting Platform.

## Quick Start

```bash
npm install
npm run dev          # http://localhost:3000
```

The backend API must be running at `http://localhost:8000` (or set `NEXT_PUBLIC_API_URL`).

## Architecture

```
src/
  app/                  # App Router pages
    login/              # JWT auth (dev token endpoint)
    data-onboarding/    # Upload CSV, auto-analyze, download config YAML
    series-explorer/    # Series listing, SBC classification, breaks, cleansing
    forecast/           # Forecast viewer with fan charts (P10-P90)
    backtest/           # Run backtest, view leaderboard + drift
    hierarchy/          # Build tree, aggregate, reconcile, sunburst viz
    sku-transitions/    # SKU mapping (Phase 1/2) + overrides
    sop/                # S&OP dashboard: FVA, comparison, constraints, AI commentary
    health/             # System health + audit log
  components/
    charts/             # Recharts + Plotly visualizations (fan-chart, sunburst, etc.)
    forecast/           # Decomposition, comparison, constrained forecast panels
    governance/         # Model cards, lineage, BI export
    pipeline/           # Multi-file analysis, pipeline execution
    sku/                # SKU mapping, override management
    layout/             # Sidebar, header
    shared/             # DataTable, LoadingCard, ComingSoon
    ui/                 # shadcn/ui primitives
  hooks/                # React Query hooks + useAsyncOperation
  lib/                  # API client, auth, types, constants
  providers/            # AuthProvider (JWT + RBAC)
```

## Key Dependencies

- **Next.js 15** (React 19, App Router, TypeScript)
- **Tailwind CSS** + **shadcn/ui** (Radix primitives)
- **Recharts** (line, bar, composed charts)
- **Plotly.js** via `react-plotly.js` (fan charts, sunburst) with `next/dynamic` SSR-off
- **@tanstack/react-query** (server state)
- **@tanstack/react-table** (data tables)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API base URL |

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start dev server |
| `npm run build` | Production build |
| `npm run start` | Start production server |
| `npm run lint` | Run Next.js linter |
