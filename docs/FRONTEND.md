# Frontend Architecture

Next.js 15 frontend for the Forecasting Product, providing a workflow-driven UI for S&OP forecasting operations.

## Tech Stack

| Category | Technology | Version |
|----------|-----------|---------|
| Framework | Next.js (App Router) | 15.1 |
| Language | TypeScript | 5.7 |
| UI | React | 19 |
| Styling | Tailwind CSS | 3.4 |
| Primitives | Radix UI | Various |
| Charts | Recharts + Plotly.js | 2.15 / 2.35 |
| Server State | TanStack React Query | 5.62 |
| Tables | TanStack React Table | 8.20 |
| Auth | NextAuth | 4.24 |
| Icons | Lucide React | 0.468 |
| Unit Tests | Vitest + Testing Library | 4.1 / 16.3 |
| E2E Tests | Playwright | 1.58 |

## Provider Hierarchy

The root layout (`src/app/layout.tsx`) wraps the entire app in three providers:

```
QueryProvider          -- TanStack React Query (staleTime: 5min, retry: 1)
  AuthProvider         -- JWT auth state, login/logout, permission checks
    LobProvider        -- Selected LOB (persisted to localStorage)
      <Sidebar />
      <Header />
      <main>{children}</main>
```

## Pages (10 routes)

| Route | Page | Description |
|-------|------|-------------|
| `/` | Dashboard | Home page |
| `/login` | Login | Username + role selection, JWT token acquisition |
| `/data-onboarding` | Data Onboarding | File upload, auto-analysis, config generation |
| `/series-explorer` | Series Explorer | SBC classification, break detection, cleansing audit, regressor screening |
| `/hierarchy` | Hierarchy | Tree building, sunburst visualization, aggregation, reconciliation |
| `/backtest` | Backtest Results | Pipeline execution, leaderboard, FVA, calibration, SHAP, AI config tuner |
| `/forecast` | Forecast Viewer | Fan chart, decomposition, comparison, constraints, NL query |
| `/sku-transitions` | SKU Transitions | Phase 1/2 SKU mapping, planner override management |
| `/sop` | S&OP Meeting | AI commentary, model cards, lineage, BI export |
| `/health` | Platform Health | Drift monitoring, alert triage, audit log, cost tracking |

## Component Structure

```
src/components/
  ai/                  # 5 components — AI-powered panels
    nl-query-panel     # Natural language Q&A about series
    triage-panel       # Drift alert triage with business impact scoring
    config-tuner-panel # AI-recommended config changes
    commentary-panel   # Executive S&OP commentary generation
    confidence-badge   # High/Medium/Low confidence indicator
  charts/              # 11 components — Data visualization
    time-series-line   # Basic time series line chart (Recharts)
    fan-chart          # Forecast with prediction intervals
    leaderboard-bar    # Model accuracy comparison bars
    fva-cascade        # FVA waterfall chart
    drift-histogram    # Drift alert distribution
    calibration-plot   # Prediction interval calibration
    hierarchy-sunburst # Hierarchy tree sunburst (Plotly)
    demand-class-donut # SBC demand class distribution
    forecastability-gauge # Forecastability score gauge
    break-timeline     # Structural break visualization
    cleansing-overlay  # Before/after cleansing overlay
    ramp-shape         # SKU ramp shape visualization
  data/                # 3 components — Data management
    file-upload        # Drag-and-drop file upload
    data-table         # TanStack Table with sorting/filtering
    config-viewer      # YAML config display
  forecast/            # 4 components — Forecast analysis
    decomposition-panel      # STL trend/seasonal/residual
    comparison-panel         # Model vs external forecast
    constrained-forecast-panel # Capacity/budget constraint application
    cross-run-comparison-panel # Cross-run forecast comparison
  governance/          # 3 components — Model governance
    model-cards-panel  # Model card registry viewer
    lineage-panel      # Forecast lineage history
    bi-export-panel    # BI report export (CSV/Parquet)
  layout/              # 2 components — App shell
    sidebar            # Navigation sidebar with workflow steps
    header             # Top bar with LOB selector, user menu
  pipeline/            # 2 components — Pipeline operations
    multi-file-panel   # Multi-file upload with auto-classification
    pipeline-execution-panel # Backtest/forecast pipeline runner
  shared/              # 6 components — Reusable utilities
    error-boundary     # React error boundary
    loading-skeleton   # Skeleton loading states
    metric-card        # KPI card with trend indicator
    no-data-guide      # Empty state with next-step guidance
    coming-soon        # Placeholder for unreleased features
    workflow-nav       # Step-by-step workflow navigation
  sku/                 # 2 components — SKU management
    sku-mapping-panel  # Phase 1/2 SKU transition mapping
    override-management-panel # Override CRUD with ramp shapes
```

## API Client

The centralized API client (`src/lib/api-client.ts`) provides typed methods for all 41 backend endpoints.

**Features:**
- Automatic JWT token injection from `localStorage`
- Token expiry detection (clears session on expired JWT)
- Retry logic: 3 retries with exponential backoff (1s, 2s, 4s)
- Retryable status codes: 502, 503, 504, 429
- FormData handling for file uploads (auto Content-Type boundary)
- Custom `ApiError` class with HTTP status code

**Usage:**
```typescript
import { api } from "@/lib/api-client";

// Typed response — ForecastResponse
const forecasts = await api.getForecasts("retail", { horizon: 12 });

// File upload — FormData handled automatically
const analysis = await api.analyze(file, "my_lob", true);

// AI endpoint
const answer = await api.aiExplain({ series_id: "SKU_001", question: "Why is this forecast high?", lob: "retail" });
```

## State Management

| Concern | Mechanism | Details |
|---------|-----------|---------|
| Server state | React Query | `staleTime: 5min`, `retry: 1`, `refetchOnWindowFocus: false` |
| Auth state | `AuthProvider` context | JWT + user object in `localStorage`, `useAuth()` hook |
| LOB selection | `LobProvider` context | Persisted to `localStorage`, `useLob()` hook |
| Async operations | `useAsyncOperation` hook | Generic loading/error/result state for ad-hoc API calls |

## Custom Hooks

| Hook | File | Purpose |
|------|------|---------|
| `useAsyncOperation<T>` | `hooks/use-async-operation.ts` | Generic async state (loading, error, result, run, reset) |
| `useForecast` | `hooks/use-forecast.ts` | Forecast data fetching |
| `useLeaderboard` | `hooks/use-leaderboard.ts` | Model leaderboard data |
| `useDrift` | `hooks/use-drift.ts` | Drift alert data |
| `useAnalyze` | `hooks/use-analyze.ts` | File upload analysis |
| `useAudit` | `hooks/use-audit.ts` | Audit log querying |

## Auth Flow

1. User enters username and selects role on `/login`
2. `AuthProvider.login()` calls `POST /auth/token?username=...&role=...`
3. JWT access token stored in `localStorage` under `access_token`
4. User object (username, role, permissions) stored under `user`
5. On mount, `AuthProvider` restores session and checks JWT expiry
6. `api-client.ts` injects `Authorization: Bearer <token>` on every request
7. Expired tokens are detected client-side (JWT `exp` claim) and session is cleared

**Roles and Permissions:**

| Role | Permissions |
|------|-------------|
| `admin` | All 11 permissions |
| `data_scientist` | VIEW_FORECASTS, VIEW_METRICS, VIEW_AUDIT_LOG, RUN_BACKTEST, RUN_PIPELINE, PROMOTE_MODEL, MODIFY_CONFIG |
| `planner` | VIEW_FORECASTS, VIEW_METRICS, CREATE_OVERRIDE, DELETE_OVERRIDE |
| `manager` | VIEW_FORECASTS, VIEW_METRICS, VIEW_AUDIT_LOG, APPROVE_OVERRIDE |
| `viewer` | VIEW_FORECASTS, VIEW_METRICS |

Use `useAuth().hasPermission("RUN_PIPELINE")` to check permissions in components.

## Type System

All TypeScript interfaces (`src/lib/types.ts`) mirror the FastAPI Pydantic schemas (`src/api/schemas.py`):

- 46 exported interfaces/types covering all API request and response shapes
- Strict typing for roles (`Role` union type), severity levels, trends, confidence levels
- `Record<string, unknown>` used for flexible/dynamic data (e.g., per-series stats)

## Styling

- **Tailwind CSS 3.4** for utility-first styling
- **Radix UI** primitives for accessible interactive components (dialogs, dropdowns, tabs, tooltips, toasts, accordions, progress bars)
- **`class-variance-authority`** + **`clsx`** + **`tailwind-merge`** for variant-based component styling
- Color constants defined in `src/lib/constants.ts` (severity, demand class, model layer, FVA, confidence, risk)

## Testing

- **Unit tests:** Vitest + React Testing Library (`npm run test:e2e` — note: script name is misleading, runs Vitest)
- **E2E tests:** Playwright (`@playwright/test` in devDependencies)
- **Test runner:** `jsdom` environment for component tests

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API base URL |

## Development

```bash
cd forecasting-product/frontend
npm install
npm run dev          # Start dev server (port 3000)
npm run build        # Production build
npm run lint         # ESLint
npm run test:e2e     # Run Vitest unit tests
```
