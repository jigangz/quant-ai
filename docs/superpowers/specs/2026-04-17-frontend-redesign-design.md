# Frontend Redesign Design Spec

**Date**: 2026-04-17  
**Author**: Harry + Claude  
**Status**: Approved (pending implementation)  
**Phase**: 3 Sub-project 4 (after Distributed Systems)

## 1. Goal

Rebuild the 6-page React UI into a pro-grade distributed-quant workbench that is:
- **Beautiful** — Tremor dashboards + shadcn/ui forms + dark Linear-style theme
- **Fast** — TanStack Query caching, no redundant API calls, optimistic updates
- **Responsive** — works on desktop (primary) and mobile (Sheet drawer nav)
- **Professional** — Lightweight Charts candlestick, Cmd+K command palette, proper toasts/dialogs
- **Interview-grade** — every page demo-able without explanation

Removes existing UI pain points (no loading spinners, no validation, no mobile, inconsistent emoji, raw confirm() dialogs, etc.).

## 2. Design Principles

- **Tremor for data, shadcn for interaction** — never mix paradigms within a component
- **No custom chart code** — Tremor AreaChart/LineChart/DonutChart/BarList/Sparkline for analytics; Lightweight Charts for K-line only
- **Query > fetch** — every API call goes through TanStack Query hooks, never raw `useEffect + fetch`
- **Form > state** — every form uses react-hook-form + zod validation, never raw useState for field values
- **Desktop-first, mobile-responsive** — 1440px reference, breakpoints at 1024 / 768 / 375
- **Dark theme only** (this round) — no theme toggle; keep scope tight
- **Components owned, not imported** — shadcn copies into `components/ui/`, we own them

## 3. Tech Stack

| Dep | Version | Purpose |
|-----|---------|---------|
| React | 19 (existing) | Framework |
| Vite | 7 (existing) | Build |
| Tailwind CSS | 3 (existing) | Styling |
| React Router | 7 (existing) | Routing |
| **@tremor/react** | ^3.18 | KPI cards, charts, tables, badges |
| **@radix-ui/react-*** | via shadcn | Primitives for shadcn |
| **shadcn/ui** | (CLI generate) | Form, Dialog, Command, Tabs, Sheet, Toast, Dropdown, Select |
| **lightweight-charts** | ^4.2 | Candlestick + markers |
| **@tanstack/react-query** | ^5 | API caching |
| **zustand** | ^4 | Minimal global store for WS live prices |
| **react-hook-form** | ^7 | Form state |
| **zod** | ^3 | Schema validation |
| **vitest** | ^1 | Unit tests |
| **@testing-library/react** | ^14 | Component tests |
| **@vitejs/plugin-react** | (existing) | Vite plugin |
| **@tailwindcss/forms** | ^0.5 | Form element resets |

## 4. Design Tokens

`tailwind.config.js` extension (keeps Tremor compatibility):

```js
theme: {
  extend: {
    colors: {
      background: "rgb(2 6 23)",      // slate-950
      surface: {
        DEFAULT: "rgb(24 24 27)",     // zinc-900 (cards)
        muted: "rgb(39 39 42)",       // zinc-800
        border: "rgb(63 63 70)",      // zinc-700
      },
      foreground: "rgb(250 250 250)", // zinc-50
      muted: "rgb(161 161 170)",      // zinc-400
      accent: {
        DEFAULT: "rgb(99 102 241)",   // indigo-500
        hover: "rgb(129 140 248)",    // indigo-400
      },
      up: "rgb(16 185 129)",          // emerald-500
      down: "rgb(244 63 94)",         // rose-500
      warn: "rgb(245 158 11)",        // amber-500
      info: "rgb(14 165 233)",        // sky-500
    },
    fontFamily: {
      sans: ["Inter", "system-ui", "sans-serif"],
      mono: ["JetBrains Mono", "monospace"],
    },
    borderRadius: { xl: "0.75rem" },
  }
}
```

Tremor tokens inherited as-is — UI adapters handle bridge.

Dark mode strategy: `darkMode: "class"` on `<html>`, always-on `class="dark"`. No toggle.

## 5. Directory Restructure

```
quant-ai-ui/src/
├── main.jsx                # mount + Providers
├── app/
│   ├── AppShell.jsx        # Sidebar + main outlet
│   ├── Sidebar.jsx         # 64px nav
│   ├── Providers.jsx       # QueryClient + Theme + Toast
│   └── router.jsx          # Route definitions
├── pages/
│   ├── ScreenerPage.jsx
│   ├── DashboardPage.jsx
│   ├── TrainingPage.jsx
│   ├── StrategyPage.jsx
│   ├── TradingPage.jsx
│   └── ExplainPage.jsx
├── features/               # domain-grouped components
│   ├── charts/
│   │   ├── CandlestickChart.jsx
│   │   ├── EquityCurve.jsx
│   │   └── SignalMarkers.jsx
│   ├── trading/
│   │   ├── OrderForm.jsx
│   │   ├── OrderList.jsx
│   │   ├── PortfolioCard.jsx
│   │   ├── PositionsList.jsx
│   │   ├── TradeHistory.jsx
│   │   └── useLivePrices.js
│   ├── training/
│   │   ├── TrainForm.jsx
│   │   ├── EnsembleConfigFields.jsx
│   │   ├── HyperparamSearchFields.jsx
│   │   ├── RunsTable.jsx
│   │   └── ModelsTable.jsx
│   ├── strategy/
│   │   ├── StrategyPicker.jsx
│   │   ├── StrategyParamsForm.jsx
│   │   ├── SignalsVisualization.jsx
│   │   └── BacktestResults.jsx
│   ├── explain/
│   │   ├── ShapFeatureList.jsx
│   │   └── SimilarCasesList.jsx
│   └── screener/
│       └── ScreenerTable.jsx
├── components/
│   ├── ui/                 # shadcn primitives (Button, Dialog, Input, Select, Tabs, Sheet, Command, Toast, Dropdown, AlertDialog, Accordion, Card, Form)
│   ├── PageHeader.jsx
│   ├── EmptyState.jsx
│   ├── LoadingSpinner.jsx
│   ├── ErrorBoundary.jsx
│   ├── ErrorState.jsx
│   ├── TickerSearch.jsx
│   └── ConfirmDialog.jsx
├── api/
│   ├── client.js           # existing fetch wrapper
│   └── queries.js          # TanStack Query hooks
├── stores/
│   └── liveStore.js        # Zustand: live prices + portfolio sync
├── hooks/
│   ├── useWebSocket.js
│   └── useBreakpoint.js
├── lib/
│   ├── utils.js            # shadcn cn() helper
│   └── formatters.js       # price, pct, date formatters
└── __tests__/              # vitest smoke tests
```

## 6. AppShell + Sidebar

### 6.1 AppShell layout

```
┌─────────────────────────────────────────────────────────────┐
│ ┌────┐  ┌──────────────────────────────────────────────┐  │
│ │ 🧠 │  │ <PageHeader title / breadcrumbs / actions>    │  │
│ │    │  ├──────────────────────────────────────────────┤  │
│ │ 📊 │  │                                                │  │
│ │ 📈 │  │                                                │  │
│ │ 🎯 │  │           <Outlet/>  (page content)            │  │
│ │ 🧪 │  │                                                │  │
│ │ 💼 │  │                                                │  │
│ │ 🔍 │  │                                                │  │
│ │    │  │                                                │  │
│ │ ⚙️ │  │                                                │  │
│ └────┘  └──────────────────────────────────────────────┘  │
│  64px                     main (padding 24px)               │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Sidebar

- Width: 64px collapsed (icon-only), 240px on hover (w/ labels)
- Items: Screener / Dashboard / Training / Strategy / Trading / Explain
- Active route: indigo accent pill
- Bottom: theme settings (placeholder, dark locked for now)
- Mobile (< 768px): hidden, replaced by `<Sheet>` drawer toggled by hamburger in `<PageHeader>`

### 6.3 Command palette (Cmd+K)

Global `<TickerSearch>` using shadcn `<Command>`:
- `Cmd+K` anywhere opens palette
- Suggest recent tickers + top screener tickers
- Enter → navigate to `/dashboard?ticker=X`

### 6.4 Toast system

shadcn `<Toast>` for:
- ✅ "Order placed successfully"
- ❌ "Training failed: insufficient data"
- ⏳ "Optimization running (trial 12/50)"
- Auto-dismiss 3s, stack bottom-right

## 7. Page-by-Page Design

### 7.1 Screener `/screener`

```
PageHeader: "Stock Screener"  Actions: [Refresh] [Sort ▾]

<ScreenerTable>:
  Columns: Ticker | Name | Last | Change | Change% (Sparkline) | Volume | Signal
  Sort by: change% (default) or volume, click header
  Row click → /dashboard?ticker=X
  Rows: 10 hot tickers from backend /data/market (repeated per-ticker)
Loading: 10× <Skeleton> rows
Empty: <EmptyState icon="chart" msg="No data" action="Refresh"/>
```

### 7.2 Dashboard `/dashboard?ticker=AAPL`

```
PageHeader: AAPL · Apple Inc · $178.23  [+1.2%]  [Cmd+K switch]

Layout: 2-column grid (60/40 on desktop, stacked on mobile)

Left column (main):
  <CandlestickChart>          // Lightweight Charts v4
    timeframe tabs: [1D] [1W] [1M] [3M] [1Y]
    markers overlay: predicted bullish (up arrow) / bearish (down arrow)
  
Right column:
  <Card title="Prediction">
    <Metric>BUY (87% confidence)</Metric>
    <ProgressBar value={0.87}/>
    Horizon: 5 days · Model: ensemble-stacking-v2
    [Promote model] [View SHAP]
  
  <Card title="SHAP Top Features">
    <BarList>
      rsi_14      ▓▓▓▓▓▓▓ 0.234
      macd        ▓▓▓▓▓   0.187
      ...
      
  <Card title="News Sentiment">
    <Metric>+0.34 (bullish)</Metric>
    Last 7 days · 12 articles · sentiment trend sparkline

Below:
  <Card title="Recent Prediction Activity">
    from consumer /stats/{ticker}
    count, avg_confidence, bullish_ratio
    → showcases distributed Kafka consumer
```

### 7.3 Training `/training`

Tabs (shadcn): **Train** · **Runs** · **Models**

Train tab:
```
<TrainForm>  (react-hook-form + zod)
  Section: Basic
    - Tickers (comma-separated, validated)
    - Date range (DateRangePicker)
    - Model type (Select, triggers show ensemble sub-fields)
  
  Section: Model config (Accordion, expands)
    - If ensemble: <EnsembleConfigFields>
    - Else: model_params JSON editor (monaco-lite or textarea w/ JSON validation)
  
  Section: Features
    - Feature groups (checkbox list from /features/groups)
    - Horizon days (Input number)
    - Train/val ratio (Slider)
  
  Section: Hyperparameter search (Accordion)
    - Mode: none / grid / optuna / optuna_multi
    - Trials + Timeout
    - [Auto-Optimize] button triggers /api/optimize/model instead
  
  [Submit: Start Training] — disabled if form invalid
```

Runs tab:
```
<RunsTable>  (Tremor table)
  Status badges: pending/running/success/failed (colored, not emoji)
  Columns: ID | Model Type | Tickers | Start | Duration | Metrics | Actions
  Filter: status, model_type
  Click row → expand details (metrics JSON, logs, reproduce command)
```

Models tab:
```
<ModelsTable>
  Columns: Name | Type | Tickers | Train Date | Val AUC | Actions
  Promoted model highlighted with gold border
  Actions: Promote / Delete / View details
  <ConfirmDialog> on destructive actions
```

### 7.4 Strategy `/strategy`

```
Sidebar (inside page, left 240px):
  <StrategyPicker>
    List of strategies from /api/strategies
    Selected highlighted
    
Main:
  PageHeader: Selected strategy name · version · description
  
  <Tabs>: Params · Results
  
  Params tab:
    <StrategyParamsForm>  (schema-driven from /api/strategies/{name})
      Dynamic field generation:
      - integer fields: <Slider> or <Input number>
      - boolean: <Switch>
      - enum: <Select>
    
    Ticker + date range + initial_cash
    [Generate Signals] [Backtest] [Optimize Parameters]
  
  Results tab:
    After backtest: 4× KPI <Card> (Sharpe, Return, Drawdown, #Trades)
    <EquityCurve> (Tremor AreaChart on returns)
    <SignalsVisualization> (Lightweight Charts with buy/sell markers)
    
  After Optimize:
    <Table> top 10 trials by target metric
    [Apply Best Params] → fills Params form
```

### 7.5 Trading `/trading`

```
Layout: 2-column (60/40)

Left column:
  <PortfolioCard>
    KPI grid: Cash · Equity · Day P&L (colored) · Total Return
    <DonutChart> positions by ticker
  
  <PositionsList>  (Tremor Table)
    Ticker | Qty | Avg Cost | Current | P&L | P&L %
    Real-time price updates from WS → Zustand → re-render rows
  
  <TradeHistory>  
    Last 20 trades, expandable
    Filter by side (buy/sell/all)

Right column:
  <OrderForm>  (react-hook-form)
    Ticker (TickerSearch mini-inline)
    Side (Buy/Sell tabs)
    Type (Market/Limit segmented control)
    Quantity + Limit price (conditional)
    [Place Order] → useMutation → <Toast> success
  
  <OrderList>  (open orders)
    ID, ticker, side, type, qty, price, status
    Cancel button per row → <ConfirmDialog>

PageHeader: Actions [Reset Portfolio] → <AlertDialog>

WebSocket: useLivePrices() hook connects /api/trading/ws/prices on mount,
  pushes updates to Zustand store, subscribed components re-render.
  Disconnect on unmount. Reconnect w/ 1s backoff on error.
```

### 7.6 Explain `/explain`

```
PageHeader: "Model Explainability"
  Ticker input (Cmd+K compatible)
  Model selector: "current promoted" or specific model_id

<Card title="SHAP Top Features">
  <ShapFeatureList>   (Tremor BarList horizontal bars)
    ~10 features, value + bar

<Card title="Similar Historical Cases">
  <SimilarCasesList>
    Accordion rows: score + summary; expand for full context
    From /search?q=...
  [Refresh Search] button
```

## 8. API Layer — TanStack Query Hooks (`api/queries.js`)

Example structure (full list in impl plan):

```js
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import * as api from "./client"

// ===== Market Data =====
export const useMarket = (ticker, opts = {}) => useQuery({
  queryKey: ["market", ticker],
  queryFn: () => api.getMarket(ticker),
  enabled: !!ticker,
  staleTime: 30_000,
  ...opts,
})

export const useScreenerTickers = () => useQuery({
  queryKey: ["screener"],
  queryFn: () => Promise.all(SCREENER_TICKERS.map(api.getMarket)),
  staleTime: 60_000,
})

// ===== Portfolio =====
export const usePortfolio = () => useQuery({
  queryKey: ["portfolio"],
  queryFn: api.getPortfolio,
  refetchInterval: 5000,
})

export const usePlaceOrder = () => {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.placeOrder,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["portfolio"] })
      qc.invalidateQueries({ queryKey: ["orders"] })
      qc.invalidateQueries({ queryKey: ["trades"] })
    },
  })
}

// ~20 hooks total
```

QueryClient config in `Providers.jsx`:
```js
new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 10_000,
      refetchOnWindowFocus: false,
    },
  },
})
```

## 9. Zustand Live Store (`stores/liveStore.js`)

Minimal, only for real-time streams:

```js
import { create } from "zustand"

export const useLiveStore = create((set) => ({
  prices: {},  // { AAPL: 178.23, TSLA: 245.10, ... }
  updatePrice: (ticker, price) =>
    set((state) => ({ prices: { ...state.prices, [ticker]: price } })),
  
  connectionStatus: "disconnected",  // disconnected | connecting | connected | error
  setConnectionStatus: (status) => set({ connectionStatus: status }),
}))
```

`useLivePrices()` hook writes to store; components read via selectors.

## 10. Form Validation (react-hook-form + zod)

Example `TrainForm`:
```js
const schema = z.object({
  tickers: z.string().regex(/^[A-Z,\s]+$/, "Only uppercase letters and commas"),
  model_type: z.enum(["logistic", "random_forest", ...]),
  horizon_days: z.number().int().min(1).max(60),
  // ...
  ensemble_config: z.object({...}).optional(),
}).refine(
  (data) => data.model_type !== "ensemble" || data.ensemble_config != null,
  { message: "ensemble_config required", path: ["ensemble_config"] },
)
```

## 11. Testing

### 11.1 Vitest unit + component tests

`src/__tests__/pages.smoke.test.jsx`:
```js
import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
// mock queries.js

describe("smoke: all pages render", () => {
  test.each([
    ["ScreenerPage", "/screener"],
    ["DashboardPage", "/dashboard?ticker=AAPL"],
    ["TrainingPage", "/training"],
    ["StrategyPage", "/strategy"],
    ["TradingPage", "/trading"],
    ["ExplainPage", "/explain"],
  ])("%s renders without error", (name, path) => {
    renderWithProviders(<App/>, { route: path })
    expect(screen.queryByText(/error/i)).toBeNull()
  })
})
```

Critical flow tests (per page):
- TrainForm validation rejects invalid input
- OrderForm mutation invalidates portfolio query
- CandlestickChart renders with markers
- ConfirmDialog cancel doesn't fire action

### 11.2 CI integration

`.github/workflows/ci.yml` add job:
```yaml
frontend-test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-node@v4
      with: { node-version: "20" }
    - run: cd quant-ai-ui && npm ci
    - run: cd quant-ai-ui && npm run lint
    - run: cd quant-ai-ui && npm test -- --run
    - run: cd quant-ai-ui && npm run build
```

### 11.3 Vercel Preview Deploy

Each PR auto-deploys preview URL via Vercel GitHub integration. Accept Harry confirms existing setup; if not, plan Task adds `vercel.json` github integration.

## 12. Migration Strategy

Not a rewrite-from-scratch — keep existing Vercel deploy working throughout:

1. Install all new deps in one commit
2. Build new `AppShell` + Sidebar alongside old App.jsx
3. Build new pages one-by-one, each replaces old page route
4. Delete old page file in same commit as new one shipped
5. Each Ralph task ends with `npm run build` green — no half-broken states
6. Old `client.js` kept but augmented with `queries.js`

## 13. Success Criteria

- `npm run build` clean, 0 errors, bundle < 500 KB gzipped
- All 6 pages render without console errors
- Vitest smoke tests pass for all pages
- Lightweight Charts K-line shows in `/dashboard` with signal markers
- shadcn Command palette (`Cmd+K`) global navigation works
- Mobile viewport (375px) — no horizontal overflow, sidebar becomes drawer
- No DisabledPanel / "Coming Soon" anywhere
- Tailwind dark theme applied globally
- CI green including new frontend-test job
- Vercel preview deploy accessible and functional

## 14. Out of Scope

- Backend API changes
- Authentication / user accounts
- Real-time trading WebSocket refactor (keep existing `/api/trading/ws/prices`)
- Mobile-first design (desktop-first)
- Light theme / theme toggle
- i18n
- Server-side rendering (stay client-side SPA)
- Advanced charting features (drawing tools, custom indicators) — use Lightweight Charts defaults
- Accessibility audit (WCAG AA) — deferred, rely on shadcn/Radix defaults
- Storybook / component catalog — deferred
- Error monitoring (Sentry) — deferred
- Analytics (PostHog / Plausible) — deferred

## 15. Interview Payoff

| Question | Answer anchor |
|----------|---------------|
| "How do you structure large React apps?" | features/ by domain, components/ for shared, clear separation |
| "How do you handle API state?" | TanStack Query for caching / invalidation / optimistic |
| "Forms and validation?" | react-hook-form + zod schemas, inline errors, no double submit |
| "Charts in React?" | Tremor for analytics, Lightweight Charts for price K-line (TradingView's lib) |
| "Component library choice?" | shadcn/ui (Radix primitives) + Tremor; own components in repo |
| "Real-time UI?" | WebSocket → Zustand → selective re-renders |
| "Testing?" | Vitest smoke tests per page, critical flow tests for forms |
| "Responsive?" | Desktop-first, Sheet drawer on mobile, 375px tested |
