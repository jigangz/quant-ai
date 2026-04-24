# P4 Signal Console Frontend Design Spec

**Date**: 2026-04-24
**Author**: Harry + Claude
**Status**: Approved (pending implementation)
**Phase**: V4 Pivot · Phase 4 · Signal Console Frontend — Gate 1 closer
**Predecessors**: P1 Volatility Backend (c689281), P2 Sub 4 Dialog + Vol Gauge (4fd6bf0), P3 Meta-Labeling Backend (v4-p3-complete)

## 1. Goal

Close Gate 1 ("V4 Full Story Demo Ready") by shipping the **Signal Console** — a frontend experience that makes the P3 meta-labeling backend visible and demonstrable. Users can see which strategy × ticker pairs have meta-models, inspect CV metrics, preview signal reliability in real-time, and gate Paper Trading orders with meta-score thresholds + half-Kelly sizing. Plus one P3 carryover fix (version bump) and one AAPL rescue attempt (Optuna tune to see if the failing benchmark result can be recovered).

After P4, the entire V4 Pivot story is demonstrable end-to-end in live prod: direction baseline → volatility forecasting → meta-labeling signal quality → Paper Trading integration. This is the **interview demo** that differentiates the Quant AI project from generic "I built a stock prediction app."

## 2. Design Principles

- **Reuse P3 backend** — no new ML code. Only a small aggregation endpoint (`/api/meta-label/coverage`) is added.
- **Independent route** — `/signal-console` is its own page (sibling of `/dashboard`, `/screener`, `/trading`). The feature is inherently multi-ticker × multi-strategy, which doesn't fit Dashboard's single-ticker focus.
- **Manual score preview** — no live auto-update on `/api/signal-score` (would spam API at ~300-500ms/call). Explicit "预览 score" button. Debounce as a future enhancement.
- **Opt-in Paper Trading integration** — meta-label filter is off by default via checkbox. Existing Paper Trading tests stay green, existing flows unchanged.
- **Badge design for interview impact** — "Meta ✓ N · AUC 0.XX" shows both coverage count AND best AUC in one tight badge. Tells the viewer "I trained meta-models on N tickers, best AUC 0.XX" at a glance.
- **Honest AAPL handling** — if Optuna rescue fails, the failure is documented, not hidden. Prado-style methodological honesty.

## 3. Scope

### In (Day 1 ship)

- **Backend** (~5% of effort):
  - `GET /api/meta-label/coverage?strategy=<name>` — aggregation for Strategy card badges
  - `app/main.py` version string bump `2.1.0` → `2.4.0` (P3 carryover)
- **Frontend new**:
  - `/signal-console` page + 3 components (TickerPicker, StrategyMatrix, SignalDetail)
  - `MetaLabelCoverageBadge` component (reused in Strategy page + Signal Console)
  - Paper Trading modal meta-label fields (checkbox + dropdown + threshold slider + preview button + score display)
  - Dashboard `VolatilityCard` 7-day signal-quality sparkline (conditional — only when meta-model exists for ticker)
- **Frontend modifications**:
  - `api/client.js` + `api/queries.js` additions
  - Strategy page `StrategyCard` integration (badge)
  - `App.jsx` route + `TopNavBar` nav link
- **AAPL rescue**:
  - `scripts/p4_aapl_optuna_rescue.py` script
  - Findings doc (success OR honest-failure path)
- Tests: ~31 new (5 backend + 26 frontend component/integration)
- P4-GATE: regression, progress log, tag `v4-p4-complete` + `v4-gate-1-complete`

### Out (deferred)

- Live auto-update score preview (manual button is MVP)
- `POST /api/meta-label/recommend` ranking endpoint (YAGNI for Gate 1 demo)
- Meta-model decay tracking dashboard (post-Gate 1)
- Custom strategy upload (never — outside V4 scope)
- Feature expansion (sentiment / regime / cross-sectional) — post P3/P4, tracked in P3 spec §12
- Dashboard full Signal Quality tab (sparkline is enough for Gate 1)

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ App Shell (existing) — TopNavBar · MigrationBanner · RAG    │
└────────────┬────────────────────┬───────────────────────────┘
             │                    │
             ▼                    ▼
┌──────────────────────┐  ┌────────────────────────────────┐
│ /signal-console      │  │ /trading (existing, enhanced)  │
│ (NEW PAGE)           │  │                                │
│ ┌──────────────────┐ │  │ Order modal adds:              │
│ │ TickerPicker     │ │  │  · meta_label_enabled checkbox │
│ │ (watchlist-scope)│ │  │  · meta_model_id dropdown      │
│ └──────────────────┘ │  │  · score_threshold slider      │
│ ┌──────────────────┐ │  │  · "预览 score" button         │
│ │ StrategyMatrix   │ │  │  · score + sizing display      │
│ │ ticker×strategy  │ │  └────────────────────────────────┘
│ │ (AUC·E[R]·CTA)   │ │
│ └──────────────────┘ │  ┌────────────────────────────────┐
│ ┌──────────────────┐ │  │ /strategy (existing, enhanced) │
│ │ SignalDetail     │ │  │ Each StrategyCard adds badge:  │
│ │ right panel      │ │  │  "Meta ✓ N · AUC 0.XX"         │
│ └──────────────────┘ │  └────────────────────────────────┘
└──────────────────────┘
                          ┌────────────────────────────────┐
                          │ Dashboard (existing)           │
                          │ VolatilityCard appends:        │
                          │  Conditional 7d sparkline      │
                          └────────────────────────────────┘
```

### Component responsibilities

| Module | Path | Responsibility | New / Mod |
|---|---|---|---|
| SignalConsolePage | `quant-ai-ui/src/pages/SignalConsolePage.jsx` | Page container + URL state (ticker selection, strategy filter) | 🆕 |
| TickerPicker | `src/features/signal-console/TickerPicker.jsx` | Reads `quant-ai:watchlist` from localStorage; multi-select up to 10 | 🆕 |
| StrategyMatrix | `src/features/signal-console/StrategyMatrix.jsx` | Grid: rows=tickers, cols=4 strategies. Each cell shows AUC·E[R]·coverage OR "Train meta" CTA. Click → select cell. | 🆕 |
| SignalDetail | `src/features/signal-console/SignalDetail.jsx` | Right panel: active cell's full metrics, latest signal status, recommended_action, sizing_hint | 🆕 |
| MetaLabelCoverageBadge | `src/components/MetaLabelCoverageBadge.jsx` | Reusable badge. Props: `strategyName`. Queries `/api/meta-label/coverage`. Shows `"Meta ✓ {count} · AUC {max_auc}"`. 404/empty → renders nothing. | 🆕 |
| `signalQueries.js` | `src/api/signalQueries.js` | TanStack Query hooks: `useMetaLabelModels(ticker)`, `useSignalScore(...)` (preview mutation), `useMetaCoverage(strategy)` | 🆕 |
| `client.js` | `src/api/client.js` | Adds `getMetaLabelModels`, `postSignalScore`, `getMetaCoverage` (and reuses P3 endpoints) | 🔧 |
| StrategyCard | `src/features/strategy/StrategyCard.jsx` | Embed `<MetaLabelCoverageBadge strategyName={name} />` in top-right | 🔧 |
| VolatilityCard | `src/features/dashboard/VolatilityCard.jsx` | Conditionally render `<MetaSparkline />` sub-component when ticker has meta-model | 🔧 |
| PaperTradingPage | `src/pages/PaperTradingPage.jsx` (or its order modal) | Meta-label section (checkbox-gated UI) + state + `/api/signal-score` preview | 🔧 |
| App.jsx + TopNavBar | `src/App.jsx` · `src/components/layout/TopNavBar.jsx` | Add `/signal-console` route + nav link (after 研究 group) | 🔧 |
| Coverage endpoint | `app/api/signal.py` | `GET /api/meta-label/coverage?strategy=X` | 🔧 |
| Coverage service | `app/services/signal_scoring_service.py` | Add `compute_coverage(strategy_name)` function that walks ModelRegistry | 🔧 |
| Version bump | `app/main.py` | `2.1.0` → `2.4.0` | 🔧 |

## 5. Data Flow

### 5.1 Signal Console page load

```
User navigates to /signal-console
  │
  ├─ TickerPicker mounts → reads localStorage.quant-ai:watchlist
  │     default view: first 5 tickers selected, up to 10 max
  │
  ├─ For each selected ticker:
  │    useMetaLabelModels(ticker) → GET /models?label_type=meta_label&ticker=X
  │    → [{model_id, extras.meta_label.primary.strategy_name,
  │        extras.meta_label.cv.metrics.auc_mean, event_count, ...}]
  │
  ├─ StrategyMatrix renders 4-column grid:
  │    cell[ticker][strategy] =
  │      if model_exists: { model_id, auc, E[R]_estimate, event_count }
  │      else: <TrainMetaCTA ticker strategy />
  │    auc < 0.5 → amber cell background with "⚠" marker
  │
  ├─ User clicks cell → setSelectedCell({ticker, strategy, model_id})
  │
  └─ SignalDetail right panel:
       useSignalScore({ticker, meta_model_id, strategy_name}) in auto-trigger mode
       → POST /api/signal-score (mode B)
       → show: triggered?, signal direction, reliability_score, expected_R,
                recommended_action, sizing_hint, primary_source, cv_auc
```

### 5.2 Strategy page badge

```
<StrategyCard name="rsi_strategy">
  <MetaLabelCoverageBadge strategyName="rsi_strategy" />
    └─ useMetaCoverage("rsi_strategy")
       → GET /api/meta-label/coverage?strategy=rsi_strategy
       → { count: 3, max_auc: 0.619, avg_auc: 0.549, tickers: [...], models: [...] }
    └─ renders: "Meta ✓ 3 · AUC 0.62"
    └─ onClick: navigate to /signal-console?strategy=rsi_strategy
  (404 / count=0 → renders null, no error toast)
```

### 5.3 Paper Trading order flow

```
User opens order modal for "Buy AAPL 10 shares"
  │
  ├─ Modal shows new collapsed section:
  │   [ ] Use meta-label filter  (checkbox)
  │
  ├─ User checks → section expands:
  │    Meta model: <Dropdown filtered by ticker=AAPL, label_type=meta_label>
  │    Threshold: [0.45 ────●──── 0.85] (default 0.55)
  │    [预览 score] button
  │
  ├─ Click "预览 score":
  │    → POST /api/signal-score {ticker: AAPL, meta_model_id, signal: +1}
  │    → inline display:
  │         Score: 0.71 · Expected R: +0.54 · Action: trade
  │         Sizing hint: 10 × 0.72 = 7 shares (half-Kelly)
  │
  ├─ User clicks "下单" → existing POST /orders + meta_model_id + score_threshold
  │    Backend (P3) applies gate in place_order
  │    → OrderPlaced (size adjusted) OR OrderRejected (reason: meta_score_below_threshold)
  │
  └─ Modal shows result inline; if rejected, stays open for user to adjust threshold
```

### 5.4 Dashboard VolatilityCard sparkline

```
<VolatilityCard ticker={ticker} volModelId={...}>
  {/* existing: year-annualized vol gauge */}
  <MetaSparkline ticker={ticker}>
    └─ useMetaLabelModels(ticker) → first meta-model
    └─ if model exists:
         → for each of last 7 daily closes:
             POST /api/signal-score {ticker, meta_model_id, signal: +1,
                                      timestamp: <that day>}
         → renders inline sparkline of last 7 reliability_scores
         (uses mini-chart lib already in quant-ai-ui)
    └─ if no model: renders nothing
  </MetaSparkline>
</VolatilityCard>
```

## 6. API Contracts

### 6.1 `GET /api/meta-label/coverage?strategy=<name>` (new)

**Request:** query param `strategy` ∈ {ma_cross, rsi_strategy, bollinger_breakout, sentiment_driven}

**Response 200 (coverage exists):**
```json
{
  "strategy_name": "rsi_strategy",
  "count": 3,
  "max_auc": 0.619,
  "avg_auc": 0.549,
  "tickers": ["MSFT", "GOOGL", "AAPL"],
  "models": [
    { "model_id": "meta_msft_a3f2", "ticker": "MSFT",
      "auc_mean": 0.619, "event_count": 483 },
    { "model_id": "meta_googl_b2c1", "ticker": "GOOGL",
      "auc_mean": 0.607, "event_count": 486 },
    { "model_id": "meta_aapl_c9d4", "ticker": "AAPL",
      "auc_mean": 0.420, "event_count": 492 }
  ]
}
```

**Response 200 (no coverage yet):**
```json
{
  "strategy_name": "bollinger_breakout",
  "count": 0,
  "max_auc": null,
  "avg_auc": null,
  "tickers": [],
  "models": []
}
```

**Response 404:**
- strategy not in whitelist (prevents typos causing silent empty-list)

**Implementation note:** walks `ModelRegistry.list_models(label_type="meta_label")`, filters by `extras.meta_label.primary.strategy_name == strategy` and `extras.meta_label.primary.source == "strategy"` (skip ML-primary models for this endpoint's badges). Aggregates max/avg AUC and ticker list.

### 6.2 Existing endpoints reused

- `GET /models?label_type=meta_label&ticker=X` (P2 G3 filter) — front-end StrategyMatrix + PaperTrading dropdown
- `POST /api/signal-score` (P3) — SignalDetail + PaperTrading preview + Dashboard sparkline

## 7. UI / UX Details

### 7.1 Badge design: `MetaLabelCoverageBadge`

Placement: top-right of StrategyCard, next to existing strategy name.

```
┌─────────────────────────────────────────────────┐
│ rsi_strategy                    [Meta ✓ 3 · 0.62]│
│ RSI oversold/overbought trigger strategy        │
│ ...                                             │
└─────────────────────────────────────────────────┘
```

- Background color logic:
  - `avg_auc ≥ 0.60` → green tint
  - `0.50 ≤ avg_auc < 0.60` → neutral
  - `avg_auc < 0.50` → amber "⚠" tint
  - No coverage → hidden entirely (no gray placeholder)
- Click → navigate to `/signal-console?strategy=<name>` with pre-selected strategy column

### 7.2 StrategyMatrix layout

```
            ma_cross      rsi_strategy   bollinger_br   sentiment
AAPL        [AUC 0.51]    [⚠ 0.42]       [— Train]      [— Train]
MSFT        [AUC 0.58]    [AUC 0.62]     [— Train]      [— Train]
GOOGL       [— Train]     [AUC 0.61]     [— Train]      [— Train]
NVDA        [— Train]     [— Train]      [— Train]      [— Train]
```

Cells are clickable when model exists. "Train" CTA → opens confirmation dialog → `POST /api/meta-label/train` with default params → polls for result → refreshes matrix.

### 7.3 SignalDetail right panel (when cell selected)

```
MSFT × rsi_strategy
─────────────────────
Model: meta_msft_a3f2
Created: 2026-04-24 06:22
Event count: 483
Class balance: 220 ✓ / 263 ✗

CV metrics (Purged K-Fold, n=5, embargo 1%)
  AUC mean ± std: 0.619 ± 0.061
  Precision @ 50%: 0.549
  E[R | trade]: +0.020

Latest signal (auto-triggered)
  ✓ Triggered at 2026-04-24
  Signal: +1 (long)
  Reliability score: 0.71
  Expected R: +0.54
  Recommended action: TRADE
  Sizing hint: half-Kelly 0.18 (capped at 0.25)

Primary source: rsi_strategy (rule)
Barrier: TP=2σ · SL=1σ · timeout=5d · vol=realized_sigma
```

### 7.4 Paper Trading modal additions

New section below existing order form (collapsed by default):

```
□ Use meta-label filter
  └── (on check, expands to:)
      Meta model: [Dropdown: rsi_strategy · meta_aapl_c9d4 · AUC 0.42 ⚠]
                                [other models for this ticker]
      Threshold:  [0.45 ──●──── 0.85]  0.55
      [预览 score]

      (after click:)
      Score: 0.71 · E[R]: +0.54 · Action: trade
      Sizing: 10 shares × 0.72 = 7 shares (half-Kelly)
```

## 8. Error Handling

| Scenario | Front-end behavior |
|---|---|
| `useMetaLabelModels(ticker)` returns empty | StrategyMatrix shows "Train" CTA for that row's cells |
| `useMetaCoverage(strategy)` → 404 | Badge renders nothing (no toast, no console error) |
| `POST /api/signal-score` returns `triggered: false` | SignalDetail shows "Strategy silent at latest close" + last N triggers history (if derivable from events) |
| `POST /api/signal-score` HTTP 500 | Toast error + SignalDetail keeps stale data with "⚠ stale" marker |
| Order rejected by meta-gate | Modal stays open, highlights score line, "Adjust threshold or skip" hint |
| AAPL meta-model AUC < 0.5 | Cell shows amber "⚠" tint + badge color logic triggers warning tint; SignalDetail shows "Model underperforms random — consider retraining with Optuna or skip" |
| Coverage endpoint times out | Badge uses last cached value (TanStack staleTime 30s) or renders null |

## 9. Testing Strategy

**Total: ~31 new tests** (5 backend contract + 26 frontend)

**Backend (Vitest-compatible Python pytest)**:
- `tests/contract/test_meta_coverage.py` (5 tests):
  - 200 with 3 models for rsi_strategy (MSFT/GOOGL/AAPL)
  - 200 with 0 models for bollinger_breakout
  - 404 for unknown strategy name
  - Aggregation math correct (max + avg AUC)
  - Skips ModelRecord with missing or malformed extras.meta_label

**Frontend (Vitest + @testing-library/react)**:
- `__tests__/components/MetaLabelCoverageBadge.test.jsx` (3): renders count+auc, hides on 404, amber tint on auc<0.5
- `__tests__/api/signalQueries.test.js` (3): useMetaLabelModels / useMetaCoverage / useSignalScore contract tests with MSW mock
- `__tests__/features/signal-console/TickerPicker.test.jsx` (3): localStorage load, multi-select, 10-ticker cap
- `__tests__/features/signal-console/StrategyMatrix.test.jsx` (4): renders 4 columns, CTA on empty cell, cell click selection, amber on low-AUC
- `__tests__/features/signal-console/SignalDetail.test.jsx` (3): renders all metrics, triggered=false state, sizing_hint display
- `__tests__/pages/SignalConsolePage.test.jsx` (2): mount integration, URL query param pre-selects strategy
- `__tests__/features/strategy/StrategyCard.test.jsx` (2): badge renders for each of 4 strategies, hidden when coverage=0
- `__tests__/pages/PaperTradingPage.test.jsx` (4): checkbox toggles section, dropdown filters by ticker, threshold slider emits, preview button + score display
- `__tests__/features/dashboard/VolatilityCard.test.jsx` (2): sparkline renders when meta-model exists, hidden when absent

**Regression**:
- `npm run test -- --run` — existing 35 tests remain green
- `npm run build` — bundle size < 250KB gzipped acceptable
- `npm run lint` — clean
- Backend: `pytest tests/contract/test_meta_label_train.py tests/contract/test_signal_score.py tests/test_paper_trading_meta.py tests/contract/test_backtest_flow.py` — all green

**Live smoke (P4-GATE)**:
- Prod: `curl /api/meta-label/coverage?strategy=rsi_strategy` returns 200 with 3 tickers
- Prod: `curl /health` shows version `2.4.0`
- Prod: `curl /openapi.json | grep signal-console` — expected 404 (frontend-only route)
- Vercel: `https://quant-ai-ui.vercel.app/signal-console` loads without error

## 10. AAPL Optuna Rescue (P4-11)

**Script:** `scripts/p4_aapl_optuna_rescue.py`

Runs `POST /api/meta-label/train` on AAPL × rsi_strategy with `search_mode="optuna"` and `n_trials=30`. Captures:
- Trial count + best AUC after trials
- Final hyperparams
- Comparison to default-params baseline (AUC 0.420 from P3 benchmark)

**Decision tree:**
- If best CV AUC ≥ 0.5 → **success**: update `docs/benchmarks/p3_meta_label_benchmark.md` with addendum, write `docs/benchmarks/p4_aapl_optuna.md` with params + AUC delta
- If best CV AUC < 0.5 → **honest failure**: write `D:/obsidian vault/Quant/03_Rejected/aapl_rsi_meta.md` with:
  - Hypothesis: AAPL × rsi_strategy × default feature set is meta-labelable
  - Result: Optuna n=30 could not lift AUC above 0.5
  - Interpretation: either RSI's signal on AAPL is noise at the 5-day horizon, or feature set needs expansion (sentiment, regime) — noted in P3 backlog
  - Interview talking point: "Prado says try methodology first, then let data speak. I did. AAPL × rsi_strategy isn't meta-labelable with default features. That's a real finding."

Either outcome is a legitimate deliverable — no fudging.

## 11. Success Criteria

- [ ] All 31 new tests + regression guard green (frontend + backend)
- [ ] `/signal-console` page loads at `https://quant-ai-ui.vercel.app/signal-console` without console errors
- [ ] Strategy page shows badge on all 4 strategy cards (content varies by coverage)
- [ ] Paper Trading modal meta-filter works end-to-end: opt-in → preview → gated order
- [ ] Dashboard VolatilityCard sparkline appears for MSFT (confirmed has meta-model)
- [ ] AAPL rescue executed and documented (success or failure path)
- [ ] `app/main.py` version `2.4.0` on prod /health
- [ ] Tags `v4-p4-complete` AND `v4-gate-1-complete` pushed
- [ ] `master-roadmap.md` Gate 1 marked ✅ with completion date
- [ ] `progress.md` Day 14 entry with deliverables + methodology note

## 12. Non-Goals

- ❌ No new primary strategies (4 rules + ML direction already supported from P3)
- ❌ No live score auto-update (manual button is shipping UX)
- ❌ No meta-model decay monitoring (post-Gate 1)
- ❌ No user-configurable action thresholds (0.65/0.45/0.45 hardcoded per P3; config layer is future)
- ❌ No custom strategy upload
- ❌ No mobile-first layout (desktop web matches existing Dashboard style)
- ❌ No i18n — all new text is Chinese/English mix matching existing style

## 13. Future Backlog (post-Gate 1)

1. Live auto-update score preview (debounced)
2. Meta-model decay tracking dashboard (rolling AUC over time)
3. `POST /api/meta-label/recommend` — rank-order strategies × tickers by expected_R
4. Signal Console time-series view: reliability score across last N days per selected cell
5. Paper Trading per-order threshold override UI + post-order retrospective "was this score accurate"
6. Advanced primary configurations in Train CTA (non-default barriers, Optuna, ensemble meta)
7. Multi-primary ensemble voting (if P3 backlog item 3 lands)

## 14. Dependencies

**Depends on (must exist):**
- P3 Meta-Labeling backend (`/api/meta-label/train`, `/api/signal-score`, Paper Trading gate) — merged
- P2 G3 `/models?label_type=meta_label&ticker=X` filter — merged
- P2 G6 model metadata in `/agents/technical` — used for Dashboard `SymbolHeader` context (read-only)
- Existing `api/client.js` + `api/queries.js` patterns
- Existing TanStack Query + React Router setup

**Unblocks:**
- Post-Gate 1 investment decisions (P5 G1 prediction_log, P6 散户产品化, P7 GKE)
- Interview-ready live demo URL
- Resume/Profile Quant AI one-liner update (already in place from P3)

## 15. Change Log

- 2026-04-24: Initial design approved after 3-section brainstorm walkthrough (Harry gave go-ahead after Q1-5 answered autonomously). Scope locked to full P4 frontend + coverage endpoint + AAPL rescue + P3 carryover (version bump). Gate 1 targets all of P1-P4 complete by end of day.

## 16. References

- `docs/superpowers/specs/2026-04-24-p3-meta-labeling-design.md` — P3 backend spec (direct consumer)
- `D:/obsidian vault/01-projects/quant-ai/p4-prep.md` — overnight prep note
- `D:/obsidian vault/01-projects/quant-ai/sub-4-modeling-dialog-wip.md` — UI pattern (TradingView-style modal reference for Paper Trading modal)
- `D:/obsidian vault/01-projects/quant-ai/master-roadmap.md` — Gate 1 definition, P4 slot
- `D:/obsidian vault/Quant/_Knowledge_Base.md` — methodology reference for AAPL rescue framing
- Prior specs: P2 (`2026-04-19-dashboard-productization-design.md`) for design language continuity
