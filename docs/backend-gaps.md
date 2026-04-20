# Backend Gaps — Surfaced During Frontend Productization

> Running list of backend capabilities the frontend needs but the current API doesn't expose cleanly. Updated as each Sub 1-7 surfaces new gaps.
>
> Each gap has a **MVP fallback** (so frontend work isn't blocked) and a **proposed endpoint** for a future backend sub-project.

## Source: Sub 1 Dashboard (Design Spec V2, 2026-04-19)

### G1 · Historical prediction accuracy (per model, per ticker, optionally per month)

- **Needed by**:
  - §11 Model Comparison — each model card shows "✓ 预测准确率 64%"
  - §13 Seasonality — 12-month accuracy heatmap
- **Current state**: Not exposed. Prediction outputs aren't persisted as a per-row historical log that can be joined against actual price outcomes.
- **Proposed endpoint**:
  ```
  GET /models/{model_id}/accuracy?ticker=AAPL[&groupby=month]
  → {
      overall_accuracy: 0.64,
      n_predictions: 127,
      by_month: [{ month: 1, accuracy: 0.68, n: 12 }, ...]  // when groupby=month
    }
  ```
- **Backend work**:
  - Add `prediction_log` table (id, run_id, model_id, ticker, predicted_at, horizon_days, predicted_direction, actual_direction_resolved_at, correct bool)
  - Populate on every `/predict` + `/agents/technical` call
  - Add a daily job to resolve `actual_direction` once horizon elapses
  - New endpoint computes groupby
- **MVP fallback (this sub)**:
  - Frontend computes approximation from `/runs` metrics if available
  - Otherwise renders "数据积累中" placeholder tile
- **Priority**: Medium — bump when users start asking "is this model actually good?"

### G2 · Sector / industry peer lookup

- **Needed by**: §10 Related Stocks (6 peer cards)
- **Current state**: No endpoint. yfinance has `company.info.sector` but we don't expose it.
- **Proposed endpoint**:
  ```
  GET /tickers/{ticker}/related?by=sector&limit=6
  → [{ ticker: "MSFT", name: "Microsoft Corp.", sector: "Technology" }, ...]
  ```
- **Backend work**:
  - Cache sector mapping (yfinance batch pull, refresh weekly)
  - Simple SQL: `SELECT ticker, name FROM tickers WHERE sector = (SELECT sector FROM tickers WHERE ticker = ?) LIMIT 6`
- **MVP fallback (this sub)**:
  - Hardcode sector-peer list in frontend (e.g. `AAPL → [MSFT, GOOGL, AMZN, NVDA, META, TSLA]`)
- **Priority**: Low — static list works for demo; dynamic needed when adding lots of tickers

### G3 · Filter `/models` by ticker

- **Needed by**: §11 Model Comparison (list only models trained on this ticker)
- **Current state**: `GET /models?status=active` works but no `?ticker=` filter
- **Proposed**: extend existing endpoint to support `?ticker=AAPL` — returns models whose `tickers[]` contains AAPL
- **Backend work**: ~5 lines — add WHERE clause on the JSON/array column
- **MVP fallback (this sub)**: fetch all active models, filter client-side
- **Priority**: Low — quick win, do it as part of the Ralph loop

### G4 · Watchlist persistence (user-scoped)

- **Needed by**: §16 Right Rail Watchlist
- **Current state**: No endpoint, no auth system
- **Proposed endpoint**:
  ```
  GET /api/watchlist          → user's tickers
  POST /api/watchlist         { ticker } → add
  DELETE /api/watchlist/{ticker} → remove
  ```
- **Backend work**:
  - New `watchlists` table in Supabase
  - Needs auth to scope by user (see G5)
- **MVP fallback (this sub)**: `localStorage` only (no cross-device sync)
- **Priority**: Deferred — needs auth first

### G5 · Auth / user accounts (prerequisite for G4, future Portfolio persistence, etc.)

- **Needed by**: G4 (watchlist) · future per-user portfolio · per-user model ownership
- **Current state**: Zero auth — all endpoints open
- **Proposed**: Supabase Auth (email + magic link) + JWT middleware in FastAPI
- **Priority**: Deferred to future phase — MVP is single-user local app. No blocker for Sub 1-7 if we stick to `localStorage` for user state.

### G6 · Model source metadata in `/agents/technical` response

- **Needed by**: §2 Symbol Header (model source tag) · §4 Description (auto-generated sentence)
- **Current state**: Likely `/agents/technical` returns `model_id` but NOT `run_id` / `git_sha` / `trained_on` / `auc`. Need to verify `app/api/agents.py`.
- **Proposed**: extend response shape to include full `model_meta: { id, run_id, git_sha, trained_on, auc }`
- **Backend work**: ~10 lines — join with model_registry + training_runs
- **MVP fallback (this sub)**: Frontend does a second call `GET /models/{id}` to fetch metadata (extra roundtrip)
- **Priority**: Medium — saves a roundtrip, improves UX

---

## Expected future sub gaps (preview)

- **Sub 2 Screener** — batch `/agents/summary` for dozens of tickers may be slow → consider a batched `/agents/bulk` endpoint with cache
- **Sub 3 Portfolio** — needs multi-ticker backtest with weighting (currently per-ticker); may need a portfolio-level optimization endpoint
- **Sub 4 Training** — needs "train progress" SSE/WebSocket stream for live Optuna trial updates (current polling works but UX is poor)
- **Sub 5 Strategy** — current `/api/strategies/{name}/signals` returns full series; may want a "latest-only" shortcut endpoint for Dashboard overlay

These will be added to this doc as each sub starts.

---

## Change Log
- 2026-04-19: Initial document, 6 gaps from Sub 1 Dashboard V2
