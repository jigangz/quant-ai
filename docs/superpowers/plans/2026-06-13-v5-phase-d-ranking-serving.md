# V5 Phase D — Serving the xs_strong ranking (/predict/ranking + Top-N UI)

> Date: 2026-06-13 · Builds on Phase C (`2026-06-12-v5-phase-c-xs-training.md`).
> The xs_strong model is already trained + in the prod registry
> (`xgboost_A_AAL_AAPL_plus524_20260613_042420`, IC 0.0484). Phase D makes it SERVE.
>
> **STATUS: ✅ COMPLETE (2026-06-13).** Backend (RankingService + /predict/ranking +
> get_prices_batch) and frontend (RankingPage + route + nav + query) shipped.
> 7 new tests; full suite 507 passed. Live end-to-end verified: real model + real
> S&P 500 prices + real FastAPI server + browser render (Top-20 board, 527/527
> scored). Remaining = deploy to Render/Vercel; Phase E = backtest + README + video.

## Goal

Expose the cross-sectional model as a live "today's Top-N strong stocks" board:
score the whole universe as of the latest close, rank, return Top-N. Backend
endpoint + a frontend page. No retraining.

## Exit criterion

`GET /predict/ranking?top_n=20` returns a ranked list of strength scores for the
universe, computed by loading the xs model, building each name's latest features,
normalizing them as ONE cross-section (same `cross_section_normalize` as
training), scoring, and sorting. A `/ranking` page renders the board. Full suite
green.

## Key serving correctness point (the Phase C contract)

The model was trained on **per-date z-scored** factors with NO stored scaler
(`cross_section.py:8-13`). So serving MUST re-normalize today's names against
**each other** as one cross-section. Implementation: assemble one row per ticker
(latest features), assign a single constant `date`, call the identical
`cross_section_normalize(panel, factor_cols)` → one group → z-scored across names
→ `predict_proba[:,1]` = strength score. `extras.xs_strong.normalization` in the
registry row documents this; do not diverge.

## Decisions

| # | Decision | Why |
|---|----------|-----|
| D1 | New endpoint added to the existing `predict.router` (api/predict.py); no main.py change. | predict.router already `include_router`ed (main.py:162). |
| D2 | Default xs model = latest active `list_models(label_type='xs_strong')[0]`. No `is_promoted` change, no prod schema migration. | Global `is_promoted` is for the per-ticker model; a separate label-scoped promotion flag is deferred. One xs model exists today → latest == it. |
| D3 | `RankingService.rank()` is pure + injectable `price_loader` (default `get_prices_df`) for testability; batch fetch via new `get_prices_batch` for prod speed. | 527 sequential Supabase reads is too slow; one `WHERE ticker = ANY(...)` query is fine for a daily-cached board. |
| D4 | Universe = the model's trained `metadata.tickers` (the names it knows). | Serving must match training feature space; those are the names with data. |
| D5 | Score = `predict_proba[:,1]` (P[next-5d top-30%]); response labels it a "strength score, NOT a price target". | Honest-by-design; the score is a cross-sectional rank probability. |
| D6 | Daily result cache keyed by (model_id, as_of_date) at the endpoint layer; `rank()` itself uncached. | Board changes once per close; first request/day pays the compute. |
| D7 | NaN factors for a name → handled by the model pipeline's `SimpleImputer(median)`; drop only names with too little history to build features at all. | The trained pipeline already imputes; matches training behavior. |

## TDD steps (tests in `tests/test_v5_phase_d_ranking.py`)

### D1 · batch price fetch (`app/db/prices_repo.py`)
- `get_prices_batch(tickers: list[str], since_days: int = 520) -> dict[str, pd.DataFrame]` — one `SELECT ... WHERE ticker = ANY(:tks) AND date >= :since ORDER BY ticker, date`; group into per-ticker ascending frames. `OperationalError` → `{}`.
- **Test**: monkeypatch the engine/execute to return synthetic rows; assert grouping + ascending sort + empty-on-error.

### D2 · RankingService (`app/services/ranking_service.py`, new)
- `default_xs_model_id() -> str | None` — `list_models(label_type='xs_strong', limit=1)`, newest `.id`.
- `RankingService.rank(model_id=None, top_n=20, universe=None, price_loader=get_prices_df) -> dict`:
  1. resolve model (id or default) via `predict_service.get_model`; 404-shape dict if none.
  2. universe = `universe or model.metadata.tickers`.
  3. per ticker: `price_loader(ticker)` → `add_technical_features` → latest row's `feature_names` factors; skip names with <60 rows.
  4. assemble panel (ticker + factor cols), add constant `date`, `cross_section_normalize(panel, factor_cols)`, reindex to model `feature_names`.
  5. `score = predict_proba[:,1]`; sort desc; build `rankings` (rank, ticker, score, percentile) for Top-N.
  6. return `{success, model_id, as_of, universe_size, scored, top_n, rankings, score_semantics}`.
- **Test** (synthetic `price_loader` over ~8 fake tickers + a tiny real xs model trained in-test via the Phase C path): rankings sorted desc, length == top_n, scores in [0,1], percentile monotonic, default-model selection works, empty-universe / no-model → graceful failure dict.

### D3 · endpoint (`app/api/predict.py`)
- `GET /predict/ranking?top_n=20&model_id=` → `RankingService().rank(...)`, with a module-level daily cache keyed by `(model_id, as_of_date)`.
- **Test**: FastAPI `TestClient` (or direct call) returns 200 + the documented shape; bad/no model → success=false, not a 500.

### D4 · Frontend Top-N page (`quant-ai-ui`)
- `src/api/client.js`: `getRanking(topN)` → `/predict/ranking?top_n=`.
- `src/api/queries.js`: `useRanking(topN)` react-query hook.
- `src/pages/RankingPage.jsx`: table of rank / ticker / strength score (bar) / percentile, reusing `ScreenerTable`/`Sparkline` styling; a clear "strength score, not a price target" disclaimer.
- `src/app/router.jsx` + `src/app/Sidebar.jsx`: `/ranking` route + nav item.
- **Verify**: build + a preview smoke (the board renders, disclaimer visible).

## Out of scope (Phase E)
Top-N portfolio backtest, explicit label-scoped promotion UI, README ranking
section + risk disclaimers + screenshots, demo video.
