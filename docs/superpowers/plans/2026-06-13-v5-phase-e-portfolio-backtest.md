# V5 Phase E — Top-N ranking portfolio backtest

> Date: 2026-06-13 · Builds on Phase C/D (xs_strong model is trained, in the registry, and served).
> Goal: turn the ranking signal into an honest **portfolio backtest** — the strongest quant-interview
> artifact ("I backtested the selection strategy net of transaction costs vs a fair benchmark").
>
> **STATUS: ✅ core done (E1–E3). 6 new tests; full suite 514 passed. Adversarially audited for
> lookahead — verdict: mechanics CLEAN (no feature lookahead, no selection-uses-future, non-overlapping
> H-day returns [horizons matched], per-date norm leak-free, costs correct). The one audit finding (OOS
> cutoff included the val split) is FIXED: metadata now persists `test_start_date` and the backtest uses
> it (+ `--since` override); only leaks for a tuned model, and the published model used `search_mode=none`.**
>
> ## Honest results (real model, 10bps/side)
>
> | Window | benchmark netSharpe | Long-only **decile** netSharpe (gross) | net CAGR% |
> |--------|--------------------|----------------------------------------|-----------|
> | val+test (~33 rebal, 2025-10→2026-06) | 1.50 | **2.17** (2.36) | 59 |
> | **test-only** (~17 rebal, 2026-02→2026-06) | 0.48 | **1.75** (1.89) | 69 |
>
> The decile selection beats the equal-weight universe net of costs on BOTH windows — and on the purer
> test-only window the benchmark was a *weaker* regime (0.48) yet the strategy still delivered, so it's
> not just beta. The edge concentrates in the decile (top-30% ≈ near-benchmark).
>
> **Caveats stated upfront (the honest story):** ~17–33 rebalances is FAR too short for a reliable Sharpe
> (wide error bars); single window / regime; survivorship in the fixed `metadata.tickers` universe inflates
> *absolute* returns (not strategy-vs-benchmark). The honest claim = "selection added value net of costs
> out-of-sample in this window," NOT "a Sharpe-1.75 strategy." Resume bullet deliberately cites Rank IC /
> precision, not a Sharpe. Proper validation = multi-year / multi-regime (backlog).

## Why a new backtest (not the existing engine)

`app/backtest/engine.py` is a **per-ticker timing** backtest: each name goes long/flat on its own
`predict_proba > threshold`, then names are weighted into a portfolio. That measures *timing* skill per
name, not *cross-sectional selection*. The xs_strong story is "each period, pick the strong names" — a
different mechanic. We add a dedicated cross-sectional backtest and **reuse** `calculate_strategy_metrics`,
`cross_section_normalize`, `get_prices_batch`, `add_technical_features`, and the equity/drawdown pattern.

## The mechanic (honest by construction)

- **Rebalance every `H` days** (H = horizon = 5). Stepping by H → non-overlapping H-day returns → no
  overlapping-horizon leakage when chaining.
- At each rebalance date `t`: score `t`'s cross-section (per-date normalized factors, the SAME
  `cross_section_normalize`), rank by score.
  - **Long-only**: equal-weight the top `top_pct` names; period return = mean of their H-day forward returns.
  - **Long-short**: long top `top_pct` − short bottom `top_pct` (market-neutral, equal gross legs).
- **Transaction costs**: turnover = (names changed vs last rebalance) / N; cost = turnover × `cost_bps`
  (per side; long-short pays both legs). Subtract from the period return → **net**. Also keep **gross**.
- **Benchmark**: equal-weight of the FULL universe's H-day return each period (the fair cross-sectional
  benchmark — isolates selection). SPY optional/context.
- **Out-of-sample**: only dates `> model.metadata.train_end_date`.
- Chain `(1 + period_ret)` → equity curve; `calculate_strategy_metrics(returns, benchmark,
  periods_per_year = 252 / H)` → CAGR / Sharpe / MaxDD / alpha / beta.

## Decisions

| # | Decision | Choice |
|---|----------|--------|
| E1 | Reuse existing engine? | No — new `app/backtest/xs_portfolio.py`; reuse metrics/normalize/curves. |
| E2 | Rebalance | H = 5 days (matches label horizon; non-overlapping). |
| E3 | Selection size | `top_pct` configurable; **report top-decile (0.10, concentrated) AND top-30% (matches label)**. |
| E4 | Variants | **long-only Top-N (headline) + long-short Top−Bottom (market-neutral)**, **gross + net** — data picks the story. |
| E5 | Weighting | equal-weight (standard factor Top-N). |
| E6 | Benchmark | equal-weight full universe (fair); SPY optional. |
| E7 | OOS | dates `> train_end_date` only; report the window length + the small-N caveat honestly. |
| E8 | Costs | turnover-based, `cost_bps` per side (default 10). |

## TDD steps (tests in `tests/test_v5_phase_e_backtest.py`)

### E1 · period-return chaining + metrics core (`app/backtest/xs_portfolio.py`)
- `_rebalance_dates(dates, H) -> list` — every H-th unique date.
- `_period_returns(panel, model, factor_cols, top_pct, long_short) -> {dates, long, bench, turnover, ...}`
  — per rebalance date: normalize is assumed already applied; score; select; equal-weight mean fwd return;
  benchmark mean; turnover vs previous selection.
- `backtest_xs_portfolio(panel, model, *, top_pct=0.10, H=5, cost_bps=10, long_short=False) -> dict`
  — applies costs (gross + net), chains equity curves, calls `calculate_strategy_metrics` (periods/yr=252/H),
  returns `{success, n_rebalances, as_of_range, gross:{metrics,equity}, net:{...}, benchmark:{...}, avg_turnover}`.
- **Tests** (synthetic panel with a *planted* signal — names whose factor predicts their fwd return):
  a strong-signal model beats the equal-weight benchmark **gross**; costs reduce net < gross; long-short
  is ~market-neutral (low beta); zero-signal model ≈ benchmark; chaining math matches a hand-computed 2-period case.

### E2 · OOS panel builder (reuse Phase C/D pieces)
- `build_backtest_panel(model, prices, H) -> (panel, factor_cols)` — per ticker `add_technical_features` +
  `add_xs_forward_return` (H-day fwd), concat, `cross_section_normalize(factor_cols)`, **filter to dates >
  train_end_date**, drop NaN-factor rows (the Phase D serve contract).
- **Test**: panel has only OOS dates, factor cols are the model's `feature_names`, `future_return` present.

### E3 · runnable report (`scripts/xs_backtest.py`)
- Load model (DB blob), `get_prices_batch(universe)`, build OOS panel, run all 4 cells
  (long-only / long-short × decile / 30%), print a table: gross & net CAGR / Sharpe / MaxDD vs benchmark +
  avg turnover + OOS window. `--dry-run`-free (read-only; no writes).
- **Verify**: run on the real published model; capture the honest numbers.

### E4 (optional, lighter) · surface it
- `GET /backtest/ranking?top_pct=&long_short=` returning the curves+metrics (cache like `/predict/ranking`),
  and a small equity-curve card on the `/ranking` page. Only if E1–E3 land cleanly with time left.

## Out of scope (backlog)
Retrain on an earlier cutoff for a longer OOS window; sector-neutralization; vol-targeting; multiple horizons;
slippage model beyond flat bps. README "backtest" section + demo video = the remaining Phase E polish.
