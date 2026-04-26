# V4 Pivot · Phase 5 · Ablation Demo

**Run date**: 2026-04-26T07:28:58+00:00
**Targets**: direction · volatility · meta_label
**Feature sets**: ta_basic vs ta_basic + sentiment
**Model**: xgboost (default params, no Optuna — fair comparison)
**Horizon**: 5 days

## Per-ticker results

### AAPL

| Target | ta_basic | ta_basic + sentiment | Δ |
|---|---|---|---|
| direction | 0.500 | 0.500 | +0.000 |
| volatility | 0.173 | 0.173 | +0.000 |
| meta_label | 0.444 | 0.444 | +0.000 |

**Summary**: Sentiment lifts direction's primary metric most (deltas: direction=+0.000, volatility=-0.000, meta_label=+0.000).
**Elapsed**: 7.7s

### MSFT

| Target | ta_basic | ta_basic + sentiment | Δ |
|---|---|---|---|
| direction | 0.500 | 0.500 | +0.000 |
| volatility | 0.290 | 0.290 | +0.000 |
| meta_label | 0.594 | 0.594 | +0.000 |

**Summary**: Sentiment lifts direction's primary metric most (deltas: direction=+0.000, volatility=-0.000, meta_label=+0.000).
**Elapsed**: 4.6s

### GOOGL

| Target | ta_basic | ta_basic + sentiment | Δ |
|---|---|---|---|
| direction | 0.500 | 0.500 | +0.000 |
| volatility | 0.413 | 0.413 | +0.000 |
| meta_label | 0.581 | 0.581 | +0.000 |

**Summary**: Sentiment lifts direction's primary metric most (deltas: direction=+0.000, volatility=-0.000, meta_label=+0.000).
**Elapsed**: 4.0s

## Honest framing

Default-params XGBoost across 3 tickers. Numbers are not optimized (no Optuna) — that's intentional: ablation shows feature contribution in isolation. Optuna in cells would obscure whether sentiment helps or hyperparameters help. If sentiment Δ is positive on a target, that target benefits from sentiment beyond what defaults can squeeze out of ta_basic alone.
