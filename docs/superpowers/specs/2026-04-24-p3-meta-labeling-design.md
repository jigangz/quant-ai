# P3 Meta-Labeling Backend Design Spec

**Date**: 2026-04-24
**Author**: Harry + Claude
**Status**: Approved (pending implementation)
**Phase**: V4 Pivot · Phase 3 · Meta-Labeling Backend
**Predecessors**: P1 Volatility Backend (merged `c689281`), P2 Sub 4 Dialog + Vol Gauge (merged `4fd6bf0`)

## 1. Goal

Implement López de Prado Ch.3 meta-labeling on top of the existing Quant AI platform. Take a primary directional signal (from one of the 4 rule strategies OR from the P1 ML direction model), use the **triple-barrier method** with **dynamic volatility-scaled barriers** (powered by the P1 volatility model) to label each signal as "was the primary right?", then train a secondary XGBoost classifier to predict **signal reliability**. Expose the trained meta-model via two API endpoints and wire it into the Paper Trading engine for pre-order gating and score-based sizing.

This is the "most hardcore" phase of the V4 Pivot per `master-roadmap.md` and directly operationalizes the methodology in `Quant/_Knowledge_Base.md`.

## 2. Design Principles

- **Composable registry pattern** — follow P1's `app/ml/labels/registry.py` style. Triple-barrier and Purged K-Fold are pure functions; services only orchestrate.
- **Reuse existing infrastructure** — ModelRegistry (already supports `label_type="meta_label"` from P1 schema), BaseModel subclasses, DatasetBuilder for OHLC+features, backtest engine where it helps.
- **Event-indexed, not time-indexed** — meta-labels live at signal timestamps, not daily rows. CV must respect event spans or results leak.
- **Prado-rigorous CV** — **Purged K-Fold with embargo** is mandatory. Simple time-series split is insufficient for the methodology's published credibility.
- **Backward compatible** — Paper Trading meta-integration is opt-in (`meta_label_enabled=False` default). Existing training/prediction flows untouched.
- **Dual primary source** — API accepts either `strategy_name` (rules) or `primary_model_id` (P1 direction ML). Connects all prior phases.
- **Dynamic barriers via P1** — TP/SL = k × predicted_vol using the P1 volatility model; falls back to realized σ if no vol model promoted for the ticker.

## 3. Scope

### In (Day 1 ship)

- `app/ml/labels/meta_label.py` — triple-barrier generator + meta-label target computation (pure functions)
- `app/ml/split/purged_kfold.py` — Purged K-Fold splitter with embargo (pure)
- `app/services/primary_signal_service.py` — dispatcher: `strategy_name | primary_model_id` → `Series[-1/0/+1]` + signal_strength
- `app/services/meta_label_service.py` — end-to-end training composer
- `app/services/signal_scoring_service.py` — inference composer (modes A/B/C)
- `app/ml/metrics/meta.py` — precision-at-K, expected_R-when-trade, hit-rate-when-trade
- `app/api/routers/signal.py` — `POST /api/meta-label/train` + `POST /api/signal-score`
- Registry wiring — `registry.py`'s `meta_label` entry points to real generator; `LABEL_TYPE_TO_TASK["meta_label"] = "classification"`
- Training Service branch — `meta_label` label_type routes to MetaLabelTrainingService instead of default pipeline
- Paper Trading integration — `app/trading/engine.py::place_order()` accepts `meta_model_id` + `score_threshold`; gates orders + half-Kelly sizing
- Ensemble support — meta-model of type `ensemble` reuses existing EnsembleModel (voting/stacking)
- Tests — 25-35 tests across 8 files (breakdown in §10)
- Benchmark — `scripts/p3_meta_label_benchmark.py` + `docs/benchmarks/p3_meta_label_benchmark.md` (AAPL+MSFT+GOOGL × rsi_strategy)

### Out (deferred, see §12 Future Backlog)

- Frontend UI (Signal Console, Paper Trading toggles) → P4 tomorrow
- Feature expansion (sentiment, regime state, cross-sectional, fundamentals)
- Meta-model monitoring / decay tracking
- Custom strategy upload, ensemble primary, neural meta-models

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   API Layer (FastAPI)                        │
│  POST /api/meta-label/train   POST /api/signal-score         │
└────────────┬──────────────────────────┬──────────────────────┘
             │                          │
             ▼                          ▼
┌──────────────────────┐    ┌──────────────────────────┐
│ MetaLabelTraining    │    │ SignalScoringService     │
│ Service (composer)   │    │ (inference composer)     │
└────┬─────┬─────┬─────┘    └──┬──────────┬────────────┘
     │     │     │              │          │
     │     │     │              │          │  (reuse)
     ▼     ▼     ▼              ▼          ▼
┌────────┐┌────┐┌──────────┐┌──────────┐┌──────────────────┐
│Primary ││Meta││PurgedKFold││Primary  ││ModelRegistry     │
│Signal  ││Labl│└──────────┘│Signal   ││(existing, reuse) │
│Service ││Gen │             │Service  │└──────────────────┘
└───┬────┘└─┬──┘             └─────────┘
    │       │
    ▼       ▼  (pure functions)
┌────────┐┌──────────────────────────┐
│existing││app/ml/labels/meta_label.py│
│strategy││ triple-barrier + dyn vol │
│registry││app/ml/split/purged_kfold │
└────────┘└──────────────────────────┘
```

### Component responsibilities

| Module | Path | Responsibility | New / Reused |
|---|---|---|---|
| Triple-barrier generator | `app/ml/labels/meta_label.py` | Pure fn: OHLC + signals + vol_series + (tp_k, sl_k, timeout) → DataFrame of events with primary_signal, tp, sl, t1_hit_time, realized_R, primary_direction_correct | 🆕 |
| Purged K-Fold splitter | `app/ml/split/purged_kfold.py` | Pure fn: event spans + n_splits + embargo_pct → (train_idx, test_idx) tuples with overlap purged | 🆕 |
| Primary Signal Service | `app/services/primary_signal_service.py` | Dispatch: `strategy_name` → rule strategy; `primary_model_id` → load + predict → sign(proba−0.5). Returns Series[-1/0/+1] + signal_strength float | 🆕 |
| Meta-Label Training Service | `app/services/meta_label_service.py` | End-to-end: OHLC fetch → primary signals → triple-barrier labels → event features → Purged CV → train XGBoost/ensemble → register | 🆕 |
| Signal Scoring Service | `app/services/signal_scoring_service.py` | Mode A (explicit signal) / B (auto-trigger) / C (fallback). Returns reliability_score + expected_R + sizing_hint | 🆕 |
| Meta-label metrics | `app/ml/metrics/meta.py` | precision-at-50%, expected_R-when-trade, hit-rate-when-trade | 🆕 |
| Label registry wiring | `app/ml/labels/registry.py` | Swap `meta_label` from `_not_implemented_meta_label` to real generator | 🔧 Mod |
| Training service routing | `app/services/training_service.py` | `LABEL_TYPE_TO_TASK["meta_label"] = "classification"` + meta-label branch | 🔧 Mod |
| API router | `app/api/routers/signal.py` | 2 endpoints + Pydantic request/response schemas | 🆕 |
| Paper Trading integration | `app/trading/engine.py` | `place_order(..., meta_model_id, score_threshold)`: gates + sizes by score | 🔧 Mod |
| Paper Trading config | `app/trading/models.py` | `PaperTradingConfig.meta_label_enabled: bool = False` + `default_score_threshold: float = 0.55` | 🔧 Mod |

### Key isolation invariants

1. Triple-barrier and Purged K-Fold have **no DB or model dependencies** — pure functions of their inputs. Unit-testable in isolation.
2. `build_event_features()` is shared between train and inference paths — prevents feature drift.
3. `vol_series` must come from the same source at train and inference (stored in meta-model metadata as `barrier.vol_source`).
4. MetaLabelTrainingService is the only orchestrator that touches DB + models; everything else is pure or a single-responsibility service.

## 5. Data Flow

### 5.1 Training pipeline · `POST /api/meta-label/train`

```
Input: { ticker, primary: {strategy_name OR model_id, strategy_params},
         barrier: {tp_k, sl_k, timeout_days, vol_source},
         cv: {n_splits, embargo_pct},
         model: {type, ensemble_mode, search_mode},
         window: {lookback_days, feature_group} }
         │
         ▼
① DatasetBuilder — OHLC + ta_basic features (existing; reused)
         │
         ▼
② PrimarySignalService.dispatch(primary)
     strategy_name → run rule strategy → Series[-1/0/+1]
     primary_model_id → load + predict → sign(proba−0.5) + proba_strength
     └─→ signals_series (mostly 0; non-zero at trigger times)
         │
         ▼
③ vol_series
     if vol_source == "p1_model": P1 vol_model.predict() per day
     else: realized σ = returns.rolling(20).std() × sqrt(252)
     If p1_model requested but unavailable → auto-fallback to realized_sigma + warning
         │
         ▼
④ triple_barrier_labels(ohlc, signals, vol_series, tp_k, sl_k, timeout_days)
     For each non-zero signal at t0 with direction d ∈ {+1, −1}:
       vol_at_t0 = vol_series[t0 − 1]  # lagged by 1 bar to avoid look-ahead
       tp_price = close[t0] × (1 + d × tp_k × vol_at_t0)
       sl_price = close[t0] × (1 − d × sl_k × vol_at_t0)
       t1 = min timestamp in (t0, t0 + timeout_days] where the bar's high/low crosses tp_price or sl_price
       If tp hit first →           realized_R = +tp_k  (in the trade's favor)
       If sl hit first →           realized_R = −sl_k
       If timeout before either →  realized_R = d × (close[t1] − close[t0]) / (sl_k × vol_at_t0 × close[t0])
       If both barriers touched in same bar (ambiguous intraday) → assume SL first (conservative)
       primary_direction_correct = 1 if realized_R > 0 else 0     # ← meta-label target
     → DataFrame indexed by event_time
         │
         ▼
⑤ build_event_features(ohlc_ta, events, vol_series)
     X @ event_time = ta_basic (~30 dim, sliced at row t0 − 1 to avoid look-ahead)
                    + signal_time_vol    (= vol_series[t0 − 1])
                    + signal_strength    (|proba−0.5|×2 for model primary; 1.0 for rules)
                    + time_since_last_signal (days since prior trigger of the same primary source)
                    ≈ 33 dim
     y = events.primary_direction_correct (binary)
         │
         ▼
⑥ PurgedKFold(n_splits=5, embargo_pct=0.01) over event spans (t0, t1)
     → 5 folds, each (train_idx, test_idx) with:
       - overlapping events purged from train
       - embargo zone after test excluded from train
         │
         ▼
⑦ Train XGBoost (single) or EnsembleModel (voting/stacking) per fold
     aggregate metrics: CV AUC (mean/std), precision@50%, expected_R_when_trade, hit_rate_when_trade
         │
         ▼
⑧ Final fit on ALL events (no held-out since events are scarce)
         │
         ▼
⑨ ModelRegistry.save(
     task="classification",
     label_type="meta_label",
     extras={
       primary: {source, strategy_name|model_id, params},
       barrier: {tp_k, sl_k, timeout_days, vol_source},
       cv: {n_splits, embargo_pct, metrics},
       event_count, class_balance,
       feature_set: [...33 names]
     }
   )
         │
         ▼
Response { model_id, event_count, class_balance, cv_metrics, barrier_config_used, warnings }
```

### 5.2 Inference pipeline · `POST /api/signal-score`

```
Request → SignalScoringService.score(req)
         │
         ├─ mode A (explicit signal):
         │    req has {signal: +1/-1, timestamp}
         │    → load meta_model (+ its feature_set, vol_source)
         │    → fetch OHLC[timestamp − lookback : timestamp]
         │    → compute vol_series (same source as training)
         │    → build_event_features(single row at timestamp)
         │    → meta_model.predict_proba()[1] → reliability_score
         │    → expected_R ≈ tp_k × score − sl_k × (1 − score)  [from barrier config]
         │
         ├─ mode B (auto-trigger):
         │    req has {strategy_name | primary_model_id} (no signal)
         │    → fetch OHLC latest lookback
         │    → PrimarySignalService.dispatch → latest non-zero signal
         │    → if latest signal == 0 → return {triggered: false, signal: 0, reason}
         │    → otherwise continue as mode A
         │
         └─ mode C (fallback logic):
              both signal + strategy_name given → A path (explicit wins)
              neither → 400
              only signal → A path
              only strategy/model → B path

Response { triggered, signal, reliability_score, expected_R,
           recommended_action (trade|skip|reduce), sizing_hint,
           meta_model {id, primary_source, cv_auc}, timestamp }
```

## 6. API Contracts

### 6.1 `POST /api/meta-label/train`

**Request:**
```json
{
  "ticker": "AAPL",
  "primary": {
    "source": "strategy",
    "strategy_name": "rsi_strategy",
    "strategy_params": { "lower": 30, "upper": 70 },
    "primary_model_id": null
  },
  "barrier": {
    "tp_k": 2.0,
    "sl_k": 1.0,
    "timeout_days": 5,
    "vol_source": "p1_model"
  },
  "cv": { "n_splits": 5, "embargo_pct": 0.01 },
  "model": {
    "type": "xgboost",
    "ensemble_mode": null,
    "search_mode": "default"
  },
  "window": { "lookback_days": 730, "feature_group": "ta_basic" }
}
```

**Primary source contract:** exactly one of `strategy_name` or `primary_model_id` must be non-null. If `source=="strategy"` → `strategy_name` required. If `source=="model"` → `primary_model_id` required.

**Response 200:**
```json
{
  "success": true,
  "model_id": "meta_aapl_rsi_20260424_a3f2",
  "registered": true,
  "event_count": 184,
  "class_balance": { "correct": 102, "wrong": 82 },
  "cv_metrics": {
    "auc_mean": 0.612, "auc_std": 0.041,
    "precision_at_50": 0.68,
    "expected_R_when_trade": 0.43,
    "hit_rate_when_trade": 0.58
  },
  "barrier_config_used": {
    "tp_k": 2.0, "sl_k": 1.0, "timeout_days": 5, "vol_source": "p1_model"
  },
  "warnings": []
}
```

**Error responses:**
- `400 insufficient_events` — `event_count < 30`
- `400 primary_source_conflict` — both `strategy_name` and `primary_model_id` given
- `400 primary_source_missing` — neither given
- `400 primary_task_mismatch` — `primary_model_id` points to a regression model (e.g., a volatility model)
- `404 primary_model_not_found`
- `422` — Pydantic validation (tp_k ≤ 0, n_splits < 2, etc.)

### 6.2 `POST /api/signal-score`

**Request (mode C, flexible):**
```json
{
  "ticker": "AAPL",
  "meta_model_id": "meta_aapl_rsi_20260424_a3f2",
  "signal": null,
  "timestamp": null,
  "strategy_name": "rsi_strategy",
  "strategy_params": null
}
```

**Response · mode A (explicit signal accepted):**
```json
{
  "triggered": true,
  "signal": 1,
  "reliability_score": 0.71,
  "expected_R": 0.54,
  "recommended_action": "trade",
  "sizing_hint": {
    "half_kelly_fraction": 0.18,
    "raw_kelly": 0.36,
    "cap": 0.25
  },
  "meta_model": {
    "id": "meta_aapl_rsi_20260424_a3f2",
    "primary_source": "strategy:rsi_strategy",
    "cv_auc": 0.612
  },
  "timestamp": "2026-04-24T20:00:00Z"
}
```

**Response · mode B (auto-trigger, primary silent):**
```json
{
  "triggered": false,
  "signal": 0,
  "reason": "rsi_strategy did not trigger at latest close",
  "timestamp": "2026-04-24T20:00:00Z"
}
```

**Recommended action thresholds (hardcoded defaults; overridable per-call in a future v2):**
- `score ≥ 0.65` → `"trade"` (full sizing)
- `0.45 ≤ score < 0.65` → `"reduce"` (half sizing)
- `score < 0.45` → `"skip"`

**expected_R formula:** `tp_k × score − sl_k × (1 − score)` using the meta-model's registered `barrier` config. This is an *approximation* assuming timeout exits average to zero; the benchmark should note if this diverges from empirical realized_R on the held-out folds.

**Error responses:**
- `400 timestamp_out_of_range` — requested timestamp outside available OHLC
- `400 mode_ambiguous` — neither signal nor strategy/model given
- `404 meta_model_not_found`
- `404 primary_model_not_found` (mode B with `primary_model_id`)

## 7. Paper Trading Integration

Change to `app/trading/engine.py::place_order`:

```python
def place_order(ticker, side, qty, meta_model_id=None, score_threshold=None, ...):
    if meta_model_id is not None:
        score_resp = signal_scoring_service.score(
            ticker=ticker,
            signal=1 if side == "buy" else -1,
            meta_model_id=meta_model_id,
            timestamp=now(),
        )
        threshold = score_threshold or PaperTradingConfig.default_score_threshold
        if score_resp.reliability_score < threshold:
            return OrderRejected(
                reason="meta_score_below_threshold",
                score=score_resp.reliability_score,
                threshold=threshold,
            )
        qty = int(qty * score_resp.sizing_hint.half_kelly_fraction / 0.25)
    # ... existing order logic unchanged
```

**Config additions** in `app/trading/models.py::PaperTradingConfig`:

```python
meta_label_enabled: bool = False          # UI toggle, off by default
default_score_threshold: float = 0.55     # override per-order via score_threshold arg
```

**Backward compatibility:** when `meta_model_id=None` (existing callers), the function behaves identically to before. Existing Paper Trading tests remain green without modification.

**UI deferred to P4:** model selector dropdown, threshold slider, score display in order confirmation, live reliability_score on open positions.

## 8. Data Models

### 8.1 Event row (internal, not exposed)

```python
@dataclass
class TripleBarrierEvent:
    event_time: pd.Timestamp
    ticker: str
    primary_signal: int            # -1 or +1 (0 signals filtered out)
    signal_strength: float         # 1.0 for rules; |proba-0.5|*2 for ML primary
    entry_price: float
    tp_price: float
    sl_price: float
    timeout_time: pd.Timestamp
    t1_hit_time: pd.Timestamp      # actual exit time
    t1_barrier: Literal["tp", "sl", "timeout"]
    realized_R: float              # in trade's favor: +tp_k at tp, −sl_k at sl, fractional at timeout
    primary_direction_correct: int # 1 if realized_R > 0 else 0 → meta-label target
```

### 8.2 Meta-model registry record extras (existing `ModelRecord.extras` JSON field)

```json
{
  "meta_label": {
    "primary": { "source": "strategy", "strategy_name": "rsi_strategy", "params": {...} },
    "barrier": { "tp_k": 2.0, "sl_k": 1.0, "timeout_days": 5, "vol_source": "p1_model" },
    "cv": { "n_splits": 5, "embargo_pct": 0.01 },
    "event_count": 184,
    "class_balance": { "correct": 102, "wrong": 82 },
    "feature_set": ["rsi_14", "macd", "..."],
    "p1_vol_model_id_used": "vol_aapl_lgbm_20260422"
  }
}
```

`label_type="meta_label"` is the canonical tag (already in `LabelConfig.label_type` Literal from P1).

## 9. Error Handling

| Scenario | Behavior |
|---|---|
| `event_count < 30` (primary rarely triggers) | `400 insufficient_events` + counts + min_required. Do not train. |
| P1 vol model missing + `vol_source="p1_model"` | Auto-fallback to `realized_sigma` + `warnings` array. Store fallback in registry record. |
| Extreme class imbalance (correct<10% or >90%) | Train anyway; attach warning; use `scale_pos_weight`/`class_weight="balanced"`. |
| CV fold with empty train or test (after purge) | Skip fold + warning. Require ≥3 usable folds total. |
| Primary model `task != "classification"` | `400 primary_task_mismatch` (guards against using a vol model as primary). |
| Signal-score timestamp outside OHLC range | `400 timestamp_out_of_range`. |
| Paper Trading meta_model ticker ≠ order ticker | `OrderRejected(reason="meta_model_ticker_mismatch")`. |
| NaN in OHLC at event_time | Skip that event; log count in warnings. |
| Triple-barrier both tp and sl hit same day (intraday ambiguity) | Conservative: assume SL hit first (worst case). Doc this in code comment. |

## 10. Testing Strategy

**Total: 25-35 tests across 8 files · TDD RED → GREEN per batch**

| Batch | File | Count | Coverage |
|---|---|---|---|
| 1 · Triple-barrier | `tests/test_meta_label_barrier.py` | 6 | TP hit → +1 realized_R=tp_k · SL hit → −1 · timeout → fractional · vol=0 edge → timeout path · NaN OHLC → skip · signal=−1 symmetry |
| 2 · Meta-label target | same file | 3 | primary=+1 & R>0 → meta=1 · primary=−1 & R>0 (reversed) → meta=0 · primary=0 events dropped |
| 3 · Purged K-Fold | `tests/test_purged_kfold.py` | 5 | overlap events purged from train · embargo applied correctly · n_splits=5 returns 5 folds · empty input raises · single event skips fold gracefully |
| 4 · Primary Signal Service | `tests/test_primary_signal_service.py` | 4 | strategy_name dispatches to rule · primary_model_id loads & predicts · both given → raise · neither given → raise |
| 5 · Event feature builder | `tests/test_meta_event_features.py` | 3 | ta_basic correctly sliced at event_time · signal_time_vol from P1 model · time_since_last_signal arithmetic |
| 6 · Contract: meta-label train | `tests/contract/test_meta_label_train.py` | 5 | 200 w/ strategy primary · 200 w/ model primary · 400 insufficient_events · 404 primary_model_not_found · 422 tp_k≤0 |
| 7 · Contract: signal-score | `tests/contract/test_signal_score.py` | 5 | mode A explicit signal · mode B auto strategy trigger · mode B strategy silent → triggered=false · signal+strategy both → A wins · meta_model 404 |
| 8 · Paper Trading integration | `tests/test_paper_trading_meta.py` | 4 | `meta_label_enabled=False` preserves legacy behavior · score<threshold → OrderRejected · score≥threshold → sized order · meta_model ticker mismatch → reject |

**Regression guard:** `tests/test_ensemble_training.py` + `tests/contract/test_train_flow.py` + `tests/test_labels.py` + `tests/contract/test_predict_volatility.py` must all remain green.

**Live benchmark (EOD ship):** `scripts/p3_meta_label_benchmark.py`. Runs `AAPL + MSFT + GOOGL × rsi_strategy` end-to-end. Trains meta-model per ticker. Reports CV AUC / precision-at-50 / expected_R_when_trade / class_balance. Written to `docs/benchmarks/p3_meta_label_benchmark.md` + `vault/.../p3-benchmark-2026-04-24.md`. Matches P1 benchmark in honesty tone — numbers may be weak because event counts are small; that is itself a valid methodology finding.

## 11. Non-Goals

- ❌ No frontend work (P4 tomorrow)
- ❌ No new feature groups (ta_basic only; sentiment/regime deferred)
- ❌ No custom user strategies (the 4 built-in rules only)
- ❌ No Dollar Bars (Prado Ch.2 — out of P3 scope; would require data layer rework)
- ❌ No Deflated Sharpe or MDA feature importance (P9 or later)
- ❌ No live meta-model decay monitoring (future backlog)
- ❌ No migration of existing models — P3 only adds a new `label_type` path

## 12. Future Backlog

Explicit deferred enhancements captured during brainstorming:

1. **Feature expansion** — add `sentiment` feature group; regime-state features (waits on P9); cross-sectional statistics (other tickers at same timestamp); fundamental ratios; intraday features if bar frequency changes.
2. **Dynamic barrier upgrade** — asymmetric TP/SL search (not fixed 2:1); per-strategy optimal barrier configuration search.
3. **Primary source extensions** — ensemble primary (vote across multiple rules); user-uploaded custom strategy; sentiment-driven primary from LLM agents.
4. **Meta-model types** — neural sequence-aware meta-model; Platt-scaling calibration layer for better probability estimates.
5. **Monitoring** — rolling CV AUC over time for meta-model decay tracking; drift detection on event feature distributions.
6. **UI (P4)** — Signal Console page with per-strategy meta-score leaderboard; Meta-Label Coverage badge on strategy cards; Paper Trading meta-toggle + threshold slider + live score display.

## 13. Success Criteria

- [ ] All 25-35 new tests + regression guard green
- [ ] `POST /api/meta-label/train` trains a model on AAPL+rsi_strategy using live yfinance data and registers it in the registry
- [ ] `POST /api/signal-score` returns valid response in mode A, mode B (triggered), mode B (silent)
- [ ] Paper Trading integration: order with `meta_model_id` + low-quality synthetic signal is rejected; high-quality signal places sized order
- [ ] Benchmark report committed with honest CV metrics
- [ ] All changes committed in a single PR or direct-to-main sequence (Harry's call on PR process)
- [ ] P4 tomorrow can consume `POST /api/signal-score` from the frontend without backend changes

## 14. Dependencies

**Depends on (must exist):**
- P1 Volatility Backend (merged) — for `vol_source="p1_model"` path and the registry schema with `label_type`
- P2 G3 (`/models?label_type=...` filter) — already shipped; frontend P4 needs this
- Existing strategies in `app/strategies/templates/` (ma_cross, rsi_strategy, bollinger_breakout, sentiment_driven)
- Existing ModelRegistry, DatasetBuilder, EnsembleModel, BaseModel machinery

**Unblocks:**
- P4 Signal Console frontend (tomorrow)
- P6 Sub 6 Paper Trading productization (meta-score is a natural citizen of that UI)

## 15. Change Log

- 2026-04-24: Initial design approved after 4-section brainstorming walkthrough with Harry. Scope locked to full backend (P4 is UI). Triple-barrier uses dynamic vol from P1 model (option C). Primary source accepts both rules and ML direction model (option B). Signal-score API is mode C (flexible A/B fallback). Purged K-Fold is mandatory.

## 16. References

- López de Prado, *Advances in Financial Machine Learning*, Ch.3 (meta-labeling, triple-barrier), Ch.7 (Purged K-Fold CV)
- Patton (2011) QLIKE loss — used for P1 vol metrics, relevant again when `vol_source="p1_model"` is chosen
- `D:\obsidian vault\Quant\_Knowledge_Base.md` §4 (Meta-Labeling, Triple-Barrier, Purged K-Fold)
- `master-roadmap.md` §P3 and `ml-pivot-task-plan.md` §Phase 3
- `sub-4-modeling-dialog-wip.md` — P4 UI dependency
- Prior spec: `2026-04-15-ensemble-models-design.md` (EnsembleModel reuse pattern)
