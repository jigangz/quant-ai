# P5 G1 Prediction Log + Leaderboard + Ablation Design Spec

**Date**: 2026-04-25
**Author**: Harry + Claude
**Status**: Approved (pending implementation)
**Phase**: V4 Pivot · Phase 5 (Gate 2 starter) — G1 backend + FE-ENH-1 Leaderboard + FE-ENH-4 Multi-target Ablation
**Predecessors**: V4 Gate 1 complete (P1+P2+P3+P4 shipped 2026-04-24, tag `v4-gate-1-complete`)

## 1. Goal

Move Quant AI past CV-only metrics into **live accuracy tracking** + **methodological feature attribution**. Three deliverables in one spec:

1. **G1 backend** — every `/predict*` and `/api/signal-score` call writes a row to a `prediction_log` table; a lazy `AccuracyService` resolves rows whose `predicted_at + horizon_days` has passed by fetching actual close prices and computing hit-rate / realized R / volatility error. Exposed via `GET /models/{id}/accuracy?window_days=30`.
2. **FE-ENH-1 Leaderboard** — `/leaderboard` page showing 3 tabs (direction / volatility / meta_label), each tab a table sorted by primary metric, augmented with the live accuracy from G1.
3. **FE-ENH-4 Ablation** — `/ablation` page driving a new `POST /api/ablation/run` endpoint which trains 6 models (3 targets × 2 feature sets `[ta_basic, ta_basic + sentiment]`), returns a delta matrix, and renders a heatmap with summary interpretation.

**Why this matters for interview narrative**: Gate 1 proves the V4 multi-task ML platform exists. P5 proves it produces **honest, measurable, attributable** results. After P5 we can answer: "How accurate is your direction model?" with real numbers, "Does sentiment help?" with quantified deltas across all 3 targets, and "Which strategy/ticker meta-model is best?" with a sortable leaderboard.

## 2. Design Principles

- **Reuse existing infra** — ModelRegistry pattern (LocalRegistry + SupabaseRegistry split), `prediction_event_publisher` Kafka events stay where they are, `training_service` handles direction/vol target trainings, `meta_label_service` handles meta_label target.
- **Lazy resolve** — `/models/{id}/accuracy` triggers resolution on-demand. No cron, no scheduled tasks. Render free tier compatible.
- **Non-blocking writes** — `prediction_log` insert wrapped in try/except in every predict service. A registry outage cannot break a prediction response.
- **Drop-feature ablation** — sentiment vs no-sentiment is the simplest, most interpretable ablation. Permutation importance is within-model and doesn't translate cross-target. Drop-feature retraining gives a single number per (target, feature_set) cell that is directly comparable.
- **Default params for fair comparison** — Ablation does NOT run Optuna. Same model_type, same horizon, same window across all 6 cells; only feature set differs.
- **Multi-task respect** — 3 sub-deliverables share table + service layer. Don't fork into per-target schemas.

## 3. Scope

### In (Day 1 ship)

- **DB layer**:
  - `app/db/prediction_log.py` — `PredictionLogRecord` + `LocalPredictionLogRepo` + `SupabasePredictionLogRepo` + `get_prediction_log_repo()` factory.
  - `scripts/migrate_create_prediction_log.sql` — Supabase table + indexes (idempotent).
- **Service layer**:
  - `app/services/accuracy_service.py` — `resolve_pending(model_id, limit=100)` + `aggregate(model_id, window_days=30)`.
  - `app/services/ablation_service.py` — `run_ablation(ticker, targets, feature_sets, horizon_days, model_type)` orchestrator.
  - `app/services/meta_label_service.py` — small backward-compat extension: `MetaLabelTrainRequest.feature_group` accepts `str | list[str]` (single string still works for existing callers).
- **API layer**:
  - `GET /models/{model_id}/accuracy` — accuracy endpoint with optional `?resolve=true` and `?window_days=30`.
  - `POST /api/ablation/run` — synchronous (~30s) ablation orchestrator endpoint.
- **Predict-write integration**:
  - `app/services/predict_service.py` writes prediction_log row after `publish_prediction_event`.
  - `app/services/volatility_predict_service.py` writes prediction_log row.
  - `app/services/signal_scoring_service.py` writes prediction_log row when Mode A actually scores.
- **Frontend**:
  - `LeaderboardPage` (3 tabs) + `LeaderboardTable` component.
  - `AblationPage` (form + matrix heatmap) + `AblationMatrix` component.
  - api/client + leaderboardQueries hooks.
  - Routes wired in `App.jsx`; nav links in `TopNavBar`.
- **Live demo**:
  - `scripts/p5_ablation_demo.py` runs ablation on AAPL+MSFT+GOOGL, writes `docs/benchmarks/p5_ablation_demo.md`.
- **Tests**: 47 (27 backend + 17 frontend + 3 hooks).
- **GATE**: regression + progress log Day 15 + tag `v4-p5-complete` + live smoke.

### Out (deferred — see §13 Future Backlog)

- ❌ Per-prediction SHAP at log time
- ❌ Rolling-AUC time-decay chart
- ❌ Cross-ticker ablation matrix
- ❌ Custom user-uploaded feature groups
- ❌ Ablation with 3+ feature sets in one run
- ❌ Optuna in ablation cells (defeats fair comparison)
- ❌ Cron-based scheduled resolution (lazy is enough for free tier)
- ❌ Scaling to >100k logged predictions per model (paginate later)

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Frontend                                                     │
│  /leaderboard   3 tabs (direction/volatility/meta_label)    │
│  /ablation      3 targets × 2 feature_sets heatmap          │
└────────────┬───────────────────────┬─────────────────────────┘
             │                       │
             ▼                       ▼
┌──────────────────────┐  ┌──────────────────────────────┐
│ Backend G1 surface   │  │ Backend Ablation surface     │
│  GET /models/{id}/   │  │  POST /api/ablation/run      │
│      accuracy        │  │       {ticker, targets,      │
│                      │  │        feature_sets}         │
└────────┬─────────────┘  └────────┬─────────────────────┘
         │                         │
         ▼                         ▼
┌──────────────────────┐  ┌──────────────────────────────┐
│ AccuracyService      │  │ AblationService              │
│  resolve_pending()   │  │  run_ablation()              │
│  aggregate(window)   │  │   ├── training_service.train()
│                      │  │   └── meta_label_service.train()
└────────┬─────────────┘  └────────┬─────────────────────┘
         │                         │
         ▼                         ▼
┌─────────────────────────────────────────────────────────────┐
│ prediction_log table (Supabase / Local JSON)                 │
│  id · model_id · ticker · label_type · predicted_at         │
│  predicted_value · predicted_signal · predicted_extras       │
│  resolve_at · actual_value · actual_return · is_correct     │
│  realized_R · resolved_at · horizon_days                    │
└──────────────────────────────────────────────────────────────┘
                  ▲
                  │ writes (non-blocking try/except)
┌────────────────┬─────────────────────┬──────────────────────┐
│predict_service │volatility_predict_  │signal_scoring_       │
│(direction)     │service (vol)        │service (meta)        │
└────────────────┴─────────────────────┴──────────────────────┘
```

### Component responsibilities

| Module | Path | Responsibility | New / Mod |
|---|---|---|---|
| PredictionLogRecord | `app/db/prediction_log.py` | Pydantic schema + write fields validation | 🆕 |
| LocalPredictionLogRepo | `app/db/prediction_log.py` | JSON-on-disk for local dev | 🆕 |
| SupabasePredictionLogRepo | `app/db/prediction_log.py` | Supabase prediction_log table client | 🆕 |
| `get_prediction_log_repo()` | `app/db/prediction_log.py` | Factory: returns Supabase if configured else Local | 🆕 |
| AccuracyService | `app/services/accuracy_service.py` | resolve_pending(model_id) + aggregate(model_id, window) | 🆕 |
| AblationService | `app/services/ablation_service.py` | run_ablation(ticker, targets, feature_sets) → matrix dict | 🆕 |
| Accuracy router | `app/api/accuracy.py` | GET /models/{id}/accuracy | 🆕 |
| Ablation router | `app/api/ablation.py` | POST /api/ablation/run | 🆕 |
| predict_service write hook | `app/services/predict_service.py` | After predict success: insert PredictionLogRecord | 🔧 |
| volatility_predict_service write hook | `app/services/volatility_predict_service.py` | Same | 🔧 |
| signal_scoring_service write hook | `app/services/signal_scoring_service.py` | Same (Mode A only) | 🔧 |
| main.py router wiring | `app/main.py` | include accuracy + ablation routers | 🔧 |
| LeaderboardPage | `quant-ai-ui/src/pages/LeaderboardPage.jsx` | 3-tab page | 🆕 |
| LeaderboardTable | `quant-ai-ui/src/features/leaderboard/LeaderboardTable.jsx` | Sortable table per tab | 🆕 |
| AblationPage | `quant-ai-ui/src/pages/AblationPage.jsx` | Form + matrix view | 🆕 |
| AblationMatrix | `quant-ai-ui/src/features/ablation/AblationMatrix.jsx` | 3×2 heatmap with delta colors | 🆕 |
| leaderboardQueries | `quant-ai-ui/src/api/leaderboardQueries.js` | useLeaderboard / useModelAccuracy / useAblationRun | 🆕 |
| api/client.js | `quant-ai-ui/src/api/client.js` | getModelAccuracy / postAblationRun | 🔧 |
| App.jsx routes | `quant-ai-ui/src/App.jsx` | /leaderboard + /ablation | 🔧 |
| TopNavBar links | `quant-ai-ui/src/components/layout/TopNavBar.jsx` | Leaderboard in 「模型」, Ablation in 「研究」 | 🔧 |
| Migration SQL | `scripts/migrate_create_prediction_log.sql` | Idempotent table + index | 🆕 |
| Demo script | `scripts/p5_ablation_demo.py` | Run ablation on 3 tickers, write markdown | 🆕 |

## 5. Database Schema

### `prediction_log` table

```sql
CREATE TABLE prediction_log (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id        TEXT NOT NULL,
  ticker          TEXT NOT NULL,
  label_type      TEXT NOT NULL CHECK (label_type IN ('direction','volatility','meta_label')),

  -- prediction data
  predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  horizon_days    INTEGER NOT NULL,
  predicted_value NUMERIC NOT NULL,        -- proba (direction/meta) | vol scalar
  predicted_signal INTEGER,                -- -1/0/+1 (direction/meta only); NULL for vol
  predicted_extras JSONB NOT NULL DEFAULT '{}'::jsonb,

  -- resolution (NULL until resolved)
  resolve_at      TIMESTAMPTZ NOT NULL,    -- predicted_at + horizon_days (business days OK approximated)
  actual_value    NUMERIC,                 -- close at resolve_at OR realized_vol over window
  actual_return   NUMERIC,                 -- (close_resolve - close_predict) / close_predict
  is_correct      BOOLEAN,                 -- direction/meta only; NULL for vol
  realized_R      NUMERIC,                 -- in trade's favor frame
  resolved_at     TIMESTAMPTZ,

  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_pred_log_model_id ON prediction_log(model_id);
CREATE INDEX idx_pred_log_resolve_pending ON prediction_log(resolve_at) WHERE resolved_at IS NULL;
CREATE INDEX idx_pred_log_ticker_label ON prediction_log(ticker, label_type);
```

LocalPredictionLogRepo serializes the same fields to `artifacts/registry/prediction_log.json` as a dict keyed by id.

### `PredictionLogRecord` Pydantic schema

```python
class PredictionLogRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    model_id: str
    ticker: str
    label_type: Literal["direction", "volatility", "meta_label"]

    predicted_at: datetime = Field(default_factory=datetime.utcnow)
    horizon_days: int
    predicted_value: float
    predicted_signal: int | None = None
    predicted_extras: dict[str, Any] = Field(default_factory=dict)

    resolve_at: datetime
    actual_value: float | None = None
    actual_return: float | None = None
    is_correct: bool | None = None
    realized_R: float | None = None
    resolved_at: datetime | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(extra="ignore")
```

## 6. Data Flow

### 6.1 Write path (3 predict services)

For each prediction:

```python
# After existing publish_prediction_event(...) call:
try:
    repo = get_prediction_log_repo()
    repo.insert(PredictionLogRecord(
        model_id=model_id,
        ticker=ticker,
        label_type=label_type,
        horizon_days=horizon,
        predicted_value=predicted_value,
        predicted_signal=signal_int_or_None,
        predicted_extras={
            "primary_source": primary_source_str,        # meta_label only
            "expected_R": expected_R,                    # meta_label only
            "feature_group": feature_group_str,
            "model_type": model_type,
        },
        resolve_at=now() + timedelta(days=horizon_days),
    ))
except Exception as e:
    logger.warning(f"prediction_log write failed (non-blocking): {e}")
```

The write is **fire-and-forget non-blocking**. Failure logs but doesn't propagate.

### 6.2 Resolve path (`AccuracyService.resolve_pending`)

```
Input: model_id (str), limit (int = 100)

1. SELECT * FROM prediction_log
   WHERE model_id = ? AND resolved_at IS NULL AND resolve_at < NOW()
   ORDER BY resolve_at ASC
   LIMIT ?

2. For each row:
   a. Fetch the OHLC slice [predicted_at − 30d, resolve_at + 1d] via
      market_data_provider. Cache the slice keyed by (ticker, predicted_at)
      so consecutive rows for the same ticker reuse the result.
   b. If fetch fails → skip row (try again next /accuracy call).
   c. Compute close_predict (close at predicted_at) and close_resolve (close at resolve_at).
      If either is NaN/missing → skip row, log warning.
   d. Compute:
      actual_return = (close_resolve - close_predict) / close_predict
      For direction/meta:
        is_correct = (predicted_signal == sign(actual_return))
        vol_at_predict = stdev(returns[predicted_at − 20d : predicted_at]) × sqrt(252)  (rolling 20d annualized vol; fallback 0.02 if zero)
        realized_R = predicted_signal × actual_return / vol_at_predict
      For volatility:
        # Realized vol needs daily closes BETWEEN predicted_at and resolve_at,
        # which is why we fetch the OHLC slice (not just two endpoints).
        returns_window = pct_change(close[predicted_at..resolve_at])
        actual_value = stdev(returns_window) × sqrt(252)  (annualized)
        is_correct = NULL  (regression target — no hit/miss)
        realized_R = NULL  (no trade frame for vol prediction)
   e. UPDATE row with resolved fields + resolved_at = NOW()

3. Return { checked, newly_resolved, errors }
```

### 6.3 Aggregate path (`AccuracyService.aggregate`)

```python
# Pseudocode:
window_start = now() - timedelta(days=window_days)
rows = repo.list_by_model_id(model_id, since=window_start)

resolved = [r for r in rows if r.resolved_at]
pending = [r for r in rows if not r.resolved_at]

stats = {
    "total_predictions": len(rows),
    "resolved": len(resolved),
    "pending": len(pending),
}

if rows[0].label_type in ("direction", "meta_label"):
    correct_resolved = [r for r in resolved if r.is_correct]
    stats["hit_rate"] = len(correct_resolved) / max(len(resolved), 1)
    stats["avg_realized_R"] = mean([r.realized_R for r in resolved if r.realized_R is not None])
    stats["best_R"] = max([r.realized_R for r in resolved if r.realized_R is not None], default=None)
    stats["worst_R"] = min(...)
elif rows[0].label_type == "volatility":
    actuals = [r.actual_value for r in resolved if r.actual_value is not None]
    preds = [r.predicted_value for r in resolved if r.actual_value is not None]
    stats["mae"] = mean([abs(a - p) for a, p in zip(actuals, preds)])
    stats["rmse"] = sqrt(mean([(a - p)**2 for ...]))

# by_ticker breakdown (group rows by ticker, same calc)
# last_predictions: most recent 20 rows with all fields

return { model_id, label_type, window_days, stats, by_ticker, last_predictions }
```

### 6.4 Ablation path (`AblationService.run_ablation`)

```
Input: ticker, targets=[...], feature_sets=[{name, groups}, ...], horizon_days, model_type

For each (target, feature_set) in cartesian product:
    if target == "direction":
        result = training_service.train(TrainRequest(
            ticker=ticker, model_type=model_type,
            label_type="direction", horizon_days=horizon_days,
            feature_groups=feature_set.groups,
            search_mode="default",  # NO Optuna
        ))
        metric = result.metrics.test_auc
    elif target == "volatility":
        result = training_service.train(TrainRequest(
            ticker=ticker, model_type=model_type,
            label_type="volatility", horizon_days=horizon_days,
            feature_groups=feature_set.groups,
        ))
        metric = result.metrics.test_qlike  # lower is better
    elif target == "meta_label":
        # NOTE: P3's MetaLabelTrainRequest.feature_group accepts a single
        # string today. For P5 ablation we extend it to accept either a
        # string OR a list[str] (backward-compatible). The internal
        # `_apply_ta_features` already calls `get_feature_builders([fg])`
        # so it just becomes pass-through. This is one of the few
        # backward-compat fixes that ships in P5 to enable cross-target
        # ablation.
        result = meta_label_service.train_meta_label_model(
            MetaLabelTrainRequest(
                ticker=ticker,
                primary=PrimarySignalSpec(source="strategy", strategy_name="rsi_strategy"),
                tp_k=2.0, sl_k=1.0, timeout_days=horizon_days,
                vol_source="realized_sigma",
                cv_n_splits=5, cv_embargo_pct=0.01,
                model_type=model_type,
                lookback_days=730,
                feature_group=feature_set.groups,  # str | list[str] after P5 extension
            ),
        )
        metric = result.cv_metrics.auc_mean

    matrix[target][feature_set.name] = { model_id, metric, secondary_metrics, ... }

# Compute deltas relative to feature_sets[0] as baseline:
for target in targets:
    baseline = matrix[target][feature_sets[0].name]
    for fs in feature_sets[1:]:
        delta = matrix[target][fs.name].metric - baseline.metric
        matrix[target][fs.name]["delta"] = delta

# Compute summary:
sentiment_helps_most = max(targets, key=lambda t: relative_lift(matrix[t]))
summary = {
    "sentiment_helps_most": ...,
    "interpretation": "Sentiment lifts AUC by X on direction (largest gain), Y on meta-label, and reduces vol QLIKE by Z%.",
}

return { ticker, matrix, summary, elapsed_seconds }
```

## 7. API Contracts

### 7.1 `GET /models/{model_id}/accuracy`

**Query params:**
- `window_days` (int, default 30, range 1-365)
- `resolve` (bool, default true) — when true, runs `resolve_pending` before aggregating

**Response 200:**
```json
{
  "model_id": "meta_msft_eaed0724",
  "label_type": "meta_label",
  "window_days": 30,
  "resolve_run": { "checked": 12, "newly_resolved": 4, "errors": 0 },
  "stats": {
    "total_predictions": 47,
    "resolved": 35,
    "pending": 12,
    "hit_rate": 0.571,
    "avg_realized_R": 0.18,
    "best_R": 1.85,
    "worst_R": -0.94
  },
  "by_ticker": [
    { "ticker": "MSFT", "total": 35, "resolved": 25, "hit_rate": 0.571, "avg_R": 0.18 }
  ],
  "last_predictions": [
    {
      "id": "uuid", "predicted_at": "2026-04-20T15:00:00Z", "ticker": "MSFT",
      "predicted_signal": 1, "predicted_value": 0.71,
      "actual_return": 0.012, "is_correct": true, "realized_R": 0.6,
      "resolved_at": "2026-04-25T15:00:00Z"
    }
  ]
}
```

**Response 200 (model with no predictions):**
```json
{
  "model_id": "...",
  "label_type": "meta_label",
  "window_days": 30,
  "resolve_run": { "checked": 0, "newly_resolved": 0, "errors": 0 },
  "stats": { "total_predictions": 0, "resolved": 0, "pending": 0,
             "hit_rate": null, "avg_realized_R": null,
             "best_R": null, "worst_R": null },
  "by_ticker": [],
  "last_predictions": []
}
```

**Errors:**
- `404 model_not_found` — model_id not in registry
- `422` — window_days out of range

### 7.2 `POST /api/ablation/run`

**Request:**
```json
{
  "ticker": "MSFT",
  "targets": ["direction", "volatility", "meta_label"],
  "feature_sets": [
    { "name": "ta_basic", "groups": ["ta_basic"] },
    { "name": "ta_basic + sentiment", "groups": ["ta_basic", "sentiment"] }
  ],
  "horizon_days": 5,
  "model_type": "xgboost"
}
```

**Response 200:**
```json
{
  "ticker": "MSFT",
  "matrix": {
    "direction": {
      "ta_basic":            { "model_id": "...", "auc": 0.523, "f1": 0.34 },
      "ta_basic + sentiment": { "model_id": "...", "auc": 0.591, "f1": 0.42, "delta_auc": 0.068 }
    },
    "volatility": {
      "ta_basic":            { "model_id": "...", "qlike": 0.171, "r2": 0.019, "mae": 0.085 },
      "ta_basic + sentiment": { "model_id": "...", "qlike": 0.142, "r2": 0.064, "mae": 0.072,
                                 "delta_qlike": -0.029 }
    },
    "meta_label": {
      "ta_basic":            { "model_id": "...", "auc_mean": 0.619, "precision_at_50": 0.55 },
      "ta_basic + sentiment": { "model_id": "...", "auc_mean": 0.641, "precision_at_50": 0.61,
                                 "delta_auc": 0.022 }
    }
  },
  "summary": {
    "sentiment_helps_most": "direction",
    "interpretation": "Sentiment lifts AUC by 6.8 points on direction (largest gain), 2.2 points on meta-label, and reduces vol QLIKE by 17%. Direction benefits most because rule-based features under-represent news-driven moves."
  },
  "feature_sets_used": [...],
  "model_type": "xgboost",
  "horizon_days": 5,
  "elapsed_seconds": 28.4
}
```

**Errors:**
- `400 unknown_feature_set` — feature_set.groups contains a name not in `get_feature_builders` registry
- `400 insufficient_data` — any cell training fails for data reasons
- `422` — invalid horizon_days, empty targets, malformed feature_sets
- `500 ablation_failed` — wraps any unhandled training error with cell context

## 8. Error Handling

| Scenario | Behavior |
|---|---|
| Predict service can't connect to prediction_log repo | Log warning, return prediction normally (non-blocking write) |
| Yahoo provider unavailable during resolve | Skip rows, return `errors: N` in resolve_run; don't fail accuracy endpoint |
| Resolve finds NaN actual_close | Skip row; log a warning; don't mark as resolved |
| Horizon exceeds available data (recent prediction) | Skip, leave pending; expected behavior |
| Ablation cell raises ValueError (e.g. insufficient_events for meta) | Wrap in matrix as `{ error: msg, model_id: null }`, continue other cells |
| Ablation cell takes > 60s | Allow up to 90s per cell; if total > 5 min, abort with `500 ablation_timeout` |
| Empty matrix (all cells failed) | Return 200 with summary: "No cells succeeded" + cell errors |
| Frontend Leaderboard model_id missing accuracy | Render "no live data yet" placeholder in row |

## 9. Testing Strategy

**Total: 47 new tests** (27 backend + 17 frontend + 3 hooks).

**Backend unit (17)**:
- `tests/test_prediction_log_repo.py` (5): insert · list_unresolved · update_resolution · get_by_model_id · repo factory chooses Supabase when configured
- `tests/test_accuracy_service.py` (8): direction resolve · vol resolve · meta resolve · hit_rate calc · realized_R calc · NULL fields for vol regression · pending count · empty model returns null stats
- `tests/test_ablation_service.py` (4): 3×2 matrix shape · sentiment lift detected from monkeypatched train results · feature_set unknown raises · per-target metric extracted correctly

**Backend contract (10)**:
- `tests/contract/test_models_accuracy.py` (5): 200 with data · 404 model_not_found · resolve=true triggers resolution · empty model returns zero stats · window_days=365 accepted
- `tests/contract/test_ablation_run.py` (4): 200 happy · 422 horizon · 400 unknown feature_set · response includes summary + elapsed_seconds
- `tests/test_predict_log_writes.py` (4 — split across 3 files actually): predict_service writes a row · volatility_predict writes a row · signal_scoring Mode A writes a row · all 3 are non-blocking on repo failure

**Frontend (14)**:
- `__tests__/api/leaderboardQueries.test.jsx` (3): useLeaderboard / useModelAccuracy / useAblationRun
- `__tests__/pages/LeaderboardPage.test.jsx` (4): 3 tabs render · table rows · sort by metric desc · empty tab placeholder
- `__tests__/pages/AblationPage.test.jsx` (4): form submit · matrix renders 3×2 · delta colors apply · summary panel
- `__tests__/components/layout/TopNavBar.test.jsx` (2): leaderboard link in 模型 group · ablation link in 研究 group
- `__tests__/features/ablation/AblationMatrix.test.jsx` (1): standalone heatmap render with mocked data

**Regression guard**:
- All P1+P2+P3+P4 tests still green (predict services + meta_label flow not broken by added log writes)
- `pytest tests/contract/test_predict_flow.py tests/contract/test_predict_volatility.py tests/contract/test_signal_score.py -v` clean

**Live demo (P5-11)**:
- `scripts/p5_ablation_demo.py` runs ablation on AAPL, MSFT, GOOGL × 3 targets × 2 feature sets
- Output: `docs/benchmarks/p5_ablation_demo.md` — actual sentiment delta numbers per (target, ticker)
- Honest reporting tradition continues: if sentiment hurts on some target, report it

## 10. Non-Goals

- ❌ Per-prediction SHAP at log time (FE-ENH-2 separately)
- ❌ Rolling AUC over time chart (FE-ENH-3 separately)
- ❌ Cross-ticker ablation matrix (FE-ENH-5)
- ❌ Custom user-uploaded feature groups
- ❌ Ablation with 3+ feature sets in one run (call multiple ablations)
- ❌ Optuna in ablation cells (defeats fair comparison)
- ❌ Cron-based scheduled resolution (lazy is sufficient)
- ❌ Scaling to >100k predictions per model (paginate later)

## 11. Future Backlog

1. **FE-ENH-2 SHAP-per-prediction** — explain wrong predictions in Leaderboard
2. **FE-ENH-3 Rolling AUC chart** — model decay tracking
3. **FE-ENH-5 Universal sentiment lift heatmap** — cross-ticker ablation
4. **Webhook resolve** — Supabase trigger pings backend when new market data arrives
5. **Ablation 3-way comparison** — `[ta_basic, +sentiment, +sentiment+regime]`
6. **Prediction explainability panel** — Leaderboard row → SHAP for specific prediction

## 12. Success Criteria

- [ ] All 47 new tests + regression guard green
- [ ] `prediction_log` table created in Supabase via migration
- [ ] `GET /models/{id}/accuracy` returns valid response for a real model_id
- [ ] `POST /api/ablation/run` returns 200 with full matrix in <60s for 1 ticker
- [ ] Frontend `/leaderboard` loads with 3 tabs, shows real models trained on prod
- [ ] Frontend `/ablation` runs full flow on MSFT and renders matrix
- [ ] `scripts/p5_ablation_demo.py` produces real sentiment-delta report
- [ ] `master-roadmap.md` Tier 2 P5 marked ✅ with completion date
- [ ] Tag `v4-p5-complete` pushed

## 13. Dependencies

**Depends on (must exist):**
- ModelRegistry pattern (Local + Supabase split) from V3
- `training_service.train()` for direction + volatility (P1+P2)
- `meta_label_service.train_meta_label_model()` (P3)
- `get_feature_builders([...])` supporting `ta_basic` + `sentiment` groups
- Yahoo market data provider for resolve close lookups

**Unblocks:**
- FE-ENH-2 SHAP-per-prediction (needs prediction_log to attach SHAP to)
- FE-ENH-3 rolling AUC (needs resolved log history)
- Real interview narrative: "I have N weeks of live accuracy data, sentiment provides X% lift on direction"

## 14. Change Log

- 2026-04-25: Initial design approved after Harry's "都做 go" + 3-section walkthrough. Scope locked to G1 + Leaderboard + Ablation in single spec. Drop-feature ablation chosen over permutation importance. Lazy resolve chosen over cron. Default params for fair comparison.

## 15. References

- `docs/superpowers/specs/2026-04-24-p3-meta-labeling-design.md` — meta_label_service consumer
- `docs/superpowers/specs/2026-04-24-p4-signal-console-design.md` — Leaderboard adjacent UI
- `D:/obsidian vault/01-projects/quant-ai/master-roadmap.md` — P5 in Tier 2
- López de Prado *Advances in Financial ML* Ch.8 (feature importance methodology — drop-feature vs permutation)
- Existing `app/services/prediction_event_publisher.py` for the publish_prediction_event hook pattern
