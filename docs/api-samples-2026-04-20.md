# API Contract Audit — 2026-04-20

Audited endpoints used by Dashboard V2 against live backend at https://quant-ai-qzrg.onrender.com.

---

## POST /agents/technical

**Request:**
```json
{ "ticker": "AAPL", "model_id": null, "include_shap": true, "top_features": 5 }
```

**Actual response (no model trained):**
```json
{
  "success": false,
  "error": "No model available. Train and promote a model first.",
  "ticker": "AAPL",
  "model_id": null,
  "timestamp": "2026-04-20T07:23:06.175940",
  "prediction": null,
  "probability": null,
  "confidence": null,
  "summary": "",
  "signals": [],
  "top_features": [],
  "raw_features": {},
  "shap_values": {},
  "evidence_type": "technical_analysis",
  "can_index": true
}
```

**Plan expected shape (Task 4):**
- `prediction` (int or null), `probability` (object with `up`/`down` or null), `confidence` (string or null), `summary` (string), `signals` (array), `top_features` (array of `{name, contribution, direction}`)

**Assessment:** ✅ Shape matches. All expected fields present. `probability` is `null` (not `{up, down}`) when no model — frontend must handle null check.

**Note:** `probability` field is `null` (not `{up: null, down: null}`) when no model available. Component code must use `probability?.up` optional chaining.

---

## POST /agents/summary

**Request:**
```json
{ "tickers": ["AAPL", "MSFT"], "model_id": null }
```

**Actual response:**
```json
{
  "success": false,
  "error": "No model available",
  "overall_signal": null,
  "bullish_count": 0,
  "bearish_count": 0,
  "analyses": [],
  "summary": "",
  "evidence_type": "portfolio_summary"
}
```

**Assessment:** ✅ Shape matches plan expectations. `overall_signal` null when no model.

---

## POST /rag/answer

**Request:**
```json
{ "query": "What is the market outlook?", "top_k": 5 }
```

**Actual response:**
```json
{
  "query": "What is the market outlook?",
  "answer": "No relevant information found. Try indexing more documents or rephrasing your question.",
  "evidence": [],
  "confidence": 0.0
}
```

**Assessment:** ✅ Shape matches. Top-level: `query`, `answer`, `evidence`, `confidence`.

---

## GET /models/{id}

**Note:** No promoted/active models exist on live backend (models list is empty). Cannot test this endpoint with a real ID. Shape assumed from backend code: model object with `id`, `run_id`, `model_type`, `tickers`, `created_at`, `metadata` fields.

**Assessment:** ⚠️ Cannot verify shape directly — no models available. Plan uses `getModel(id)` and passes result to `AboutBlock`. Frontend must handle 404 gracefully.

---

## GET /models?status=active

**Actual response:**
```json
{ "models": [], "total": 0 }
```

**Assessment:** ✅ Returns `{ models: [], total: N }` object — NOT a plain array. Plan's `getModelsForTicker` already normalizes this: `const models = allActive.models ?? allActive`. No mismatch.

---

## GET /data/market?ticker=AAPL&period=6mo

**Actual response (abbreviated):**
```json
[
  { "ticker": "AAPL", "date": "2026-04-17", "open": 266.96, "high": 272.30, "low": 266.72, "close": 270.23, "volume": 61314800 },
  ...
]
```

**Fields:** `ticker`, `date` (YYYY-MM-DD string), `open`, `high`, `low`, `close` (floats), `volume` (int).

**Assessment:** ✅ Flat array, fields match what ChartSection needs. Date is YYYY-MM-DD which Lightweight Charts accepts directly as `time`. No normalization needed.

**Historical note:** Prior incident — this endpoint returned flat array while frontend expected `{rows:[...]}`. That normalization was already applied in client.js. Confirmed flat array is the correct shape.

---

## GET /data/sentiment?ticker=AAPL&days=30

**Actual response (abbreviated):**
```json
[
  {
    "ticker": "AAPL",
    "date": "2026-04-01",
    "source": "mock_social",
    "sentiment_score": -0.0756,
    "volume": 146,
    "bullish_count": 43,
    "bearish_count": 43,
    "neutral_count": 60
  },
  ...
]
```

**Fields:** `ticker`, `date`, `source`, `sentiment_score` (float -1..+1), `volume`, `bullish_count`, `bearish_count`, `neutral_count`.

**Assessment:** ✅ Flat array. Source is `"mock_social"` (mock data). `sentiment_score` is the primary signal field.

---

## Summary — Mismatches vs Plan

| Endpoint | Status | Notes |
|----------|--------|-------|
| POST /agents/technical | ✅ Match | `probability` is null (not `{up,down}`) when no model — use optional chaining |
| POST /agents/summary | ✅ Match | Empty `analyses[]` when no model |
| POST /rag/answer | ✅ Match | `answer`/`evidence`/`confidence` all present |
| GET /models/{id} | ⚠️ Unverified | No models to test against |
| GET /models?status=active | ✅ Match | `{models:[], total:N}` — plan already normalizes |
| GET /data/market | ✅ Match | Flat array, YYYY-MM-DD date format |
| GET /data/sentiment | ✅ Match | Flat array, `sentiment_score` field |

**No blocking mismatches found.** All shapes either match the plan or are already normalized in client.js.
