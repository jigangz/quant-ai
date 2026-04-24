# P4 Signal Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the frontend Signal Console experience + one new aggregation backend endpoint + Paper Trading UI gate + Strategy badges + Dashboard sparkline + P3 carryover fixes. Close V4 Pivot Gate 1.

**Architecture:** One new page (`/signal-console`) with 3 components (TickerPicker / StrategyMatrix / SignalDetail). One reusable `MetaLabelCoverageBadge` (Strategy page + Signal Console). Paper Trading modal adds opt-in meta-label section. Dashboard `VolatilityCard` gets conditional sparkline. Backend: one new `/api/meta-label/coverage` endpoint.

**Tech Stack:** React 18 + Vite + TanStack Query v5 + React Router v6 + Tailwind + Vitest + @testing-library/react. Backend: FastAPI + Pydantic v2 + pytest.

**Spec:** [`docs/superpowers/specs/2026-04-24-p4-signal-console-design.md`](../specs/2026-04-24-p4-signal-console-design.md)

**Branch:** direct-to-main (P2 + P3 precedent). Ralph Loop handles per-task commits.

---

## File Structure

### New files (frontend)
- `quant-ai-ui/src/pages/SignalConsolePage.jsx` — page container
- `quant-ai-ui/src/features/signal-console/TickerPicker.jsx` — watchlist-scoped ticker selector
- `quant-ai-ui/src/features/signal-console/StrategyMatrix.jsx` — ticker × strategy cell grid
- `quant-ai-ui/src/features/signal-console/SignalDetail.jsx` — right panel for selected cell
- `quant-ai-ui/src/features/signal-console/MetaSparkline.jsx` — 7-day sparkline (used from VolatilityCard)
- `quant-ai-ui/src/components/MetaLabelCoverageBadge.jsx` — reusable badge
- `quant-ai-ui/src/api/signalQueries.js` — TanStack hooks for P3/P4 endpoints

### Modified files (frontend)
- `quant-ai-ui/src/api/client.js` — add `getMetaLabelModels`, `postSignalScore`, `getMetaCoverage`, `postMetaLabelTrain`
- `quant-ai-ui/src/features/strategy/StrategyCard.jsx` — embed badge
- `quant-ai-ui/src/features/dashboard/VolatilityCard.jsx` — conditional `<MetaSparkline />`
- `quant-ai-ui/src/pages/TradingPage.jsx` — meta-label section in order form (or its sub-component)
- `quant-ai-ui/src/App.jsx` — add `/signal-console` route
- `quant-ai-ui/src/components/layout/TopNavBar.jsx` — add Signal Console nav link

### New files (backend + scripts)
- `tests/contract/test_meta_coverage.py` — 5 tests for new endpoint
- `scripts/p4_aapl_optuna_rescue.py` — Optuna rescue script
- `docs/benchmarks/p4_aapl_optuna.md` — addendum report (or failure doc)

### Modified files (backend)
- `app/services/signal_scoring_service.py` — add `compute_coverage()` function
- `app/api/signal.py` — add `GET /api/meta-label/coverage` endpoint
- `app/main.py` — version string bump

### New test files (frontend)
- `quant-ai-ui/__tests__/components/MetaLabelCoverageBadge.test.jsx` (3)
- `quant-ai-ui/__tests__/api/signalQueries.test.jsx` (3)
- `quant-ai-ui/__tests__/features/signal-console/TickerPicker.test.jsx` (3)
- `quant-ai-ui/__tests__/features/signal-console/StrategyMatrix.test.jsx` (4)
- `quant-ai-ui/__tests__/features/signal-console/SignalDetail.test.jsx` (3)
- `quant-ai-ui/__tests__/features/signal-console/MetaSparkline.test.jsx` (2)
- `quant-ai-ui/__tests__/features/strategy/StrategyCard.test.jsx` (2 — additions)
- `quant-ai-ui/__tests__/pages/SignalConsolePage.test.jsx` (2)
- `quant-ai-ui/__tests__/pages/TradingPage.meta.test.jsx` (4)

### Shared test helpers
- `quant-ai-ui/__tests__/_helpers/queryWrapper.jsx` — shared TanStack QueryClientProvider wrapper (one-time setup to reduce per-test boilerplate)

---

## Task 0: P3 Carryover — Version Bump + Dockerfile Verification

**Files:**
- Modify: `app/main.py` (version field)
- No new test (verified via prod smoke in GATE)

- [ ] **Step 0.1: Find version string**

```bash
grep -n "2.1.0\|version" C:/Users/zjg09/projects/quant-ai/app/main.py | head -5
```

Locate the root handler that returns `{"name": "Quant AI Backend", "version": "2.0.0"}` and the health endpoint that returns `"version": "2.1.0"`.

- [ ] **Step 0.2: Bump versions**

In `app/main.py` find the root handler and health response. Update both version strings to `"2.4.0"` (signals V4 P4 ship).

```python
# root handler
@app.get("/")
def root():
    return {"name": "Quant AI Backend", "version": "2.4.0", "docs": "/docs", "health": "/health"}

# /health returns include:
"version": "2.4.0",
```

- [ ] **Step 0.3: Run smoke locally**

```bash
cd C:/Users/zjg09/projects/quant-ai && python -c "from app.main import app; print('ok')"
```

Expected: prints `ok` with no import error.

- [ ] **Step 0.4: Commit**

```bash
git add app/main.py
git commit -m "fix(p4): version bump 2.1.0 -> 2.4.0 (P3 carryover)"
```

---

## Task 1: Backend `GET /api/meta-label/coverage`

**Files:**
- Modify: `app/services/signal_scoring_service.py` (add `compute_coverage`)
- Modify: `app/api/signal.py` (add endpoint)
- Create: `tests/contract/test_meta_coverage.py`

- [ ] **Step 1.1: Write failing tests**

Create `tests/contract/test_meta_coverage.py`:

```python
"""Contract tests for GET /api/meta-label/coverage (V4 Phase 4)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    from app.main import app
    from app.services import signal_scoring_service

    FAKE_RECORDS = [
        {"model_id": "meta_msft_a", "extras": {"meta_label": {
            "primary": {"source": "strategy", "strategy_name": "rsi_strategy"},
            "cv": {"metrics": {"auc_mean": 0.619}},
            "event_count": 483,
        }}, "metadata": {"ticker": "MSFT", "label_type": "meta_label"}},
        {"model_id": "meta_googl_b", "extras": {"meta_label": {
            "primary": {"source": "strategy", "strategy_name": "rsi_strategy"},
            "cv": {"metrics": {"auc_mean": 0.607}},
            "event_count": 486,
        }}, "metadata": {"ticker": "GOOGL", "label_type": "meta_label"}},
        {"model_id": "meta_aapl_c", "extras": {"meta_label": {
            "primary": {"source": "strategy", "strategy_name": "rsi_strategy"},
            "cv": {"metrics": {"auc_mean": 0.420}},
            "event_count": 492,
        }}, "metadata": {"ticker": "AAPL", "label_type": "meta_label"}},
        {"model_id": "meta_msft_ma", "extras": {"meta_label": {
            "primary": {"source": "strategy", "strategy_name": "ma_cross"},
            "cv": {"metrics": {"auc_mean": 0.55}},
            "event_count": 200,
        }}, "metadata": {"ticker": "MSFT", "label_type": "meta_label"}},
    ]

    def fake_list_meta_records():
        return FAKE_RECORDS

    monkeypatch.setattr(
        signal_scoring_service, "_list_meta_records", fake_list_meta_records
    )
    return TestClient(app)


def test_200_rsi_strategy_three_models(client):
    resp = client.get("/api/meta-label/coverage?strategy=rsi_strategy")
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy_name"] == "rsi_strategy"
    assert body["count"] == 3
    assert body["max_auc"] == pytest.approx(0.619, abs=1e-3)
    assert abs(body["avg_auc"] - (0.619 + 0.607 + 0.420) / 3) < 1e-3
    assert set(body["tickers"]) == {"MSFT", "GOOGL", "AAPL"}
    assert len(body["models"]) == 3


def test_200_zero_coverage(client):
    resp = client.get("/api/meta-label/coverage?strategy=bollinger_breakout")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["max_auc"] is None
    assert body["avg_auc"] is None
    assert body["tickers"] == []
    assert body["models"] == []


def test_404_unknown_strategy(client):
    resp = client.get("/api/meta-label/coverage?strategy=not_real")
    assert resp.status_code == 404


def test_aggregation_math_correct(client):
    resp = client.get("/api/meta-label/coverage?strategy=ma_cross")
    body = resp.json()
    assert body["count"] == 1
    assert body["max_auc"] == pytest.approx(0.55, abs=1e-3)
    assert body["avg_auc"] == pytest.approx(0.55, abs=1e-3)


def test_malformed_record_is_skipped(client, monkeypatch):
    from app.services import signal_scoring_service

    def broken_records():
        return [
            {"model_id": "broken", "extras": {}, "metadata": {"ticker": "X", "label_type": "meta_label"}},
            {"model_id": "ok", "extras": {"meta_label": {
                "primary": {"source": "strategy", "strategy_name": "rsi_strategy"},
                "cv": {"metrics": {"auc_mean": 0.6}},
            }}, "metadata": {"ticker": "MSFT", "label_type": "meta_label"}},
        ]
    monkeypatch.setattr(
        signal_scoring_service, "_list_meta_records", broken_records
    )
    resp = client.get("/api/meta-label/coverage?strategy=rsi_strategy")
    assert resp.status_code == 200
    # The broken record is skipped; 1 valid model counted
    assert resp.json()["count"] == 1
```

- [ ] **Step 1.2: Run tests — verify they fail**

```bash
cd C:/Users/zjg09/projects/quant-ai
pytest tests/contract/test_meta_coverage.py -v
```

Expected: fail with `AttributeError: module has no attribute '_list_meta_records'` or 404 on the endpoint.

- [ ] **Step 1.3: Add `compute_coverage` to signal_scoring_service**

Append to `app/services/signal_scoring_service.py`:

```python
KNOWN_STRATEGIES = {"ma_cross", "rsi_strategy", "bollinger_breakout", "sentiment_driven"}


def _list_meta_records() -> list[dict]:
    """Return all registered meta-label model records. Wrapped for monkeypatch in tests."""
    from app.services.model_cache import get_model_cache
    cache = get_model_cache()
    try:
        all_records = cache.list_all(label_type="meta_label")
    except AttributeError:
        # Fallback path: older ModelCache may not expose list_all
        all_records = []
        for model_id in getattr(cache, "_index", {}).keys():
            info = cache.load(model_id)
            if info and info.metadata.get("label_type") == "meta_label":
                all_records.append({
                    "model_id": model_id,
                    "metadata": info.metadata,
                    "extras": info.extras or {},
                })
    return all_records


def compute_coverage(strategy_name: str) -> dict:
    """Aggregate meta-label coverage for a given primary strategy."""
    if strategy_name not in KNOWN_STRATEGIES:
        raise ValueError(f"strategy_not_found:{strategy_name}")

    records = _list_meta_records()
    models = []
    for r in records:
        extras = r.get("extras", {}) or {}
        meta_cfg = extras.get("meta_label") or {}
        primary_cfg = meta_cfg.get("primary") or {}
        if primary_cfg.get("source") != "strategy":
            continue
        if primary_cfg.get("strategy_name") != strategy_name:
            continue
        auc = (
            meta_cfg.get("cv", {})
            .get("metrics", {})
            .get("auc_mean")
        )
        if auc is None:
            continue
        ticker = r.get("metadata", {}).get("ticker") or "UNKNOWN"
        models.append({
            "model_id": r["model_id"],
            "ticker": ticker,
            "auc_mean": float(auc),
            "event_count": int(meta_cfg.get("event_count", 0)),
        })

    if not models:
        return {
            "strategy_name": strategy_name,
            "count": 0, "max_auc": None, "avg_auc": None,
            "tickers": [], "models": [],
        }

    aucs = [m["auc_mean"] for m in models]
    tickers = sorted({m["ticker"] for m in models})
    return {
        "strategy_name": strategy_name,
        "count": len(models),
        "max_auc": max(aucs),
        "avg_auc": sum(aucs) / len(aucs),
        "tickers": tickers,
        "models": models,
    }
```

- [ ] **Step 1.4: Add endpoint to `app/api/signal.py`**

Append at the bottom of `app/api/signal.py`:

```python
# =====================================
# GET /api/meta-label/coverage
# =====================================

@router.get("/api/meta-label/coverage")
def meta_label_coverage(strategy: str):
    try:
        return signal_scoring_service.compute_coverage(strategy)
    except ValueError as e:
        msg = str(e)
        if msg.startswith("strategy_not_found"):
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
```

- [ ] **Step 1.5: Run tests**

```bash
pytest tests/contract/test_meta_coverage.py -v
```

Expected: `5 passed`.

- [ ] **Step 1.6: Commit**

```bash
git add app/services/signal_scoring_service.py app/api/signal.py tests/contract/test_meta_coverage.py
git commit -m "feat(p4): GET /api/meta-label/coverage endpoint + compute_coverage (5 tests)"
```

---

## Task 2: Frontend API Client + Test Helper

**Files:**
- Modify: `quant-ai-ui/src/api/client.js`
- Create: `quant-ai-ui/src/api/signalQueries.js`
- Create: `quant-ai-ui/__tests__/_helpers/queryWrapper.jsx`
- Create: `quant-ai-ui/__tests__/api/signalQueries.test.jsx`

- [ ] **Step 2.1: Create shared test wrapper**

Create `quant-ai-ui/__tests__/_helpers/queryWrapper.jsx`:

```jsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

export function makeQueryWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } },
  });
  return ({ children }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}
```

- [ ] **Step 2.2: Write failing tests for signalQueries**

Create `quant-ai-ui/__tests__/api/signalQueries.test.jsx`:

```jsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { makeQueryWrapper } from "../_helpers/queryWrapper";

vi.mock("@/api/client", () => ({
  getMetaLabelModels: vi.fn(),
  getMetaCoverage: vi.fn(),
  postSignalScore: vi.fn(),
  postMetaLabelTrain: vi.fn(),
}));

import * as client from "@/api/client";
import { useMetaLabelModels, useMetaCoverage, useSignalScorePreview } from "@/api/signalQueries";

beforeEach(() => { vi.clearAllMocks(); });

describe("useMetaLabelModels", () => {
  it("calls getMetaLabelModels with ticker and returns data", async () => {
    client.getMetaLabelModels.mockResolvedValue([
      { model_id: "meta_a", extras: { meta_label: { primary: { strategy_name: "rsi_strategy" } } } },
    ]);
    const { result } = renderHook(() => useMetaLabelModels("AAPL"), { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(result.current.data).toBeDefined());
    expect(client.getMetaLabelModels).toHaveBeenCalledWith("AAPL");
    expect(result.current.data).toHaveLength(1);
  });
});

describe("useMetaCoverage", () => {
  it("calls getMetaCoverage with strategy name and returns data", async () => {
    client.getMetaCoverage.mockResolvedValue({ count: 3, max_auc: 0.62, avg_auc: 0.55, tickers: ["MSFT"] });
    const { result } = renderHook(() => useMetaCoverage("rsi_strategy"), { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(result.current.data).toBeDefined());
    expect(result.current.data.count).toBe(3);
  });
});

describe("useSignalScorePreview", () => {
  it("exposes a mutate function that calls postSignalScore", async () => {
    client.postSignalScore.mockResolvedValue({ triggered: true, reliability_score: 0.71, signal: 1 });
    const { result } = renderHook(() => useSignalScorePreview(), { wrapper: makeQueryWrapper() });
    await result.current.mutateAsync({ ticker: "AAPL", meta_model_id: "meta_a", signal: 1 });
    expect(client.postSignalScore).toHaveBeenCalledWith({ ticker: "AAPL", meta_model_id: "meta_a", signal: 1 });
  });
});
```

- [ ] **Step 2.3: Run tests — fail**

```bash
cd C:/Users/zjg09/projects/quant-ai/quant-ai-ui
npm run test -- --run __tests__/api/signalQueries.test.jsx
```

Expected: fail with `Cannot find module '@/api/signalQueries'`.

- [ ] **Step 2.4: Extend api/client.js**

Append to `quant-ai-ui/src/api/client.js`:

```javascript
// ===================================
// V4 Phase 4 — Meta-Labeling (P3/P4)
// ===================================

const _request = async (path, options = {}) => {
  // Local alias to existing `request` helper (already defined at top of file).
  // If you see this line duplicated at runtime, remove — it's defensive.
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json();
};

/** GET /models?label_type=meta_label&ticker=X */
export function getMetaLabelModels(ticker) {
  const qs = ticker ? `&ticker=${encodeURIComponent(ticker)}` : "";
  return _request(`/models?label_type=meta_label${qs}`).then((body) => body.models || body || []);
}

/** GET /api/meta-label/coverage?strategy=X */
export function getMetaCoverage(strategyName) {
  return _request(`/api/meta-label/coverage?strategy=${encodeURIComponent(strategyName)}`);
}

/** POST /api/signal-score */
export function postSignalScore(payload) {
  return _request(`/api/signal-score`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/** POST /api/meta-label/train */
export function postMetaLabelTrain(payload) {
  return _request(`/api/meta-label/train`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
```

**Note:** If `request()` is already exported or easily referenced, replace `_request` with direct calls to `request()`. The duplicated helper is defensive in case of scoping issues — prefer the existing pattern.

- [ ] **Step 2.5: Create signalQueries.js**

Create `quant-ai-ui/src/api/signalQueries.js`:

```javascript
import { useQuery, useMutation } from "@tanstack/react-query";
import * as api from "./client";

/** Meta-label models for a specific ticker (list + filter via P2 G3). */
export function useMetaLabelModels(ticker, opts = {}) {
  return useQuery({
    queryKey: ["meta-label-models", ticker],
    queryFn: () => api.getMetaLabelModels(ticker),
    enabled: !!ticker,
    staleTime: 30_000,
    ...opts,
  });
}

/** Coverage for a single strategy (Strategy card badge + Signal Console). */
export function useMetaCoverage(strategyName, opts = {}) {
  return useQuery({
    queryKey: ["meta-coverage", strategyName],
    queryFn: () => api.getMetaCoverage(strategyName),
    enabled: !!strategyName,
    staleTime: 60_000,
    retry: (failureCount, error) => {
      if (String(error.message).includes("API error 404")) return false;
      return failureCount < 2;
    },
    ...opts,
  });
}

/** Mutation — manual score preview. */
export function useSignalScorePreview() {
  return useMutation({
    mutationFn: (payload) => api.postSignalScore(payload),
  });
}

/** Mutation — train a new meta-label model from Signal Console CTA. */
export function useMetaLabelTrain() {
  return useMutation({
    mutationFn: (payload) => api.postMetaLabelTrain(payload),
  });
}
```

- [ ] **Step 2.6: Run tests**

```bash
npm run test -- --run __tests__/api/signalQueries.test.jsx
```

Expected: `3 passed`.

- [ ] **Step 2.7: Commit**

```bash
git add quant-ai-ui/src/api/client.js quant-ai-ui/src/api/signalQueries.js quant-ai-ui/__tests__/_helpers/queryWrapper.jsx quant-ai-ui/__tests__/api/signalQueries.test.jsx
git commit -m "feat(p4): api client + signalQueries hooks for meta-label endpoints (3 tests)"
```

---

## Task 3: `MetaLabelCoverageBadge` Component

**Files:**
- Create: `quant-ai-ui/src/components/MetaLabelCoverageBadge.jsx`
- Create: `quant-ai-ui/__tests__/components/MetaLabelCoverageBadge.test.jsx`

- [ ] **Step 3.1: Write failing tests**

Create `quant-ai-ui/__tests__/components/MetaLabelCoverageBadge.test.jsx`:

```jsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { makeQueryWrapper } from "../_helpers/queryWrapper";
import MetaLabelCoverageBadge from "@/components/MetaLabelCoverageBadge";

vi.mock("@/api/client", () => ({
  getMetaCoverage: vi.fn(),
  getMetaLabelModels: vi.fn(), postSignalScore: vi.fn(), postMetaLabelTrain: vi.fn(),
}));
import * as client from "@/api/client";

beforeEach(() => vi.clearAllMocks());

describe("MetaLabelCoverageBadge", () => {
  it("renders count and max AUC when coverage exists", async () => {
    client.getMetaCoverage.mockResolvedValue({
      count: 3, max_auc: 0.619, avg_auc: 0.549, tickers: ["MSFT", "GOOGL", "AAPL"],
    });
    render(<MetaLabelCoverageBadge strategyName="rsi_strategy" />, { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(screen.getByText(/Meta/i)).toBeInTheDocument());
    expect(screen.getByText(/Meta/i).textContent).toMatch(/3/);
    expect(screen.getByText(/Meta/i).textContent).toMatch(/0\.62/);
  });

  it("renders nothing when coverage is zero (count=0 or 404)", async () => {
    client.getMetaCoverage.mockResolvedValue({
      count: 0, max_auc: null, avg_auc: null, tickers: [],
    });
    const { container } = render(
      <MetaLabelCoverageBadge strategyName="rsi_strategy" />,
      { wrapper: makeQueryWrapper() },
    );
    await waitFor(() => {
      // Badge should not be in DOM after query resolves
      expect(container.textContent.trim()).toBe("");
    });
  });

  it("applies warning tint when avg_auc < 0.5", async () => {
    client.getMetaCoverage.mockResolvedValue({
      count: 1, max_auc: 0.42, avg_auc: 0.42, tickers: ["AAPL"],
    });
    const { container } = render(
      <MetaLabelCoverageBadge strategyName="rsi_strategy" />,
      { wrapper: makeQueryWrapper() },
    );
    await waitFor(() => expect(container.querySelector("[data-variant='warn']")).toBeTruthy());
  });
});
```

- [ ] **Step 3.2: Run tests — fail**

```bash
npm run test -- --run __tests__/components/MetaLabelCoverageBadge.test.jsx
```

Expected: fail on `Cannot find module '@/components/MetaLabelCoverageBadge'`.

- [ ] **Step 3.3: Create component**

Create `quant-ai-ui/src/components/MetaLabelCoverageBadge.jsx`:

```jsx
import { useNavigate } from "react-router-dom";
import { useMetaCoverage } from "@/api/signalQueries";

export default function MetaLabelCoverageBadge({ strategyName }) {
  const { data, isLoading, isError } = useMetaCoverage(strategyName);
  const navigate = useNavigate();

  if (isLoading || isError) return null;
  if (!data || data.count === 0) return null;

  const avg = data.avg_auc ?? 0;
  const variant = avg >= 0.60 ? "good" : avg >= 0.50 ? "neutral" : "warn";
  const warnMark = variant === "warn" ? " ⚠" : "";

  const onClick = () => navigate(`/signal-console?strategy=${encodeURIComponent(strategyName)}`);

  const bg = variant === "good" ? "bg-emerald-500/15 text-emerald-400"
    : variant === "warn" ? "bg-amber-500/15 text-amber-400"
    : "bg-slate-500/15 text-slate-300";

  return (
    <button
      type="button"
      onClick={onClick}
      data-variant={variant}
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium ${bg}`}
      title={`${data.count} meta-model${data.count > 1 ? "s" : ""} · avg AUC ${avg.toFixed(2)} · tickers: ${data.tickers.join(", ")}`}
    >
      Meta ✓ {data.count} · AUC {data.max_auc.toFixed(2)}{warnMark}
    </button>
  );
}
```

- [ ] **Step 3.4: Run tests**

```bash
npm run test -- --run __tests__/components/MetaLabelCoverageBadge.test.jsx
```

Expected: `3 passed`.

- [ ] **Step 3.5: Commit**

```bash
git add quant-ai-ui/src/components/MetaLabelCoverageBadge.jsx quant-ai-ui/__tests__/components/MetaLabelCoverageBadge.test.jsx
git commit -m "feat(p4): MetaLabelCoverageBadge component (3 tests)"
```

---

## Task 4: TickerPicker Component

**Files:**
- Create: `quant-ai-ui/src/features/signal-console/TickerPicker.jsx`
- Create: `quant-ai-ui/__tests__/features/signal-console/TickerPicker.test.jsx`

- [ ] **Step 4.1: Write failing tests**

Create `quant-ai-ui/__tests__/features/signal-console/TickerPicker.test.jsx`:

```jsx
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import TickerPicker from "@/features/signal-console/TickerPicker";

beforeEach(() => {
  localStorage.clear();
  localStorage.setItem(
    "quant-ai:watchlist",
    JSON.stringify(["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"])
  );
});

describe("TickerPicker", () => {
  it("loads tickers from localStorage on mount", () => {
    render(<TickerPicker selected={["AAPL"]} onChange={() => {}} />);
    expect(screen.getByText("AAPL")).toBeInTheDocument();
    expect(screen.getByText("MSFT")).toBeInTheDocument();
  });

  it("toggles a ticker when clicked (multi-select)", () => {
    const onChange = vi.fn();
    render(<TickerPicker selected={["AAPL"]} onChange={onChange} />);
    fireEvent.click(screen.getByText("MSFT"));
    expect(onChange).toHaveBeenCalledWith(["AAPL", "MSFT"]);
  });

  it("caps selection at 10 tickers", () => {
    const big = Array.from({ length: 12 }, (_, i) => `T${i}`);
    localStorage.setItem("quant-ai:watchlist", JSON.stringify(big));
    const onChange = vi.fn();
    render(<TickerPicker selected={big.slice(0, 10)} onChange={onChange} />);
    fireEvent.click(screen.getByText("T10"));
    // Already at cap, onChange should NOT be called with 11 items
    expect(onChange).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 4.2: Run tests — fail**

```bash
npm run test -- --run __tests__/features/signal-console/TickerPicker.test.jsx
```

- [ ] **Step 4.3: Create component**

Create `quant-ai-ui/src/features/signal-console/TickerPicker.jsx`:

```jsx
import { useEffect, useState } from "react";

const STORAGE_KEY = "quant-ai:watchlist";
const MAX_SELECTED = 10;

export default function TickerPicker({ selected = [], onChange }) {
  const [available, setAvailable] = useState([]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      setAvailable(raw ? JSON.parse(raw) : []);
    } catch {
      setAvailable([]);
    }
  }, []);

  const toggle = (t) => {
    if (selected.includes(t)) {
      onChange(selected.filter((x) => x !== t));
    } else {
      if (selected.length >= MAX_SELECTED) return;
      onChange([...selected, t]);
    }
  };

  return (
    <div className="flex flex-wrap gap-2 p-3 bg-slate-900/40 rounded-lg">
      <div className="text-xs text-slate-400 mr-2 self-center">Watchlist:</div>
      {available.map((t) => (
        <button
          key={t}
          type="button"
          onClick={() => toggle(t)}
          className={`px-2 py-1 text-xs rounded border ${
            selected.includes(t)
              ? "bg-emerald-600/20 border-emerald-600/50 text-emerald-300"
              : "bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-200"
          }`}
        >
          {t}
        </button>
      ))}
      <div className="text-xs text-slate-500 ml-auto self-center">
        {selected.length}/{MAX_SELECTED} selected
      </div>
    </div>
  );
}
```

- [ ] **Step 4.4: Run tests**

```bash
npm run test -- --run __tests__/features/signal-console/TickerPicker.test.jsx
```

Expected: `3 passed`.

- [ ] **Step 4.5: Commit**

```bash
git add quant-ai-ui/src/features/signal-console/TickerPicker.jsx quant-ai-ui/__tests__/features/signal-console/TickerPicker.test.jsx
git commit -m "feat(p4): TickerPicker with 10-ticker cap + localStorage (3 tests)"
```

---

## Task 5: StrategyMatrix Component

**Files:**
- Create: `quant-ai-ui/src/features/signal-console/StrategyMatrix.jsx`
- Create: `quant-ai-ui/__tests__/features/signal-console/StrategyMatrix.test.jsx`

- [ ] **Step 5.1: Write failing tests**

Create `quant-ai-ui/__tests__/features/signal-console/StrategyMatrix.test.jsx`:

```jsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { makeQueryWrapper } from "../../_helpers/queryWrapper";
import StrategyMatrix from "@/features/signal-console/StrategyMatrix";

vi.mock("@/api/client", () => ({
  getMetaLabelModels: vi.fn(),
  getMetaCoverage: vi.fn(), postSignalScore: vi.fn(), postMetaLabelTrain: vi.fn(),
}));
import * as client from "@/api/client";

const MODEL_AAPL_RSI = {
  model_id: "meta_aapl_c", metadata: { ticker: "AAPL", label_type: "meta_label" },
  extras: { meta_label: {
    primary: { source: "strategy", strategy_name: "rsi_strategy" },
    cv: { metrics: { auc_mean: 0.42, precision_at_50: 0.42, expected_R_when_trade: -0.05 } },
    barrier: { tp_k: 2, sl_k: 1 }, event_count: 492,
  }},
};
const MODEL_MSFT_RSI = {
  model_id: "meta_msft_a", metadata: { ticker: "MSFT", label_type: "meta_label" },
  extras: { meta_label: {
    primary: { source: "strategy", strategy_name: "rsi_strategy" },
    cv: { metrics: { auc_mean: 0.62, precision_at_50: 0.55, expected_R_when_trade: 0.02 } },
    barrier: { tp_k: 2, sl_k: 1 }, event_count: 483,
  }},
};

beforeEach(() => {
  vi.clearAllMocks();
  client.getMetaLabelModels.mockImplementation((ticker) => {
    if (ticker === "AAPL") return Promise.resolve([MODEL_AAPL_RSI]);
    if (ticker === "MSFT") return Promise.resolve([MODEL_MSFT_RSI]);
    return Promise.resolve([]);
  });
});

describe("StrategyMatrix", () => {
  it("renders a row per ticker with 4 strategy columns", async () => {
    render(<StrategyMatrix tickers={["AAPL", "MSFT"]} onSelect={() => {}} />, { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(screen.getByText("AAPL")).toBeInTheDocument());
    ["ma_cross", "rsi_strategy", "bollinger_breakout", "sentiment_driven"].forEach((s) => {
      expect(screen.getByText(s)).toBeInTheDocument();
    });
    expect(screen.getByText("MSFT")).toBeInTheDocument();
  });

  it("shows 'Train' CTA for cells without a model", async () => {
    render(<StrategyMatrix tickers={["AAPL"]} onSelect={() => {}} />, { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(screen.getAllByText(/Train/i).length).toBeGreaterThan(0));
  });

  it("calls onSelect with cell details when a cell is clicked", async () => {
    const onSelect = vi.fn();
    render(<StrategyMatrix tickers={["MSFT"]} onSelect={onSelect} />, { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(screen.getByText(/0\.62/)).toBeInTheDocument());
    fireEvent.click(screen.getByText(/0\.62/).closest("[data-cell]"));
    expect(onSelect).toHaveBeenCalledWith(
      expect.objectContaining({ ticker: "MSFT", strategy: "rsi_strategy", model_id: "meta_msft_a" })
    );
  });

  it("applies warn data-variant for cells with AUC < 0.5", async () => {
    const { container } = render(<StrategyMatrix tickers={["AAPL"]} onSelect={() => {}} />, { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(container.querySelector("[data-variant='warn']")).toBeTruthy());
  });
});
```

- [ ] **Step 5.2: Run tests — fail**

```bash
npm run test -- --run __tests__/features/signal-console/StrategyMatrix.test.jsx
```

- [ ] **Step 5.3: Create component**

Create `quant-ai-ui/src/features/signal-console/StrategyMatrix.jsx`:

```jsx
import { useMetaLabelModels } from "@/api/signalQueries";

const STRATEGIES = ["ma_cross", "rsi_strategy", "bollinger_breakout", "sentiment_driven"];

function Cell({ ticker, strategy, model, onSelect, onTrain }) {
  if (!model) {
    return (
      <td className="px-2 py-2">
        <button
          type="button"
          onClick={() => onTrain?.({ ticker, strategy })}
          className="w-full text-[10px] text-slate-500 hover:text-emerald-300 border border-dashed border-slate-700 rounded px-1 py-2"
        >
          — Train meta
        </button>
      </td>
    );
  }
  const auc = model.extras.meta_label.cv.metrics.auc_mean;
  const er = model.extras.meta_label.cv.metrics.expected_R_when_trade;
  const variant = auc < 0.5 ? "warn" : auc >= 0.60 ? "good" : "neutral";
  const bg = variant === "warn" ? "bg-amber-500/10" : variant === "good" ? "bg-emerald-500/10" : "bg-slate-700/30";
  return (
    <td className="px-2 py-2">
      <button
        type="button"
        data-cell
        data-variant={variant}
        onClick={() => onSelect({ ticker, strategy, model_id: model.model_id })}
        className={`w-full text-xs rounded px-2 py-2 text-left ${bg} hover:ring-1 hover:ring-emerald-500/50`}
      >
        <div className="font-medium">AUC {auc.toFixed(2)}{variant === "warn" ? " ⚠" : ""}</div>
        <div className="text-[10px] text-slate-400">E[R] {er >= 0 ? "+" : ""}{er.toFixed(2)}</div>
      </button>
    </td>
  );
}

function TickerRow({ ticker, onSelect, onTrain }) {
  const { data = [], isLoading } = useMetaLabelModels(ticker);
  const byStrategy = {};
  for (const m of data) {
    const s = m?.extras?.meta_label?.primary?.strategy_name;
    if (s) byStrategy[s] = m;
  }
  return (
    <tr>
      <td className="px-3 py-2 text-sm font-medium text-slate-200 border-r border-slate-800">{ticker}</td>
      {STRATEGIES.map((s) => (
        <Cell
          key={s}
          ticker={ticker}
          strategy={s}
          model={byStrategy[s]}
          onSelect={onSelect}
          onTrain={onTrain}
        />
      ))}
    </tr>
  );
}

export default function StrategyMatrix({ tickers, onSelect, onTrain }) {
  if (!tickers || tickers.length === 0) {
    return (
      <div className="p-6 text-sm text-slate-500 text-center bg-slate-900/40 rounded-lg">
        Select one or more tickers from the watchlist above.
      </div>
    );
  }
  return (
    <div className="overflow-x-auto bg-slate-900/40 rounded-lg">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-[10px] uppercase tracking-wide text-slate-400 border-b border-slate-800">
            <th className="px-3 py-2 text-left">Ticker</th>
            {STRATEGIES.map((s) => <th key={s} className="px-2 py-2 text-left">{s}</th>)}
          </tr>
        </thead>
        <tbody>
          {tickers.map((t) => (
            <TickerRow key={t} ticker={t} onSelect={onSelect} onTrain={onTrain} />
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 5.4: Run tests**

```bash
npm run test -- --run __tests__/features/signal-console/StrategyMatrix.test.jsx
```

Expected: `4 passed`.

- [ ] **Step 5.5: Commit**

```bash
git add quant-ai-ui/src/features/signal-console/StrategyMatrix.jsx quant-ai-ui/__tests__/features/signal-console/StrategyMatrix.test.jsx
git commit -m "feat(p4): StrategyMatrix — ticker x strategy grid (4 tests)"
```

---

## Task 6: SignalDetail Right Panel

**Files:**
- Create: `quant-ai-ui/src/features/signal-console/SignalDetail.jsx`
- Create: `quant-ai-ui/__tests__/features/signal-console/SignalDetail.test.jsx`

- [ ] **Step 6.1: Write failing tests**

Create `quant-ai-ui/__tests__/features/signal-console/SignalDetail.test.jsx`:

```jsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { makeQueryWrapper } from "../../_helpers/queryWrapper";
import SignalDetail from "@/features/signal-console/SignalDetail";

vi.mock("@/api/client", () => ({
  postSignalScore: vi.fn(),
  getMetaLabelModels: vi.fn(), getMetaCoverage: vi.fn(), postMetaLabelTrain: vi.fn(),
}));
import * as client from "@/api/client";

beforeEach(() => vi.clearAllMocks());

describe("SignalDetail", () => {
  it("renders score, expected_R, recommended_action when triggered", async () => {
    client.postSignalScore.mockResolvedValue({
      triggered: true, signal: 1, reliability_score: 0.71, expected_R: 0.54,
      recommended_action: "trade",
      sizing_hint: { half_kelly_fraction: 0.18, raw_kelly: 0.36, cap: 0.25 },
      meta_model: { id: "meta_a", primary_source: "strategy:rsi_strategy", cv_auc: 0.62 },
      timestamp: "2026-04-24T20:00:00Z",
    });
    render(
      <SignalDetail selection={{ ticker: "AAPL", strategy: "rsi_strategy", model_id: "meta_a" }} />,
      { wrapper: makeQueryWrapper() },
    );
    await waitFor(() => expect(screen.getByText(/0\.71/)).toBeInTheDocument());
    expect(screen.getByText(/TRADE/i)).toBeInTheDocument();
    expect(screen.getByText(/\+0\.54/)).toBeInTheDocument();
  });

  it("shows 'silent' message when triggered=false", async () => {
    client.postSignalScore.mockResolvedValue({
      triggered: false, signal: 0, reason: "rsi_strategy did not trigger at latest close",
      timestamp: "2026-04-24T20:00:00Z",
    });
    render(
      <SignalDetail selection={{ ticker: "AAPL", strategy: "rsi_strategy", model_id: "meta_a" }} />,
      { wrapper: makeQueryWrapper() },
    );
    await waitFor(() => expect(screen.getByText(/silent|did not trigger/i)).toBeInTheDocument());
  });

  it("renders nothing when selection is null", () => {
    const { container } = render(<SignalDetail selection={null} />, { wrapper: makeQueryWrapper() });
    expect(container.textContent.toLowerCase()).toContain("select a cell");
  });
});
```

- [ ] **Step 6.2: Run tests — fail**

```bash
npm run test -- --run __tests__/features/signal-console/SignalDetail.test.jsx
```

- [ ] **Step 6.3: Create component**

Create `quant-ai-ui/src/features/signal-console/SignalDetail.jsx`:

```jsx
import { useEffect, useState } from "react";
import { useSignalScorePreview } from "@/api/signalQueries";

export default function SignalDetail({ selection }) {
  const preview = useSignalScorePreview();
  const [resp, setResp] = useState(null);

  useEffect(() => {
    setResp(null);
    if (!selection?.model_id) return;
    preview.mutate(
      { ticker: selection.ticker, meta_model_id: selection.model_id, strategy_name: selection.strategy },
      { onSuccess: setResp },
    );
  }, [selection?.model_id]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!selection) {
    return (
      <div className="p-6 text-sm text-slate-500 bg-slate-900/40 rounded-lg">
        Select a cell in the matrix to see signal detail.
      </div>
    );
  }

  if (preview.isPending || !resp) {
    return (
      <div className="p-6 text-sm text-slate-400 bg-slate-900/40 rounded-lg">
        Loading signal score for {selection.ticker} × {selection.strategy}...
      </div>
    );
  }

  if (!resp.triggered) {
    return (
      <div className="p-6 bg-slate-900/40 rounded-lg space-y-2">
        <div className="text-xs text-slate-400">{selection.ticker} × {selection.strategy}</div>
        <div className="text-sm text-amber-300">
          Strategy silent at latest close — {resp.reason || "did not trigger"}
        </div>
      </div>
    );
  }

  const score = resp.reliability_score ?? 0;
  const action = resp.recommended_action?.toUpperCase() ?? "—";
  const actionColor = action === "TRADE" ? "text-emerald-400" : action === "SKIP" ? "text-rose-400" : "text-amber-300";
  const sizing = resp.sizing_hint ?? {};

  return (
    <div className="p-6 bg-slate-900/40 rounded-lg space-y-4">
      <div>
        <div className="text-xs text-slate-400">
          {selection.ticker} × {selection.strategy}
        </div>
        <div className="text-sm text-slate-500">Model: {resp.meta_model?.id}</div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <Metric label="Reliability score" value={score.toFixed(2)} />
        <Metric
          label="Expected R"
          value={`${resp.expected_R >= 0 ? "+" : ""}${(resp.expected_R ?? 0).toFixed(2)}`}
        />
        <Metric label="Signal" value={resp.signal > 0 ? "Long (+1)" : "Short (-1)"} />
        <Metric label="CV AUC" value={(resp.meta_model?.cv_auc ?? 0).toFixed(2)} />
      </div>

      <div className={`text-lg font-bold ${actionColor}`}>Action: {action}</div>

      {sizing.half_kelly_fraction !== undefined && (
        <div className="text-xs text-slate-400 space-y-1">
          <div>Sizing hint (half-Kelly): {(sizing.half_kelly_fraction * 100).toFixed(1)}% of capital</div>
          <div>Raw Kelly: {(sizing.raw_kelly * 100).toFixed(1)}% · Cap: {(sizing.cap * 100).toFixed(0)}%</div>
        </div>
      )}

      <div className="text-[10px] text-slate-500 pt-2 border-t border-slate-800">
        Primary: {resp.meta_model?.primary_source} · Timestamp: {resp.timestamp}
      </div>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="bg-slate-800/50 rounded px-3 py-2">
      <div className="text-[10px] uppercase text-slate-500">{label}</div>
      <div className="text-lg font-semibold text-slate-100">{value}</div>
    </div>
  );
}
```

- [ ] **Step 6.4: Run tests**

```bash
npm run test -- --run __tests__/features/signal-console/SignalDetail.test.jsx
```

Expected: `3 passed`.

- [ ] **Step 6.5: Commit**

```bash
git add quant-ai-ui/src/features/signal-console/SignalDetail.jsx quant-ai-ui/__tests__/features/signal-console/SignalDetail.test.jsx
git commit -m "feat(p4): SignalDetail right panel (3 tests)"
```

---

## Task 7: SignalConsolePage + Routing

**Files:**
- Create: `quant-ai-ui/src/pages/SignalConsolePage.jsx`
- Modify: `quant-ai-ui/src/App.jsx`
- Modify: `quant-ai-ui/src/components/layout/TopNavBar.jsx`
- Create: `quant-ai-ui/__tests__/pages/SignalConsolePage.test.jsx`

- [ ] **Step 7.1: Write failing tests**

Create `quant-ai-ui/__tests__/pages/SignalConsolePage.test.jsx`:

```jsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { makeQueryWrapper } from "../_helpers/queryWrapper";
import SignalConsolePage from "@/pages/SignalConsolePage";

vi.mock("@/api/client", () => ({
  getMetaLabelModels: vi.fn().mockResolvedValue([]),
  getMetaCoverage: vi.fn().mockResolvedValue({ count: 0, max_auc: null, avg_auc: null, tickers: [], models: [] }),
  postSignalScore: vi.fn(), postMetaLabelTrain: vi.fn(),
}));

beforeEach(() => {
  localStorage.clear();
  localStorage.setItem("quant-ai:watchlist", JSON.stringify(["AAPL", "MSFT"]));
});

describe("SignalConsolePage", () => {
  it("mounts with 3 sections (picker, matrix, detail)", async () => {
    render(
      <MemoryRouter initialEntries={["/signal-console"]}>
        <SignalConsolePage />
      </MemoryRouter>,
      { wrapper: makeQueryWrapper() },
    );
    expect(screen.getByText(/Watchlist/i)).toBeInTheDocument();
    expect(screen.getByText(/Signal Console/i)).toBeInTheDocument();
  });

  it("applies ?strategy= query param (reads without crashing)", async () => {
    render(
      <MemoryRouter initialEntries={["/signal-console?strategy=rsi_strategy"]}>
        <SignalConsolePage />
      </MemoryRouter>,
      { wrapper: makeQueryWrapper() },
    );
    expect(screen.getByText(/rsi_strategy|Signal Console/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 7.2: Run tests — fail**

- [ ] **Step 7.3: Create page**

Create `quant-ai-ui/src/pages/SignalConsolePage.jsx`:

```jsx
import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import TickerPicker from "@/features/signal-console/TickerPicker";
import StrategyMatrix from "@/features/signal-console/StrategyMatrix";
import SignalDetail from "@/features/signal-console/SignalDetail";
import { useMetaLabelTrain } from "@/api/signalQueries";

export default function SignalConsolePage() {
  const [params] = useSearchParams();
  const initialStrategy = params.get("strategy") || null;
  const [selectedTickers, setSelectedTickers] = useState(["AAPL", "MSFT", "GOOGL"]);
  const [selection, setSelection] = useState(null);
  const train = useMetaLabelTrain();

  const onTrain = ({ ticker, strategy }) => {
    train.mutate(
      {
        ticker,
        primary: { source: "strategy", strategy_name: strategy },
        barrier: { tp_k: 2.0, sl_k: 1.0, timeout_days: 5, vol_source: "realized_sigma" },
        cv: { n_splits: 5, embargo_pct: 0.01 },
        model: { type: "xgboost" },
        window: { lookback_days: 730, feature_group: "ta_basic" },
      },
      {
        onSuccess: (data) => {
          setSelection({ ticker, strategy, model_id: data.model_id });
        },
      },
    );
  };

  return (
    <div className="p-6 space-y-4 max-w-7xl mx-auto">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold">Signal Console</h1>
        <p className="text-sm text-slate-400">
          Meta-label signal quality across strategies × tickers. Click a cell to preview its latest signal score.
          {initialStrategy && <span className="ml-2 text-emerald-400">· filtered: {initialStrategy}</span>}
        </p>
      </header>

      <TickerPicker selected={selectedTickers} onChange={setSelectedTickers} />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <StrategyMatrix tickers={selectedTickers} onSelect={setSelection} onTrain={onTrain} />
          {train.isPending && (
            <div className="mt-2 text-xs text-amber-400">Training meta-model... (may take ~5s)</div>
          )}
          {train.isError && (
            <div className="mt-2 text-xs text-rose-400">
              Train failed: {String(train.error?.message || "unknown")}
            </div>
          )}
        </div>
        <div>
          <SignalDetail selection={selection} />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7.4: Wire route in App.jsx**

Edit `quant-ai-ui/src/App.jsx`. Add import:

```jsx
import SignalConsolePage from "@/pages/SignalConsolePage";
```

Add route inside the `<Routes>` block (alongside existing page routes):

```jsx
<Route path="/signal-console" element={<SignalConsolePage />} />
```

- [ ] **Step 7.5: Add TopNavBar link**

In `quant-ai-ui/src/components/layout/TopNavBar.jsx` find the nav-link list and add a new link between 研究 / 模型 groupings:

```jsx
<Link to="/signal-console" className="hover:text-emerald-400 text-sm">信号</Link>
```

- [ ] **Step 7.6: Run tests**

```bash
npm run test -- --run __tests__/pages/SignalConsolePage.test.jsx
```

Expected: `2 passed`.

- [ ] **Step 7.7: Commit**

```bash
git add quant-ai-ui/src/pages/SignalConsolePage.jsx quant-ai-ui/src/App.jsx quant-ai-ui/src/components/layout/TopNavBar.jsx quant-ai-ui/__tests__/pages/SignalConsolePage.test.jsx
git commit -m "feat(p4): SignalConsolePage + /signal-console route + TopNav link (2 tests)"
```

---

## Task 8: Strategy Card Badge Integration

**Files:**
- Modify: `quant-ai-ui/src/features/strategy/StrategyCard.jsx`
- Create or extend: `quant-ai-ui/__tests__/features/strategy/StrategyCard.test.jsx`

- [ ] **Step 8.1: Discover StrategyCard shape**

```bash
head -60 "C:/Users/zjg09/projects/quant-ai/quant-ai-ui/src/features/strategy/StrategyCard.jsx"
```

Note the component's prop shape and where the header/title is rendered.

- [ ] **Step 8.2: Write failing tests**

Create or append to `quant-ai-ui/__tests__/features/strategy/StrategyCard.test.jsx`:

```jsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { makeQueryWrapper } from "../../_helpers/queryWrapper";
import StrategyCard from "@/features/strategy/StrategyCard";

vi.mock("@/api/client", () => ({
  getMetaCoverage: vi.fn(),
  getMetaLabelModels: vi.fn(), postSignalScore: vi.fn(), postMetaLabelTrain: vi.fn(),
}));
import * as client from "@/api/client";

beforeEach(() => vi.clearAllMocks());

const FIXTURE_STRATEGY = {
  name: "rsi_strategy",
  description: "RSI oversold/overbought trigger strategy",
  version: "1.0.0",
};

describe("StrategyCard meta badge", () => {
  it("shows badge when coverage exists", async () => {
    client.getMetaCoverage.mockResolvedValue({
      count: 3, max_auc: 0.62, avg_auc: 0.55, tickers: ["MSFT", "GOOGL", "AAPL"],
    });
    render(
      <MemoryRouter>
        <StrategyCard strategy={FIXTURE_STRATEGY} />
      </MemoryRouter>,
      { wrapper: makeQueryWrapper() },
    );
    await waitFor(() => expect(screen.getByText(/Meta/i)).toBeInTheDocument());
  });

  it("hides badge when coverage is empty", async () => {
    client.getMetaCoverage.mockResolvedValue({
      count: 0, max_auc: null, avg_auc: null, tickers: [], models: [],
    });
    render(
      <MemoryRouter>
        <StrategyCard strategy={FIXTURE_STRATEGY} />
      </MemoryRouter>,
      { wrapper: makeQueryWrapper() },
    );
    // Resolves with count=0 → badge renders null → no "Meta" label
    await waitFor(() => expect(screen.queryByText(/Meta ✓/i)).toBeNull());
  });
});
```

- [ ] **Step 8.3: Run tests — fail (badge not yet embedded)**

- [ ] **Step 8.4: Embed badge in StrategyCard**

Open `quant-ai-ui/src/features/strategy/StrategyCard.jsx`. At the top add import:

```jsx
import MetaLabelCoverageBadge from "@/components/MetaLabelCoverageBadge";
```

Find the header/title line (likely something like `<h3>{strategy.name}</h3>` or a flex container). Add the badge to the right of the name:

```jsx
<div className="flex items-center justify-between mb-2">
  <h3 className="font-semibold">{strategy.name}</h3>
  <MetaLabelCoverageBadge strategyName={strategy.name} />
</div>
```

**If the existing layout is different**, adapt: the essential requirement is the badge is rendered somewhere inside the card, receiving `strategyName={strategy.name}` prop.

- [ ] **Step 8.5: Run tests**

```bash
cd C:/Users/zjg09/projects/quant-ai/quant-ai-ui
npm run test -- --run __tests__/features/strategy/StrategyCard.test.jsx
```

Expected: `2 passed`.

- [ ] **Step 8.6: Commit**

```bash
git add quant-ai-ui/src/features/strategy/StrategyCard.jsx quant-ai-ui/__tests__/features/strategy/StrategyCard.test.jsx
git commit -m "feat(p4): embed MetaLabelCoverageBadge in StrategyCard (2 tests)"
```

---

## Task 9: Paper Trading Modal Meta-Label Integration

**Files:**
- Modify: `quant-ai-ui/src/pages/TradingPage.jsx` (or its order-placement sub-component)
- Create: `quant-ai-ui/__tests__/pages/TradingPage.meta.test.jsx`

- [ ] **Step 9.1: Discover current order form**

```bash
grep -n "side\|qty\|place_order\|submitOrder" "C:/Users/zjg09/projects/quant-ai/quant-ai-ui/src/pages/TradingPage.jsx" | head -15
```

Locate the order form's submit handler and the JSX where inputs are rendered. Identify whether there's a dedicated `<OrderForm>` sub-component.

- [ ] **Step 9.2: Write failing tests**

Create `quant-ai-ui/__tests__/pages/TradingPage.meta.test.jsx`:

```jsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { makeQueryWrapper } from "../_helpers/queryWrapper";
import TradingPage from "@/pages/TradingPage";

vi.mock("@/api/client", () => ({
  getMetaLabelModels: vi.fn(),
  postSignalScore: vi.fn(),
  getMetaCoverage: vi.fn().mockResolvedValue({ count: 0 }),
  postMetaLabelTrain: vi.fn(),
  // preserve other client exports used by TradingPage
  getMarket: vi.fn().mockResolvedValue([]),
  predict: vi.fn(),
}));
import * as client from "@/api/client";

beforeEach(() => vi.clearAllMocks());

describe("TradingPage meta-label integration", () => {
  it("renders meta-label checkbox (off by default)", async () => {
    render(<MemoryRouter><TradingPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    const cb = await screen.findByLabelText(/meta-label/i);
    expect(cb).not.toBeChecked();
  });

  it("expands meta section with dropdown + threshold slider when checked", async () => {
    client.getMetaLabelModels.mockResolvedValue([
      { model_id: "meta_a", extras: { meta_label: {
        primary: { strategy_name: "rsi_strategy" },
        cv: { metrics: { auc_mean: 0.62 } }, barrier: {},
      }}, metadata: { ticker: "AAPL" }},
    ]);
    render(<MemoryRouter><TradingPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    const cb = await screen.findByLabelText(/meta-label/i);
    fireEvent.click(cb);
    await waitFor(() => expect(screen.getByLabelText(/threshold/i)).toBeInTheDocument());
  });

  it("calls postSignalScore when preview button clicked", async () => {
    client.getMetaLabelModels.mockResolvedValue([
      { model_id: "meta_a", extras: { meta_label: {
        primary: { strategy_name: "rsi_strategy" },
        cv: { metrics: { auc_mean: 0.62 } }, barrier: {},
      }}, metadata: { ticker: "AAPL" }},
    ]);
    client.postSignalScore.mockResolvedValue({
      triggered: true, signal: 1, reliability_score: 0.71,
      expected_R: 0.54, recommended_action: "trade",
      sizing_hint: { half_kelly_fraction: 0.18, raw_kelly: 0.36, cap: 0.25 },
      meta_model: { id: "meta_a", primary_source: "strategy:rsi_strategy", cv_auc: 0.62 },
    });
    render(<MemoryRouter><TradingPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    const cb = await screen.findByLabelText(/meta-label/i);
    fireEvent.click(cb);
    const previewBtn = await screen.findByRole("button", { name: /预览 score|preview score/i });
    fireEvent.click(previewBtn);
    await waitFor(() => expect(client.postSignalScore).toHaveBeenCalled());
  });

  it("displays score + sizing after preview", async () => {
    client.getMetaLabelModels.mockResolvedValue([
      { model_id: "meta_a", extras: { meta_label: {
        primary: { strategy_name: "rsi_strategy" },
        cv: { metrics: { auc_mean: 0.62 } }, barrier: {},
      }}, metadata: { ticker: "AAPL" }},
    ]);
    client.postSignalScore.mockResolvedValue({
      triggered: true, signal: 1, reliability_score: 0.71,
      expected_R: 0.54, recommended_action: "trade",
      sizing_hint: { half_kelly_fraction: 0.18, raw_kelly: 0.36, cap: 0.25 },
      meta_model: { id: "meta_a", primary_source: "strategy:rsi_strategy", cv_auc: 0.62 },
    });
    render(<MemoryRouter><TradingPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    fireEvent.click(await screen.findByLabelText(/meta-label/i));
    fireEvent.click(await screen.findByRole("button", { name: /预览 score|preview score/i }));
    await waitFor(() => expect(screen.getByText(/0\.71/)).toBeInTheDocument());
    expect(screen.getByText(/TRADE/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 9.3: Run tests — fail**

- [ ] **Step 9.4: Add meta-label section to TradingPage**

Open `quant-ai-ui/src/pages/TradingPage.jsx`. Near the top of the component add state + hook usage:

```jsx
import { useState } from "react";
import { useMetaLabelModels, useSignalScorePreview } from "@/api/signalQueries";

// ...inside the component:
const [metaEnabled, setMetaEnabled] = useState(false);
const [metaModelId, setMetaModelId] = useState("");
const [metaThreshold, setMetaThreshold] = useState(0.55);
const [metaScore, setMetaScore] = useState(null);
const metaModels = useMetaLabelModels(ticker, { enabled: metaEnabled && !!ticker });
const preview = useSignalScorePreview();

const onPreview = () => {
  if (!metaModelId || !ticker) return;
  preview.mutate(
    { ticker, meta_model_id: metaModelId, signal: side === "buy" ? 1 : -1 },
    { onSuccess: setMetaScore },
  );
};
```

Replace `ticker` and `side` with whatever variable names exist in the page.

Insert this JSX section into the order form (below the existing qty input, above the submit button):

```jsx
<div className="mt-4 border-t border-slate-800 pt-4">
  <label className="flex items-center gap-2 text-sm cursor-pointer">
    <input
      type="checkbox"
      checked={metaEnabled}
      onChange={(e) => setMetaEnabled(e.target.checked)}
      aria-label="Use meta-label filter"
    />
    <span>Use meta-label filter</span>
  </label>
  {metaEnabled && (
    <div className="mt-3 space-y-3 pl-6">
      <div>
        <label className="block text-xs text-slate-400 mb-1">Meta model</label>
        <select
          value={metaModelId}
          onChange={(e) => setMetaModelId(e.target.value)}
          className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm"
        >
          <option value="">— select a model —</option>
          {(metaModels.data || []).map((m) => {
            const prim = m.extras?.meta_label?.primary?.strategy_name || "—";
            const auc = m.extras?.meta_label?.cv?.metrics?.auc_mean ?? 0;
            return (
              <option key={m.model_id} value={m.model_id}>
                {prim} · {m.model_id.slice(0, 12)}... · AUC {auc.toFixed(2)}
                {auc < 0.5 ? " ⚠" : ""}
              </option>
            );
          })}
        </select>
      </div>
      <div>
        <label className="block text-xs text-slate-400 mb-1">
          Threshold: {metaThreshold.toFixed(2)}
        </label>
        <input
          type="range"
          min="0.45"
          max="0.85"
          step="0.01"
          value={metaThreshold}
          onChange={(e) => setMetaThreshold(parseFloat(e.target.value))}
          aria-label="Threshold"
          className="w-full"
        />
      </div>
      <button
        type="button"
        onClick={onPreview}
        disabled={!metaModelId}
        className="px-3 py-1 text-sm bg-emerald-600/20 border border-emerald-600/40 rounded hover:bg-emerald-600/30 disabled:opacity-50"
      >
        预览 score
      </button>
      {metaScore?.triggered && (
        <div className="p-3 bg-slate-800/50 rounded text-xs space-y-1">
          <div>
            Score: <span className="font-semibold">{metaScore.reliability_score.toFixed(2)}</span>
            {" · "}E[R]: {(metaScore.expected_R ?? 0).toFixed(2)}
          </div>
          <div className="uppercase font-bold text-sm">
            Action: <span className={
              metaScore.recommended_action === "trade" ? "text-emerald-400"
                : metaScore.recommended_action === "skip" ? "text-rose-400" : "text-amber-400"
            }>{metaScore.recommended_action}</span>
          </div>
          {metaScore.sizing_hint && (
            <div className="text-slate-400">
              Sizing hint: {(metaScore.sizing_hint.half_kelly_fraction * 100).toFixed(1)}% of capital
            </div>
          )}
        </div>
      )}
      {metaScore && !metaScore.triggered && (
        <div className="text-xs text-amber-400">
          Strategy silent at latest close — {metaScore.reason || "did not trigger"}
        </div>
      )}
    </div>
  )}
</div>
```

Also update the submit-order payload to include `meta_model_id` + `score_threshold` when enabled (so backend Paper Trading gate kicks in):

```jsx
const orderPayload = {
  ticker, side, qty,
  ...(metaEnabled && metaModelId ? { meta_model_id: metaModelId, score_threshold: metaThreshold } : {}),
};
```

- [ ] **Step 9.5: Run tests**

```bash
npm run test -- --run __tests__/pages/TradingPage.meta.test.jsx
```

Expected: `4 passed`.

- [ ] **Step 9.6: Commit**

```bash
git add quant-ai-ui/src/pages/TradingPage.jsx quant-ai-ui/__tests__/pages/TradingPage.meta.test.jsx
git commit -m "feat(p4): Paper Trading modal meta-label filter (checkbox + dropdown + slider + preview) (4 tests)"
```

---

## Task 10: Dashboard VolatilityCard Sparkline

**Files:**
- Create: `quant-ai-ui/src/features/signal-console/MetaSparkline.jsx`
- Modify: `quant-ai-ui/src/features/dashboard/VolatilityCard.jsx`
- Create: `quant-ai-ui/__tests__/features/signal-console/MetaSparkline.test.jsx`

- [ ] **Step 10.1: Write failing tests**

Create `quant-ai-ui/__tests__/features/signal-console/MetaSparkline.test.jsx`:

```jsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { makeQueryWrapper } from "../../_helpers/queryWrapper";
import MetaSparkline from "@/features/signal-console/MetaSparkline";

vi.mock("@/api/client", () => ({
  getMetaLabelModels: vi.fn(),
  postSignalScore: vi.fn(),
  getMetaCoverage: vi.fn(), postMetaLabelTrain: vi.fn(),
}));
import * as client from "@/api/client";

beforeEach(() => vi.clearAllMocks());

describe("MetaSparkline", () => {
  it("renders sparkline when meta-model exists for the ticker", async () => {
    client.getMetaLabelModels.mockResolvedValue([
      { model_id: "meta_m", extras: { meta_label: {
        primary: { strategy_name: "rsi_strategy" },
        cv: { metrics: { auc_mean: 0.62 } }, barrier: {},
      }}, metadata: { ticker: "MSFT" }},
    ]);
    client.postSignalScore.mockResolvedValue({
      triggered: true, reliability_score: 0.6, signal: 1, expected_R: 0.4,
      recommended_action: "trade", sizing_hint: {}, meta_model: {},
    });
    const { container } = render(<MetaSparkline ticker="MSFT" />, { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(container.querySelector("svg,canvas")).toBeTruthy());
  });

  it("renders nothing when no meta-model for ticker", async () => {
    client.getMetaLabelModels.mockResolvedValue([]);
    const { container } = render(<MetaSparkline ticker="NVDA" />, { wrapper: makeQueryWrapper() });
    await waitFor(() => {
      // Resolved query with empty data → component returns null
      expect(container.textContent.trim()).toBe("");
    });
  });
});
```

- [ ] **Step 10.2: Run tests — fail**

- [ ] **Step 10.3: Create MetaSparkline component**

Create `quant-ai-ui/src/features/signal-console/MetaSparkline.jsx`:

```jsx
import { useEffect, useState } from "react";
import { useMetaLabelModels } from "@/api/signalQueries";
import * as api from "@/api/client";

/** Last-7-days reliability score mini-line. */
export default function MetaSparkline({ ticker }) {
  const { data: models = [] } = useMetaLabelModels(ticker);
  const model = models[0];
  const [series, setSeries] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!model?.model_id) return;
    let cancelled = false;
    setLoading(true);
    (async () => {
      const now = new Date();
      const days = Array.from({ length: 7 }, (_, i) => {
        const d = new Date(now);
        d.setDate(now.getDate() - (6 - i));
        return d.toISOString().slice(0, 10);
      });
      const scores = await Promise.all(
        days.map((day) =>
          api
            .postSignalScore({
              ticker, meta_model_id: model.model_id,
              signal: 1, timestamp: day,
            })
            .then((r) => (r?.triggered ? r.reliability_score : null))
            .catch(() => null)
        ),
      );
      if (!cancelled) {
        setSeries(scores);
        setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [model?.model_id, ticker]);

  if (!model) return null;
  if (loading && series.length === 0) {
    return <div className="text-[10px] text-slate-500 mt-2">Loading signal quality...</div>;
  }

  const values = series.filter((v) => v !== null);
  if (values.length === 0) {
    return <div className="text-[10px] text-slate-500 mt-2">No recent triggers</div>;
  }

  // Simple inline SVG sparkline (no dep)
  const W = 120, H = 28;
  const min = Math.min(...values, 0.4);
  const max = Math.max(...values, 0.7);
  const range = Math.max(max - min, 0.01);
  const points = series
    .map((v, i) => {
      if (v === null) return null;
      const x = (i * (W - 6)) / 6 + 3;
      const y = H - 3 - ((v - min) / range) * (H - 6);
      return `${x},${y}`;
    })
    .filter(Boolean)
    .join(" ");

  return (
    <div className="mt-2 flex items-center gap-2">
      <div className="text-[10px] text-slate-400">7d signal quality:</div>
      <svg width={W} height={H} className="overflow-visible">
        <polyline points={points} fill="none" stroke="rgb(16 185 129)" strokeWidth="1.5" />
      </svg>
      <div className="text-[10px] text-slate-500">
        {values[values.length - 1].toFixed(2)}
      </div>
    </div>
  );
}
```

- [ ] **Step 10.4: Embed in VolatilityCard**

Open `quant-ai-ui/src/features/dashboard/VolatilityCard.jsx`. At the top:

```jsx
import MetaSparkline from "@/features/signal-console/MetaSparkline";
```

Inside the JSX, after the existing gauge content and before the closing card wrapper:

```jsx
<MetaSparkline ticker={ticker} />
```

(The component self-decides whether to render based on meta-model availability.)

- [ ] **Step 10.5: Run tests**

```bash
npm run test -- --run __tests__/features/signal-console/MetaSparkline.test.jsx
```

Expected: `2 passed`.

- [ ] **Step 10.6: Commit**

```bash
git add quant-ai-ui/src/features/signal-console/MetaSparkline.jsx quant-ai-ui/src/features/dashboard/VolatilityCard.jsx quant-ai-ui/__tests__/features/signal-console/MetaSparkline.test.jsx
git commit -m "feat(p4): Dashboard VolatilityCard 7d signal-quality sparkline (2 tests)"
```

---

## Task 11: AAPL Optuna Rescue

**Files:**
- Create: `scripts/p4_aapl_optuna_rescue.py`
- Create: `docs/benchmarks/p4_aapl_optuna.md` (success) OR `D:/obsidian vault/Quant/03_Rejected/aapl_rsi_meta.md` (failure)

- [ ] **Step 11.1: Create rescue script**

Create `scripts/p4_aapl_optuna_rescue.py`:

```python
"""
P4 AAPL Optuna Rescue — V4 Phase 4.

P3 benchmark showed AAPL × rsi_strategy meta-model AUC = 0.420 with default
XGBoost params. This script runs Optuna (n=30 trials) to see if hyperparameter
search can lift AUC above 0.5.

Run:
    python -m scripts.p4_aapl_optuna_rescue

Writes:
    - docs/benchmarks/p4_aapl_optuna.md (if AUC >= 0.5)
    - or a Quant/03_Rejected note (caller decides based on stdout)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from app.services.meta_label_service import (
    MetaLabelTrainRequest, train_meta_label_model,
)
from app.services.primary_signal_service import PrimarySignalSpec


def main():
    print("P4 AAPL × rsi_strategy Optuna rescue — 30 trials")
    t0 = time.time()
    req = MetaLabelTrainRequest(
        ticker="AAPL",
        primary=PrimarySignalSpec(source="strategy", strategy_name="rsi_strategy"),
        tp_k=2.0, sl_k=1.0, timeout_days=5,
        vol_source="realized_sigma",
        cv_n_splits=5, cv_embargo_pct=0.01,
        model_type="xgboost",
        search_mode="optuna",
        lookback_days=730, feature_group="ta_basic",
    )
    try:
        result = train_meta_label_model(req)
    except Exception as e:
        print(f"TRAINING FAILED: {e}")
        return
    elapsed = time.time() - t0
    auc = result["cv_metrics"]["auc_mean"]
    print(json.dumps(result, indent=2, default=str))
    print(f"\nElapsed: {elapsed:.1f}s · Optuna best AUC: {auc:.3f}")

    from pathlib import Path
    out = Path("docs/benchmarks/p4_aapl_optuna.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    status = "✅ rescued" if auc >= 0.5 else "❌ honest failure"
    md = f"""# P4 · AAPL × rsi_strategy Optuna Rescue

**Run date**: {now}
**Baseline (P3 default)**: AUC = 0.420
**Optuna (30 trials)**: AUC = {auc:.3f}
**Status**: {status}

## CV Metrics

- AUC mean ± std: {result['cv_metrics']['auc_mean']:.3f} ± {result['cv_metrics']['auc_std']:.3f}
- Precision @ 50%: {result['cv_metrics']['precision_at_50']:.3f}
- E[R | trade]: {result['cv_metrics']['expected_R_when_trade']:+.3f}
- Hit rate: {result['cv_metrics']['hit_rate_when_trade']:.3f}
- Folds used: {result['cv_metrics']['folds_used']}
- Event count: {result['event_count']}

## Interpretation

{_interpret(auc)}

## Raw response

```json
{json.dumps(result, indent=2, default=str)}
```
"""
    out.write_text(md, encoding="utf-8")
    print(f"Report: {out}")


def _interpret(auc: float) -> str:
    if auc >= 0.55:
        return (
            "Optuna successfully lifted AAPL's meta-label AUC above the useful-signal threshold. "
            "This suggests AAPL × rsi_strategy IS meta-labelable but requires tuned hyperparameters. "
            "Recommend updating P3 benchmark addendum with Optuna params."
        )
    if auc >= 0.5:
        return (
            "Optuna barely lifted AUC above 0.5 — marginal rescue. "
            "AAPL × rsi_strategy × ta_basic features are weakly labelable at best. "
            "Consider feature expansion (sentiment, regime) for real lift."
        )
    return (
        "Optuna could not lift AUC above 0.5 even with 30 trials. "
        "This is an **honest failure case**: AAPL × rsi_strategy × default feature set "
        "(ta_basic, 2y window) is not meta-labelable. "
        "Next investigations: (1) longer window (5y+), (2) sentiment feature group, "
        "(3) different primary strategy (momentum or bollinger_breakout), "
        "(4) different barrier config (asymmetric TP/SL). "
        "This is exactly the kind of methodological signal Prado Ch.3 promises: "
        "when the model says 'I can't learn this', you trust it rather than forcing it."
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 11.2: Run the rescue script**

```bash
cd C:/Users/zjg09/projects/quant-ai
python -m scripts.p4_aapl_optuna_rescue 2>&1 | tee p4_aapl_rescue.log
```

Expected: Optuna runs for several minutes. Final output shows AUC + writes `docs/benchmarks/p4_aapl_optuna.md`.

- [ ] **Step 11.3: Copy report to vault (success path) or create rejected note (failure path)**

**If AUC >= 0.5:**
```bash
cp "docs/benchmarks/p4_aapl_optuna.md" \
   "D:/obsidian vault/01-projects/quant-ai/p4-aapl-rescue.md"
```

**If AUC < 0.5:** append a short note to `D:/obsidian vault/Quant/03_Rejected/aapl_rsi_meta.md`:
```bash
# Create directory if needed (Obsidian may auto-create on next open)
mkdir -p "D:/obsidian vault/Quant/03_Rejected"

# Write rejected note
cat > "D:/obsidian vault/Quant/03_Rejected/aapl_rsi_meta.md" <<'EOF'
# AAPL × rsi_strategy meta-labeling — Rejected

Date: 2026-04-24

## Hypothesis

AAPL × rsi_strategy with ta_basic features + 2y daily bars should be meta-labelable
(reliability score >= 0.5 AUC) via Prado Ch.3 methodology.

## Test

P3 default params: AUC 0.420.
P4 Optuna 30 trials: AUC <value from script>.

## Result

Both below 0.5 — the meta-model cannot distinguish real signals from noise on
AAPL × rsi_strategy at the 5-day horizon with ta_basic features.

## Interpretation

This is a legitimate Prado-style methodological signal: when the data speaks
"no learnable edge at this configuration", the honest answer is to accept it,
not to p-hack until something works.

## Next investigations (not blocking Gate 1)

- Longer window (5y+ data)
- Feature expansion: sentiment, regime state, cross-sectional
- Different primary: momentum, bollinger_breakout
- Different barrier: asymmetric TP/SL, non-volatility-scaled

See [[01-projects/quant-ai/p3-meta-labeling-design]] §12 Future Backlog.
EOF
```

- [ ] **Step 11.4: Commit**

```bash
git add scripts/p4_aapl_optuna_rescue.py docs/benchmarks/p4_aapl_optuna.md
git commit -m "feat(p4): AAPL Optuna rescue script + findings (success or honest-failure)"
```

---

## Task 12: P4 GATE — Regression + Progress Log + Tags + Live Smoke

**Files:**
- Modify: `D:/obsidian vault/01-projects/quant-ai/ml-pivot-progress.md` (Day 14 entry)
- Modify: `D:/obsidian vault/01-projects/quant-ai/master-roadmap.md` (P4 + Gate 1 marked complete)

- [ ] **Step 12.1: Frontend regression guard**

```bash
cd C:/Users/zjg09/projects/quant-ai/quant-ai-ui
npm run test -- --run
npm run lint
npm run build
```

Expected: all existing tests + 26 new P4 tests green; lint clean; build succeeds.

- [ ] **Step 12.2: Backend regression guard**

```bash
cd C:/Users/zjg09/projects/quant-ai
pytest tests/contract/test_meta_coverage.py \
       tests/contract/test_meta_label_train.py \
       tests/contract/test_signal_score.py \
       tests/test_paper_trading_meta.py \
       tests/test_meta_label_barrier.py \
       tests/test_purged_kfold.py \
       tests/test_labels.py \
       tests/test_ensemble_training.py \
       tests/contract/test_train_flow.py \
       -v
```

Expected: all green (P1 + P2 + P3 + P4 regression + new P4 backend).

- [ ] **Step 12.3: Append Day 14 entry to progress log**

Append to `D:/obsidian vault/01-projects/quant-ai/ml-pivot-progress.md`:

```markdown
### Day 14 Sprint · 2026-04-24 (Thu) · P4 Ship · Gate 1 Closer

**Mode**: Full-day P4 sprint per Harry's "今天P3 明天P4" directive.

#### ✅ Delivered

- `app/api/signal.py` + `signal_scoring_service.py` · `GET /api/meta-label/coverage` (5 tests)
- `quant-ai-ui/src/components/MetaLabelCoverageBadge.jsx` (3 tests)
- `quant-ai-ui/src/api/signalQueries.js` + `api/client.js` extensions (3 tests)
- `quant-ai-ui/src/features/signal-console/TickerPicker.jsx` (3 tests)
- `quant-ai-ui/src/features/signal-console/StrategyMatrix.jsx` (4 tests)
- `quant-ai-ui/src/features/signal-console/SignalDetail.jsx` (3 tests)
- `quant-ai-ui/src/features/signal-console/MetaSparkline.jsx` (2 tests)
- `quant-ai-ui/src/pages/SignalConsolePage.jsx` + route (2 tests)
- `quant-ai-ui/src/features/strategy/StrategyCard.jsx` — badge integration (2 tests)
- `quant-ai-ui/src/pages/TradingPage.jsx` — meta-label filter UI (4 tests)
- `app/main.py` — version bump 2.1.0 → 2.4.0 (P3 carryover)
- `scripts/p4_aapl_optuna_rescue.py` + report (success or honest-failure)

**Test total**: ~31 new P4 tests + full regression green.

**Methodology**: Fractional V4 story closing — frontend makes Prado Ch.3 meta-labeling visible + clickable. Signal Console is the end-to-end interview demo page.

**Design trace**: [[p4-signal-console-design]] + `docs/superpowers/specs/2026-04-24-p4-signal-console-design.md`
**Plan trace**: `docs/superpowers/plans/2026-04-24-p4-signal-console.md`

**🏁 GATE 1 — V4 Full Story Demo Ready · COMPLETE**
```

- [ ] **Step 12.4: Update master-roadmap.md**

In `D:/obsidian vault/01-projects/quant-ai/master-roadmap.md` find the P4 section and update title to "✅ 完成 (2026-04-24)". Also update the Gate 1 line to "🏁 Gate 1 · ✅ COMPLETE (2026-04-24)".

- [ ] **Step 12.5: Tag + push**

```bash
cd C:/Users/zjg09/projects/quant-ai
git tag -a v4-p4-complete -m "V4 Pivot P4 Signal Console frontend complete"
git tag -a v4-gate-1-complete -m "V4 Pivot Gate 1 complete — full story demo ready"
git push origin main --follow-tags
```

- [ ] **Step 12.6: Live smoke (post-deploy)**

Wait for Render auto-deploy then:

```bash
curl -s https://quant-ai-qzrg.onrender.com/health | python -c "import json,sys; print(json.load(sys.stdin)['version'])"
# Expected: 2.4.0

curl -s "https://quant-ai-qzrg.onrender.com/api/meta-label/coverage?strategy=rsi_strategy" -w "\nHTTP %{http_code}\n"
# Expected: HTTP 200 (with count:3 if MSFT/GOOGL/AAPL meta-models exist in prod, else count:0)

curl -s -o /dev/null -w "HTTP %{http_code}\n" https://quant-ai-ui.vercel.app/signal-console
# Expected: HTTP 200
```

- [ ] **Step 12.7: Final commit (if not already via vault edits)**

Vault files don't need git commit (vault is not a git repo). Code-repo commits have already been made per task.

Verify:
```bash
cd C:/Users/zjg09/projects/quant-ai
git log --oneline origin/main..HEAD
# Expected: empty (all commits pushed)

git tag | grep v4-
# Expected: includes v4-p1-complete (if exists), v4-p3-complete, v4-p4-complete, v4-gate-1-complete
```

---

## Self-Review

**1. Spec coverage (§ references from P4 spec):**
- §3 Scope in — all 13 in-scope items covered across Tasks 0-11 ✅
- §4 Architecture components table — all covered ✅
- §5 Data flow (Signal Console load, Strategy badge, Paper Trading, Dashboard sparkline) — Tasks 3, 5, 6, 7, 9, 10 ✅
- §6 API contracts — Task 1 (new coverage), Tasks 2-10 reuse existing P3 ✅
- §7 UI / UX details — Tasks 3, 5, 6, 9 ✅
- §8 Error handling rows — distributed across Tasks 3, 5, 6, 9 ✅
- §9 Testing strategy (31 tests) — all 26 frontend + 5 backend mapped ✅
- §10 AAPL rescue — Task 11 ✅
- §11 Success criteria — all fall out of Tasks 11 + 12 ✅

**2. Placeholder scan:** no TBDs, every code step has full code. ✅

**3. Type consistency:**
- `useMetaLabelModels(ticker)` → returns `[{ model_id, metadata, extras.meta_label.{primary, cv, barrier, event_count} }]` — consistent across Tasks 2, 5, 9, 10 ✅
- `useMetaCoverage(strategyName)` → returns `{ count, max_auc, avg_auc, tickers, models }` — consistent across Tasks 2, 3, 8 ✅
- `useSignalScorePreview()` mutation → accepts `{ ticker, meta_model_id, signal, timestamp?, strategy_name? }` returns `{ triggered, signal, reliability_score, expected_R, recommended_action, sizing_hint, meta_model, timestamp }` — consistent across Tasks 2, 6, 9, 10 ✅
- `MetaLabelCoverageBadge` prop: `strategyName` (singular) — consistent Tasks 3, 8 ✅
- `TickerPicker` props: `selected, onChange` — consistent Tasks 4, 7 ✅
- `StrategyMatrix` props: `tickers, onSelect, onTrain` — consistent Tasks 5, 7 ✅
- `SignalDetail` prop: `selection: { ticker, strategy, model_id } | null` — consistent Tasks 6, 7 ✅

**4. Ambiguity fixes applied:**
- MAX_SELECTED=10 for TickerPicker — explicit ✅
- Threshold slider range 0.45-0.85 step 0.01 default 0.55 — explicit in Task 9 ✅
- Coverage aggregation: skip malformed records, sort tickers — explicit in Task 1 step 1.3 ✅
- Inline sparkline uses raw SVG (no new dep) — explicit in Task 10 ✅

All checks passed. Plan ready for Ralph execution.
