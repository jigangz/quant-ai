# Quant AI P6 Productization Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task-by-task. Steps use `- [ ]` checkboxes.

**Goal:** Turn Quant AI from a multi-page tool into a product a cold visitor can use in 60 seconds and a client can actually run — maximizing resume / Mercor / Upwork value.

**Architecture:** Backend FastAPI (20 routers, Render). Frontend Vite + React 19 + Tailwind + Tremor + shadcn/radix (Vercel). This plan: fix 2 honest ablation bugs, add a Portfolio page, add a client-usable Demo mode + first-visit tour, unify the design language, rewrite the README with screenshots + live links.

**Tech Stack:** React 19, react-router-dom 7, @tanstack/react-query 5, zustand 4, Tailwind design tokens, pytest, vitest.

---

## File Structure

```
app/services/ablation_service.py        # MODIFY — honest metric extraction + labeled mock data
tests/services/test_ablation_service.py # MODIFY/CREATE — metric extraction tests
quant-ai-ui/src/
  pages/PortfolioPage.jsx                # CREATE — Sub 3 new page
  features/portfolio/PortfolioSummary.jsx# CREATE — bullish/bearish distribution + weights
  features/portfolio/portfolioQueries.js # CREATE — react-query hook over /agents/summary
  components/DemoBanner.jsx              # CREATE — "Demo data — Render cold start ~30s" banner
  features/onboarding/Tour.jsx           # CREATE — first-visit 4-step tour (localStorage-gated)
  lib/demoMode.js                        # CREATE — demo ticker presets + cold-start helper
  app/router.jsx                         # MODIFY — add /portfolio route
  app/Sidebar.jsx                        # MODIFY — add Portfolio nav item
  pages/LeaderboardPage.jsx              # MODIFY — slate-* → design tokens
  pages/AblationPage.jsx                 # MODIFY — slate-* → design tokens
README.md                               # MODIFY — full rewrite
docs/screenshots/                       # CREATE — captured UI screenshots
```

---

## Phase P6-1: Honest Ablation Bug Fix

### Task 1: Make ablation metric extraction honest + explicit

**Root cause:** `_extract_metrics(target, result)` reads `metrics.get("test_auc", 0.0)`. Training writes keys as `{split}_{k}` (e.g. `test_auc`), so the key is right — an AUC of 0.0 means the cell genuinely failed to produce a usable metric (empty metrics dict / model didn't converge), but the UI shows a bare `0.000` indistinguishable from a real score. Fix = distinguish "metric absent" (None → render "n/a") from "metric is genuinely 0".

**Files:**
- Modify: `app/services/ablation_service.py:67-87` (`_extract_metrics`)
- Test: `tests/services/test_ablation_service.py`

- [ ] **Step 1: Write failing test** — missing key → None, not 0.0

```python
# tests/services/test_ablation_service.py
from app.services.ablation_service import _extract_metrics

class _R:
    def __init__(self, metrics): self.metrics = metrics

def test_direction_missing_auc_returns_none_not_zero():
    out = _extract_metrics("direction", _R({}))      # no test_auc key
    assert out["auc"] is None                          # honest: absent, not 0.0

def test_direction_real_auc_passthrough():
    out = _extract_metrics("direction", _R({"test_auc": 0.62, "test_f1": 0.55}))
    assert out["auc"] == 0.62 and out["f1"] == 0.55

def test_direction_genuine_zero_is_kept():
    out = _extract_metrics("direction", _R({"test_auc": 0.0}))
    assert out["auc"] == 0.0                            # real 0 stays 0
```

- [ ] **Step 2: Run, verify fail** — `pytest tests/services/test_ablation_service.py -v` → FAIL (currently returns 0.0)

- [ ] **Step 3: Implement** — None when key absent, float when present

```python
def _opt(metrics: dict, key: str):
    v = metrics.get(key)
    return float(v) if v is not None else None

def _extract_metrics(target: str, result) -> dict:
    if target == "meta_label":
        cv = result.get("cv_metrics", {}) if isinstance(result, dict) else {}
        return {"auc_mean": _opt(cv, "auc_mean"), "precision_at_50": _opt(cv, "precision_at_50")}
    metrics = getattr(result, "metrics", None) or {}
    if target == "direction":
        return {"auc": _opt(metrics, "test_auc"), "f1": _opt(metrics, "test_f1")}
    if target == "volatility":
        return {"qlike": _opt(metrics, "test_qlike"), "r2": _opt(metrics, "test_r2"), "mae": _opt(metrics, "test_mae")}
    return {}
```

Guard delta math (`run_ablation`): skip delta when `baseline_val is None` (already does) AND when `cell[primary_metric] is None`.

- [ ] **Step 4: Run, verify pass** — `pytest tests/services/test_ablation_service.py -v` → PASS

- [ ] **Step 5: Frontend renders None as "n/a"** — `features/ablation/AblationMatrix.jsx`: cell value `null/undefined` → render muted `n/a`, not `0.000`. Add vitest.

- [ ] **Step 6: Commit** — `git commit -m "fix(ablation): distinguish absent metric (n/a) from genuine 0 — honest ablation"`

### Task 2: Label sentiment feature group as mock (honesty over fake delta)

**Root cause:** sentiment delta=0 because `MockSentimentProvider` returns constant values → no signal. Don't fake it. Surface it honestly.

- [ ] **Step 1:** In ablation summary output, when a feature set name contains `sentiment` and provider is mock, add `"note": "sentiment is mock-provider in this build — delta reflects no real news signal"`.
- [ ] **Step 2:** AblationMatrix shows an ℹ️ tooltip on sentiment column with that note.
- [ ] **Step 3:** Commit — `git commit -m "feat(ablation): label mock-sentiment columns honestly in matrix"`

---

## Phase P6-2: Portfolio Page (Sub 3)

### Task 3: portfolioQueries hook over /agents/summary

**Files:** Create `quant-ai-ui/src/features/portfolio/portfolioQueries.js`, test alongside.

`/agents/summary` (POST `{tickers, model_id}`) already exists (agents.py:259). Hook:

```javascript
import { useQuery } from "@tanstack/react-query";
import { agentSummary } from "@/api/client";

export function usePortfolioSummary(tickers, enabled = true) {
  return useQuery({
    queryKey: ["portfolio-summary", tickers],
    queryFn: () => agentSummary({ tickers }),
    enabled: enabled && tickers.length > 0,
    staleTime: 60_000,
  });
}
```

- [ ] Test: mock `agentSummary`, assert hook disabled when `tickers=[]`. Commit.

### Task 4: PortfolioSummary component + PortfolioPage

**Files:** Create `features/portfolio/PortfolioSummary.jsx`, `pages/PortfolioPage.jsx`.

Layout (reuse Dashboard design tokens + Tremor): ticker multi-select (reuse `TickerSearch`) → bullish/bearish/neutral distribution donut (Tremor `DonutChart`) → per-ticker signal cards with suggested weight → "copy to Paper Trading" button (deep-link `/trading?prefill=...`). Empty state via existing `EmptyState`. Error via `ErrorState`. Loading via `LoadingSpinner`.

- [ ] Build component with the three states; vitest renders empty + populated. Commit.

### Task 5: Wire route + nav

- [ ] `app/router.jsx`: add `const PortfolioPage = lazy(...)` + `<Route path="portfolio" ...>`.
- [ ] `app/Sidebar.jsx`: add `{ to: "/portfolio", label: "Portfolio", icon: Briefcase }` (import `Briefcase` from lucide-react).
- [ ] vitest: router renders PortfolioPage at /portfolio. Commit `feat(portfolio): new Portfolio page + nav (Sub 3)`.

---

## Phase P6-3: Demo Mode + Onboarding (the "client can use it" core)

### Task 6: demoMode helper + cold-start banner

**Files:** Create `lib/demoMode.js`, `components/DemoBanner.jsx`.

```javascript
// lib/demoMode.js
export const DEMO_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA"];
export const DEMO_PORTFOLIO = ["AAPL", "MSFT", "NVDA"];
// Render free tier cold-starts ~30s; ping health on app mount, expose status
export async function pingBackend(base) {
  const t0 = Date.now();
  try { await fetch(`${base}/health`, { signal: AbortSignal.timeout(45000) }); return { up: true, ms: Date.now() - t0 }; }
  catch { return { up: false, ms: Date.now() - t0 }; }
}
```

`DemoBanner`: dismissible top banner — "🎬 Live demo on Render free tier — first request may take ~30s to wake the server. Try AAPL / MSFT / NVDA." Persists dismissal in localStorage.

- [ ] Build + vitest (renders, dismiss writes localStorage). Mount in `AppShell`. Commit.

### Task 7: First-visit Tour

**Files:** Create `features/onboarding/Tour.jsx`.

4-step overlay (localStorage `qa_tour_done` gate), no extra deps — a fixed-position card with Next/Skip cycling through copy:
1. "Pick a stock → Screener shows AI signals"
2. "Dashboard = one-screen analysis: prediction + SHAP + agent summary"
3. "Portfolio = bullish/bearish across your watchlist"
4. "Everything is live — models retrain, accuracy tracked honestly in Leaderboard"

- [ ] Build + vitest (first visit shows, after done hidden). Mount in AppShell. Commit `feat(onboarding): first-visit tour + demo mode`.

---

## Phase P6-4: Polish / Design-language Consistency

### Task 8: Migrate P5 pages to design tokens

**Files:** `pages/LeaderboardPage.jsx`, `pages/AblationPage.jsx`, `features/leaderboard/*`, `features/ablation/*`.

Replace raw dark classes with tokens: `text-slate-400`→`text-muted`, `border-slate-800`→`border-surface-border`, `text-emerald-300`→`text-accent`, `bg-slate-*`→`bg-surface-card`. Match Dashboard/Sidebar.

- [ ] Grep `slate-` under those files, replace, run `npm run build` + vitest. Commit `style(p5): migrate Leaderboard/Ablation to design tokens`.

### Task 9: Unify empty/error/loading across pages

- [ ] Audit each page (Screener/Dashboard/Strategy/Trading/Training/SignalConsole/Portfolio): query error → `<ErrorState>`, empty → `<EmptyState>`, loading → `<LoadingSpinner>`. Replace ad-hoc "Loading..." text. Commit `refactor(ui): unify empty/error/loading states`.

---

## Phase P6-5: README Rewrite + Screenshots

### Task 10: Capture screenshots

- [ ] Use preview tools (or Vercel live) to screenshot: Dashboard, Screener, Portfolio, Leaderboard, Ablation. Save to `docs/screenshots/*.png`. Commit.

### Task 11: Rewrite README

Sections: hero one-liner + live links badge row · 30s "what it does" + GIF/screenshot · **architecture diagram** (mermaid: Next-ish UI → FastAPI 20 routers → XGB/LGBM/CatBoost + Optuna + SHAP → Supabase/Redis; Kafka KRaft + K8s HPA for distributed train) · feature list mapped to screenshots · honest "what's real vs demo" table (live accuracy real; sentiment mock — own it) · **one-command quickstart** (`docker-compose up`) · tech stack · roadmap link.

- [ ] Write README, verify links resolve. Commit `docs: rewrite README — screenshots, architecture, quickstart, live links`.

---

## Phase P6-GATE: Verify + Push

- [ ] `cd quant-ai-ui && npm run build` → 0 errors
- [ ] `npm test -- --run` → green
- [ ] `pytest tests/services/test_ablation_service.py -v` → green
- [ ] `git push origin main`
- [ ] Vercel auto-deploy → `curl -I https://quant-ai-ui.vercel.app/portfolio` → 200
- [ ] Render smoke → `curl https://quant-ai-qzrg.onrender.com/health`
- [ ] Update PROFILE.md + projects.md with Portfolio page + honest-ablation + demo mode (background-update protocol)

---

## Self-Review

- **Spec coverage:** Sub 3 (Portfolio) ✅ Task 3-5. Demo/onboarding ✅ Task 6-7. Honest data ✅ Task 1-2. Polish ✅ Task 8-9. README ✅ Task 10-11. Sub 4/5/6 (Training drawer / Strategy diagram / Paper Trading context) = deferred to P6-6 phase 2 (refinements of working pages, lower marginal demo value than a new page + client-usable demo + a README recruiters actually read). **This is an explicit scope call, noted in task-plan.**
- **No placeholders:** each task has concrete files + code.
- **Type consistency:** `agentSummary` (existing client.js:221), `EmptyState`/`ErrorState`/`LoadingSpinner` (existing components), design tokens (existing theme/tokens.css) all verified present.
