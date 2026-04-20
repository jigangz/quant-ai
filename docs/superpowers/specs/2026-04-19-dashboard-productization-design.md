# Dashboard Productization — Design Spec V2

> **Phase 3 Sub 5 · Frontend Productization — Sub-project 1 of 7**
> Date: 2026-04-19 (V2 — pivoted to TradingView-faithful layout + light theme, app-wide staged migration)
> Parent brief: `D:/obsidian vault/01-projects/quant-ai/frontend-productization-brief.md`
> Companion: `docs/superpowers/specs/2026-04-19-dashboard-claude-design-prompt.md`
> Wireframe: `.superpowers/brainstorm/1726-1776662616/content/08-dashboard-tv-faithful.html`
> Reference: <https://cn.tradingview.com/symbols/CBOE-VIX/>

## Context

Harry reviewed V1 ("三段式" dark theme) against the actual TradingView symbol page and pivoted to clone TradingView faithfully with light theme, app-wide. This spec V2 supersedes V1.

Why: target user = retail investor familiar with TradingView / 东方财富 / 同花顺. TV's information density helps, not hurts. TV's signature 3-gauge widget maps perfectly onto our rule signals + AI prediction.

## Goals

1. **Retail familiarity**: Dashboard looks and feels like a TradingView symbol page
2. **AI differentiation layered**: insight band (预测 + 摘要 + SHAP) + AI gauge + model comparison section + seasonality all hit our differentiators
3. **All 16 backend features surface** (the Dashboard + Right Rail together expose every API domain)
4. **Define light theme tokens** for the whole app (Sub 2-7 each migrate one page)
5. **Staged migration**, not big-bang — other 5 pages keep dark until their sub

## Out of Scope

- Migrating Screener / Training / Strategy / Paper Trading / Explain / Portfolio pages to light (Sub 2-7 each)
- Mobile <768px (simplified fallback only)
- Backend endpoint for historical prediction accuracy per ticker per month (use client-side fallback or "data accumulating" placeholder)
- Watchlist persistence in DB (localStorage for MVP)
- Buy signal snapshotting for P&L attribution (Sub 6 Paper Trading)

## Layout: 17 sections

Inherit AppShell (64px collapsed sidebar). Main area: 2-column grid — content (1fr, max 1200px) + right rail (280px sticky).

```
┌ Sidebar 64px │ Top Nav Bar (shared) ────────────────────────────┐
│              ├──────────────────────────────────────────────────┤
│              │ CONTENT (1fr, 1200px max)    │ RIGHT RAIL 280px  │
│              │                              │ (sticky)          │
│              │ §1 Breadcrumb                │                   │
│              │ §2 Symbol Header             │ §16 Watchlist     │
│              │ §3 Tabs                      │   - Indices       │
│              │ §4 AI Insight Band (3 cards) │   - Holdings      │
│              │ §5 Chart + perf pills        │   - Current card  │
│              │ §7 Key Data (4 cols)         │   - Performance   │
│              │ §8 Description               │                   │
│              │ §9 Related Stocks (6 cards)  │                   │
│              │ §10 News (4-col grid)        │                   │
│              │ §11 Model Comparison         │                   │
│              │ §12 3 Gauges 🎯              │                   │
│              │ §13 Seasonality (12 mo)      │                   │
│              │ §14 CTA Row                  │                   │
└──────────────┴──────────────────────────────┴─ §15 Floating ❓ ─┘
```

### §1 Top Nav Bar (shared, inherited into AppShell)
- Logo "Quant AI" (indigo-600 bold) · nav `市场 · 研究 · 模型 · 更多` · search input (`Ctrl+K`) · "升级" CTA · user avatar
- White bg, 1px zinc-200 border-bottom, 48px tall

### §2 Breadcrumb
- `市场 / 美国 / 股票 / AAPL` · zinc-500 12px

### §3 Symbol Header
- 52px circular logo (company-specific icon, fallback to ticker initial)
- Company name (24px bold) + ticker chip + exchange chip
- Price row: price (28px mono) + USD label + change $ + change %
- "在 {date} {tz} 收盘" 10px zinc-400

### §4 Tab Navigation
- Tabs: **概览** (active) · 新闻 · 社区 · 技术指标 · 模型历史 · 预测记录
- This spec covers 概览 only; other tabs stubbed (§future subs)

### §5 AI Insight Band (our key insertion — 3 cards in one row)
Grid 1fr / 1.5fr / 1fr, 10px gap.

- **Card 1 · 🤖 AI 预测** (subtle emerald gradient): direction arrow + label ("↗ 看涨") · confidence chip (高/中/低) · horizon · `prob_up 0.68`
- **Card 2 · ⚡ 为什么这么说** (indigo left-border accent): natural-language summary 3-4 lines
- **Card 3 · 📊 SHAP Top 3**: 3 horizontal bars (feature name · bar · value)

Source: single `POST /agents/technical` call drives all 3 cards + the 3 gauges later.

### §6 Chart Section (note: combined with perf pills below; kept as §5-§7 in wireframe, consolidated here for spec)
- Header: "图表 ›" (clickable fullscreen) · export icons
- Lightweight Charts candlestick + area fill under close line · 6M default · 380px tall · zinc-200 grid

### §7 Performance Pills
- 9-column row: `1天 / 5天 / 1月 / 6月 / YTD / 1年 / 5年 / 10年 / 全部`
- Each: label above, % colored by sign below
- Active range highlighted · click to change chart range

### §8 关键数据点
- 4 columns: 成交量 · 前一次收盘 · 开盘价 · 当日价格范围
- Label 10px zinc-500 above value 14px mono

### §9 About / Description
- 2-3 line auto-generated paragraph. Template:
  > `{company_name}` 是一家 `{industry}` 公司。AI 模型基于过去 2 年日线数据训练，使用技术指标（RSI/MACD/Bollinger）、动量、波动率、成交量、情绪、新闻 6 组特征。当前模型 `{model_name}` · run #`{run_id}` · git `{sha}`（`{trained_on}` 训练，AUC `{auc}`）。

### §10 相关股票
- Title "相关股票" + subtitle "同行业 + AI 预测信号"
- 6-card horizontal grid: ticker · company · price · 🤖 signal (emerald/rose/zinc)
- Click → navigate to `/dashboard?ticker=XYZ`
- Sources: hardcoded sector-peer list (MVP) + `/data/market` + `POST /agents/summary`

### §11 News Grid
- Title "新闻 ›" clickable to full News tab
- 4-column grid, 8 headlines default, "继续阅读" to expand
- Each tile: `{time} · {source}` (9px zinc-500) / headline (2-line max)
- Source: `GET /data/sentiment?ticker&days=30` (news array)

### §12 历史模型对此股的预测 (replaces TV's 观点)
- Title + subtitle "你训练过的 / 线上其他模型对此股的预测对比"
- 4 cards: sparkline header + model name + AUC + accuracy %
- Current promoted marked with ⭐
- Sources: `GET /models?ticker=AAPL&status=active` + (stretch) backend accuracy endpoint / fallback client-side compute from runs
- Fallback if no history: single tile "首次训练后开启"

### §13 技术指标 · 3 Gauges ⭐
- Title "技术指标 ›" + subtitle "总结指标的建议"
- 3 semi-circle SVG gauges side by side, middle (AI) slightly emphasized (indigo-50 tint bg)
- **Gauge 1 · 震荡指标 (RSI/MACD)** — from `/agents/technical` signals[] filtered to momentum/oscillator
- **Gauge 2 · 🤖 AI 模型总结** — from `/agents/technical` prediction + probability
- **Gauge 3 · 移动平均线** — from signals[] filtered to MA-type indicators

Scale: 强烈卖出 / 卖出 / 中立 / 买入 / 强烈买入. Needle position from aggregated score, bottom label text.

Scoring:
```
AI gauge:
  prob_up < 0.3  → 强烈卖出 (-2)
  0.3..0.45      → 卖出     (-1)
  0.45..0.55     → 中立      (0)
  0.55..0.7      → 买入     (+1)
  prob_up > 0.7  → 强烈买入 (+2)

震荡 / MA gauges:
  score = sum(signal.direction * signal.strength)
    direction ∈ {bullish: +1, bearish: -1, neutral: 0}
    strength  ∈ {strong: 1.0, moderate: 0.67, weak: 0.33}
  normalize to -2..+2
```

Colors (band by band): rose-600 · rose-300 · zinc-300 · emerald-300 · emerald-600.

### §14 季节性 (Seasonality)
- Title "季节性 ›" + subtitle "过去模型在这个月份的预测准确率"
- 12-cell horizontal heatmap (1月..12月)
- Cell: month label · accuracy %. Color band: <50% rose-100 · 50-60% amber-100 · >60% emerald-100
- Source: client-side from run history OR backend stretch endpoint
- Fallback: "数据积累中" placeholder tile

### §15 CTA Row
2 equal buttons, 16px gap, 48px tall:
- Primary "🛒 基于此信号纸上下单" (indigo-600 fill) → `/trading?ticker=AAPL&side=buy&suggestion_source=dashboard&confidence=high&prediction_timestamp=...`
- Secondary "🧪 训练新模型 (AAPL 专属)" (zinc-300 outline) → `/training?ticker=AAPL&preset=xgboost_default`

### §16 Right Rail Watchlist (sticky, 280px)
zinc-50 bg, 1px zinc-200 left border, 12px padding.

Content (top to bottom):
- Header: "Watchlist" + `+ ⚙` icons
- ▼ INDICES: VIX / DXY / NDQ (seeded, removable)
- ▼ YOUR HOLDINGS: AAPL / TSLA / MSFT / AMZN (seeded from current ticker + paper-trading positions)
- Current ticker card (highlighted): big price + AI prediction summary
- 表现 mini-cards: `1W / 1M / YTD` colored tiles

Source: localStorage (MVP) + `GET /data/market` for each ticker + `POST /agents/summary` for signals batch.

### §17 Floating RAG Q&A Button (global, all pages)
- `fixed bottom-6 right-6` (may need `right-[304px]` on Dashboard to avoid covering watchlist — use responsive offset)
- 56px indigo-600 circle + white ❓
- Click → dialog with input textarea + RAG answer
- Shared `<GlobalRagButton>` in `AppShell`

## Data Flow

```
URL: /dashboard?ticker=AAPL[&modelId=42]
│
├── useMarket(ticker, '6mo') ─── GET /data/market
│     → Chart + performance pills + Key Data
│
├── useAgentTechnical(ticker, modelId?) ─── POST /agents/technical
│     body: { ticker, model_id, include_shap: true, top_features: 5 }
│     → AI Insight Band · 3 Gauges · SHAP Top 3 · Rule signals derivation
│
├── useSentiment(ticker, 30) ─── GET /data/sentiment
│     → News grid · current sentiment score (for rail summary)
│
├── useModelMeta(modelId) ─── GET /models/{id} (enabled: !!modelId)
│     → Description block (trained_on, AUC, git_sha)
│
├── useModelsForTicker(ticker) ─── GET /models?ticker&status=active
│     → Model Comparison §12
│
├── useRelatedStocks(ticker) ─── static sector list + parallel /data/market + /agents/summary
│     → Related Stocks §10
│
├── useWatchlist() ─── localStorage.getItem('quant-ai:watchlist')
│     → Watchlist rail data (plus fetches market + summary for each)
│
└── useSeasonalAccuracy(ticker) ─── client-side compute OR (stretch) backend
      → Seasonality §14
```

All via TanStack Query. Stale time: 60s market, 300s predictions, 600s sector peers.

## States

### Loading (per-section skeleton, no page-level spinner)
- AI Band & Gauges: single skeleton group (they share the same `/agents/technical` call)
- Chart: shimmer
- Related / News / Model Comparison / Seasonality: skeleton tiles
- Watchlist rail: skeleton rows

### Empty
- No ticker in URL: redirect to `/screener` with toast "选一支股票先"
- Unknown ticker (404 market): error card + "Back to Screener"
- No sector peers: hide §10 section
- No seasonality data: "数据积累中" placeholder
- No model history: "首次训练后开启" placeholder

### Error
- `/agents/technical` 500 → AI Band + 3 Gauges replaced by error card + retry; other sections unaffected
- `/agents/technical` partial (no SHAP): SHAP card → "SHAP 未安装" muted; Gauges still work (AI gauge uses prediction, others use signals[])
- `/data/sentiment` 500 → News section "新闻暂不可用"
- `/data/market` 404 → chart placeholder; everything else OK

## Responsive

- **≥1280px**: as designed
- **1024-1280px**: rail 240px, chart 320px, fonts -1 step
- **768-1024px**: rail collapses to top horizontal drawer; 相关股票 wraps 3×2; News 3 cols
- **<768px**: single-column "Best on desktop" simplified view (header + AI band + chart + CTA only)

## Accessibility

- `<main>` + `<article>` semantic structure
- Gauge: `role="meter"` · `aria-valuenow` · `aria-valuemin=-2` · `aria-valuemax=2` · labeled
- Direction via icon (↗ ↘ →) + color + text (never color-only)
- `aria-live="polite"` on §5 Agent Summary for screen-reader announce
- Keyboard: `/` → search · `T` → train CTA · `B` → buy CTA
- Contrast: zinc-900 on white = 15.8:1 ✓

## Light Theme Token Migration

### Token strategy

Add light tokens alongside existing dark tokens. Root `<html data-theme="light|dark">` drives which CSS vars resolve.

```css
/* tokens.css */
:root[data-theme="light"] {
  --bg-page: #ffffff;
  --bg-surface: #ffffff;
  --bg-sunken: #fafafa;
  --border: #e5e7eb;
  --text-primary: #18181b;
  --text-secondary: #52525b;
  --text-muted: #71717a;
  --accent: #4f46e5;     /* indigo-600 */
  --accent-ink: #ffffff;
  --up: #059669;          /* emerald-600 */
  --down: #e11d48;        /* rose-600 */
  --warn: #d97706;        /* amber-600 */
}

:root[data-theme="dark"] {
  --bg-page: #020617;
  --bg-surface: #18181b;
  --bg-sunken: #27272a;
  --border: #27272a;
  --text-primary: #f4f4f5;
  --text-secondary: #d4d4d8;
  --text-muted: #a1a1aa;
  --accent: #6366f1;
  --up: #10b981;
  --down: #f43f5e;
  --warn: #f59e0b;
}
```

### Per-page theme scope

New component `<ThemeScope value="light">` wraps a route to force its theme regardless of global setting. Dashboard uses it:

```jsx
// DashboardPage.jsx
<ThemeScope value="light">
  <DashboardContent />
</ThemeScope>
```

Other pages stay on default dark until their sub touches them.

### Migration banner (temporary, Sub 1→Sub 7)

`<MigrationBanner>` rendered in AppShell when current route is NOT migrated:
> 🎨 Quant AI 正在换新视觉。以下页面仍是旧样式：`Training` · `Strategy` · ... 近期更新。

Removed after Sub 7.

## Testing Strategy

### Unit (vitest)
- All 17 section components render correctly in light theme
- `<Gauge>`: score [-2..+2] → needle angle (0°..180°) + label; edge cases
- `<AIInsightBand>`: 9 combinations (up/down/neutral × high/med/low)
- `<ThemeScope>`: applies `data-theme` attribute to child subtree
- Watchlist: add / remove / persist localStorage

### Integration (MSW)
- Happy path: all 7 API calls → all 17 sections render with data
- `/agents/technical` 500 → AI Band + Gauges show error, other 12 sections fine
- Partial SHAP: gauges still work, SHAP card fallback
- Empty related stocks: section hidden
- No model history: fallback placeholder
- Model change (dropdown): refetch agentTechnical + gauges re-render

### E2E (Playwright)
- `dashboard.tv.spec.ts`:
  - Load `/dashboard?ticker=AAPL` → visual screenshot matches baseline (light theme)
  - All 17 sections present
  - Click perf pill `1年` → chart range changes
  - Click related stock card → navigates to that ticker
  - Click gauge → tooltip shows underlying signals
  - Add ticker to Watchlist → persists through refresh
  - CTA primary → lands on `/trading` with correct params
  - CTA secondary → lands on `/training` with correct params
- Test matrix: AAPL (happy), TSLA (high vol), V (low vol), ZZZZ (unknown)

### Visual Regression
- Playwright screenshot diff per section, baselines checked in

## Acceptance Criteria

- [ ] All 17 sections render for AAPL with real backend
- [ ] `/agents/technical` single call drives AI Band (3 cards) + 3 Gauges
- [ ] 3 Gauges show live data (AI from prediction; 震荡/MA from signals[])
- [ ] Right rail Watchlist persists to localStorage
- [ ] Light theme tokens added to Tailwind config; Dashboard uses them
- [ ] Other 5 pages unchanged (still dark) and functional
- [ ] Migration banner visible when on non-migrated pages
- [ ] FCP <1.5s on Vercel preview (ticker=AAPL cold load)
- [ ] Bundle impact for `/dashboard` chunk <60KB gzipped
- [ ] Lighthouse a11y ≥95
- [ ] Screenshot regression baseline checked in, passes
- [ ] All unit + integration + E2E tests pass in CI

## Implementation Notes (handoff to writing-plans)

### New files

```
quant-ai-ui/src/
├── theme/
│   ├── ThemeScope.jsx        # wraps children with data-theme
│   ├── tokens.ts              # canonical token map (light + dark)
│   └── tokens.css             # :root[data-theme=*] vars
├── components/layout/
│   ├── TopNavBar.jsx          # TV-style header (new, shared)
│   ├── RightRailWatchlist.jsx # §16
│   ├── MigrationBanner.jsx    # temp banner
│   └── GlobalRagButton.jsx    # §17 (shared)
├── features/dashboard/v2/
│   ├── SymbolHeader.jsx       # §2 + §3
│   ├── SymbolTabs.jsx         # §4
│   ├── AIInsightBand.jsx      # §5 (wraps Prediction/Summary/ShapMini)
│   ├── PredictionCard.jsx     # extracted
│   ├── AgentSummaryCard.jsx   # new
│   ├── ShapMiniCard.jsx       # new (top 3 only)
│   ├── ChartSection.jsx       # §6 + perf pills
│   ├── KeyDataGrid.jsx        # §7 (relabel from §8)
│   ├── AboutBlock.jsx         # §8 (relabel from §9)
│   ├── RelatedStocks.jsx      # §9 (relabel from §10)
│   ├── NewsGrid.jsx           # §10 (relabel from §11)
│   ├── ModelComparison.jsx    # §11 (relabel from §12)
│   ├── GaugesSection.jsx      # §12 (relabel from §13)
│   ├── Gauge.jsx              # reusable SVG semi-circle
│   ├── SeasonalityHeatmap.jsx # §13 (relabel from §14)
│   └── CTARow.jsx             # §14 (relabel from §15)
```

(Section numbers in code comments should match this spec's relabeling: 1-17 as presented in Layout section.)

### Modified files

- `quant-ai-ui/src/pages/DashboardPage.jsx` — rewrite, composed from v2 components, wrapped in `<ThemeScope value="light">`
- `quant-ai-ui/src/AppShell.jsx` — inject TopNavBar + RightRail slot + MigrationBanner + GlobalRagButton
- `quant-ai-ui/tailwind.config.js` — extend theme with CSS-var-driven token refs
- `quant-ai-ui/src/index.css` — import tokens.css
- `quant-ai-ui/src/api/queries.js` — add hooks (useAgentTechnical, useModelsForTicker, useRelatedStocks, useWatchlist, useSeasonalAccuracy)
- `quant-ai-ui/src/api/client.js` — add corresponding client fns (agentTechnical, ragAnswer, getModelsForTicker, getRelatedStocks, getSeasonalAccuracy)

### Backend considerations

- All endpoints exist except historical accuracy (§12 + §14). Options:
  - **MVP**: client-side compute from `/runs` + per-run prediction logs if available; fallback to placeholder
  - **Stretch (Sub 1.5)**: add `GET /models/{id}/accuracy?ticker=X&groupby=month` endpoint (deferred)
- Verify `/agents/technical` response includes `model_id` field (check `app/api/agents.py` response schema)

### Out of this sub, tracked as prerequisites for Sub 2-7
- Each page's `<ThemeScope>` switch to light
- Migration banner route list updates
- Token usage audit (no hardcoded hex in components; must use CSS vars)

---

## Related

- Parent brief: `D:/obsidian vault/01-projects/quant-ai/frontend-productization-brief.md`
- Claude Design prompt: `docs/superpowers/specs/2026-04-19-dashboard-claude-design-prompt.md`
- Wireframe: `.superpowers/brainstorm/1726-1776662616/content/08-dashboard-tv-faithful.html`
- TradingView reference: <https://cn.tradingview.com/symbols/CBOE-VIX/>
- Dependency map: `D:/obsidian vault/01-projects/quant-ai/dependency-map.md`

## Change Log

- 2026-04-19 V1: three-section vertical dark (superseded)
- 2026-04-19 V2: TradingView-faithful 17 sections, light theme, app-wide staged migration (this version)
