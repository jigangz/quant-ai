# Dashboard · Claude Design High-Fidelity Prompt V2

> **Usage**: Go to <https://claude.ai/design>, select "High fidelity", name the project "Quant AI Dashboard · TV-style", and paste the block below as the design brief.
>
> **V2 supersedes V1**. V2 target: TradingView-faithful layout + light theme + 3 gauges + right-rail watchlist + all TV sections.

---

Design a high-fidelity mockup for **Quant AI · Dashboard page** — an AI-powered stock research hub for retail investors learning quantitative trading.

The layout is a **faithful clone of the TradingView symbol page** (reference: <https://cn.tradingview.com/symbols/CBOE-VIX/>), with our AI differentiation layered in as an "AI Insight Band" and an "AI Gauge". Chinese UI (Simplified).

## Theme: LIGHT (matching TradingView default)

Colors:
- Page bg: **#ffffff** (white)
- Surfaces (cards): **#ffffff** with **#e5e7eb** 1px borders
- Muted surface (right rail, highlighted pill): **#fafafa**
- Text primary: **#18181b** (zinc-900)
- Text secondary: **#52525b** (zinc-600)
- Text muted: **#71717a** (zinc-500)
- Accent (interactive, primary CTA): **#4f46e5** (indigo-600)
- Up / buy / positive: **#059669** (emerald-600)
- Down / sell / negative: **#e11d48** (rose-600)
- Warning / medium confidence: **#d97706** (amber-600)

Typography:
- Body: Inter, 14–16px
- Prices & IDs: JetBrains Mono
- Headings: Inter, semibold

Cards: 6–8px radius · subtle 1px zinc-200 border · 12–16px internal padding · no shadow by default; shadow-sm on hover.

## Overall Structure

Main layout: 2-column grid
- Left sidebar (64px, icon-only, inherited): Screener · Dashboard (active) · Training · Strategy · Portfolio · Paper Trading · Settings
- Main content (1fr, max-width 1200px): vertical sections
- Right rail watchlist (280px fixed, sticky): separate column

Top nav bar above everything (white, 1px zinc-200 border-bottom, 48px tall):
- Left: "Quant AI" logo (indigo-600 bold, 18px)
- Middle-left: nav items "市场 · 研究 · 模型 · 更多" (13px zinc-600)
- Center: search input (240px wide, zinc-100 fill, placeholder "🔍 搜索 (Ctrl+K)")
- Right: "升级" pill button (indigo-600 filled) + circular avatar (32px)

## Content Sections (left column, top to bottom)

### §1 Breadcrumb
"市场 / 美国 / 股票 / AAPL" — zinc-500, 12px, 8px margin-bottom

### §2 Symbol Header
Horizontal flex (16px gap):
- Circle logo 52px — emerald-600 fill with white Apple icon (🍎)
- Stack:
  - "Apple Inc." — 24px bold zinc-900
  - "AAPL · [NASDAQ]" — ticker 11px zinc-500 + NASDAQ chip (zinc-100 bg / zinc-200 border / 5px padding / 9px uppercase)
  - 8px gap
  - Price row: "270.23" (28px bold JetBrains Mono zinc-900) · "USD" (11px zinc-500) · "+2.59 +2.59%" (13px emerald-600)
  - "在 4月19日 GMT-7 13:15 收盘" (10px zinc-400)

### §3 Tab Navigation
Horizontal tabs, 20px gap, 1px zinc-200 border-bottom, 10px vertical padding:
- "概览" (active: 2px zinc-900 bottom border, bold)
- "新闻" · "社区" · "技术指标" · "模型历史" · "预测记录" (zinc-500, 13px)

### §4 AI Insight Band (our signature insertion — 3 cards)
Grid 1fr / 1.5fr / 1fr, 10px gap, 16px vertical margin.

**Card 1 — 🤖 AI 预测**:
- White bg with subtle linear-gradient(135deg, rgba(5,150,105,0.08), #ffffff)
- 1px zinc-200 border, 6px radius, 10px padding
- "🤖 AI 预测" label (9px uppercase tracked zinc-500)
- Direction: "↗ 看涨" (20px bold emerald-600) — use arrow icon
- Chip row: "置信度 高" (emerald-100 bg, emerald-700 text, 1px 5px padding, 2px radius) · "5 天" (zinc-100 bg, zinc-700 text)
- "prob_up 0.68" (10px mono zinc-700)

**Card 2 — ⚡ 为什么这么说** (Agent Summary):
- White bg, 3px indigo-600 left border accent, 1px zinc-200 border (top/right/bottom), 6px radius, 10px padding
- "⚡ 为什么这么说" label (9px indigo-600 uppercase tracked)
- Italic body (10.5px zinc-900, line-height 1.5):
  > "5 天方向上涨。主因 RSI 超卖反弹 + MA 金叉 + 正面新闻情绪。Top 驱动：RSI 14 +28%、MA 10 +21%。"

**Card 3 — 📊 SHAP Top 3**:
- Plain white card, 1px zinc-200 border
- "📊 SHAP Top 3" label (9px uppercase tracked zinc-500)
- 3 horizontal bars (stack, 10px font):
  - `RSI   ████████████ (80% emerald-600) +28%`
  - `MA 10 █████████ (60%) +21%`
  - `情绪   ██████ (40%) +12%`
- Bar track zinc-100 · fill emerald-600 · 10px tall · 2px radius · label 40px left · value mono 30px right

### §5 图表 (Chart)
Full-width card, 1px zinc-200 border, 6px radius.
- Header: "图表 ›" (14px bold, clickable) on left · export icons (📷 camera, `</>` code) on right
- Chart body: candlestick with area fill under close line (emerald-600 line, rgba(5,150,105,0.08) area), **380px tall**, zinc-200 grid lines
- Price axis on right (11px mono zinc-500)
- Crosshair on hover

### §6 Performance Pills
9-column row, 2px gap, directly below chart (same card or separate strip). Each cell:
- Top: label (10px zinc-500) — "1天 · 5天 · 1月 · 6月 · YTD · 1年 · 5年 · 10年 · 全部"
- Bottom: % colored by sign (12px mono)
- **Active range** (e.g. "6月"): zinc-100 background + 4px rounded

Example values: 1天 +2.59% · 5天 +5.20% · 1月 +8.10% · 6月 +25.30% · YTD +15.30% · 1年 +42.00% · 5年 +180% · 10年 +320% · 全部 +2100%

### §7 关键数据点
Title "关键数据点" (14px bold) · 4-column grid (12px gap):
- 成交量   →  `48.2M`
- 前一次收盘 → `267.64`
- 开盘价    → `268.10`
- 当日价格范围 → `267.88 — 272.45`

Each item: label 10px zinc-500 above value 14px JetBrains Mono zinc-900

### §8 描述 (About)
Plain paragraph, 10.5px zinc-600, line-height 1.6, 3 lines:
> Apple Inc. 设计、制造和销售智能手机、个人电脑、平板电脑、可穿戴设备和配件。AI 模型基于过去 2 年日线数据训练，使用技术指标（RSI/MACD/Bollinger）、动量、波动率、成交量、情绪、新闻 6 组特征。当前使用 XGBoost · run #42 · git abc1234（2026-04-15 训练，AUC 0.62）。

### §9 相关股票
Title "相关股票" (14px bold) · subtitle "同行业 + AI 预测信号" (10px zinc-500).

6-card equal-width grid, 8px gap, each card:
- 1px zinc-200 border · 6px radius · 8px padding
- Ticker (11px bold zinc-900)
- Company name (9px zinc-500)
- Price (10px mono zinc-900)
- AI signal row: "🤖 看涨 · 高" (9px emerald-600) / "🤖 看跌 · 低" (rose-600) / "🤖 中性" (zinc-500)

Sample: MSFT (Microsoft, 465.12, 看涨 高) · GOOGL (Alphabet, 185.40, 看涨 中) · AMZN (Amazon, 215.33, 看跌 低) · NVDA (Nvidia, 148.62, 看涨 高) · META (Meta, 615.90, 中性) · TSLA (Tesla, 400.62, 看涨 高)

### §10 新闻 (News grid)
Title "新闻 ›" (14px bold, clickable to News tab).

4-column grid, 8 headlines (2 rows × 4 columns), 10px gap, 10px body font:
- Top row of each tile: "前天 · Reuters" (9px zinc-500)
- Body: 2-line headline max (zinc-900)

Sample headlines:
- "Apple beats Q2 earnings, raises guidance for FY"
- "iPhone sales surge in China market 25% YoY"
- "Wall Street analysts upgrade AAPL to Strong Buy"
- "Tim Cook signals AI product roadmap expansion"
- "Apple's services revenue hits record high"
- "New iPhone launch expected in Q4"
- "Apple partners with major automaker for EV software"
- "Supply chain improvements boost margins"

### §11 历史模型对此股的预测 (replaces TV's 观点 community section)
Title "历史模型对此股的预测 ›" · subtitle "你训练过的 / 线上其他模型对 AAPL 的预测对比" (10px zinc-500).

4 cards in a row, 8px gap, each:
- Top 50px strip: soft pastel gradient sparkline background (mixed pastel zinc/emerald/amber)
- Body (8px padding):
  - Model name bold 10.5px — e.g. "XGBoost v2 · 当前 ⭐"
  - Metrics line 9px zinc-500 — "AUC 0.62 · run #42"
  - Accuracy line 9px colored by performance — "✓ 预测准确率 64%" (emerald-600) or "准确率 51%" (rose-600)

Sample: XGBoost v2 (⭐ current) · LightGBM v1 · Ensemble · Logistic (baseline)

### §12 技术指标 · 3 Gauges (hero AI visualization)
Title "技术指标 ›" (14px bold) · subtitle "总结指标的建议" (10px zinc-500).

Grid of 3 equal columns, center-aligned, 16px gap.

Each gauge cell (center-aligned):
- Top label (10px zinc-500, 6px margin-bottom):
  - Left: "震荡指标 (RSI/MACD)"
  - Middle: **"🤖 AI 模型总结"** (indigo-600 tint, slight emphasis)
  - Right: "移动平均线"
- SVG semi-circle arc 160px wide:
  - Background arc: 8px stroke zinc-200, path `M 10 55 A 40 40 0 0 1 90 55`
  - Filled arc: same path but stopping at needle angle, stroke = colored by side
  - Divider ticks at band boundaries (5 bands: 强烈卖出 / 卖出 / 中立 / 买入 / 强烈买入)
  - Band colors (left to right): rose-600 → rose-300 → zinc-300 → emerald-300 → emerald-600
  - Needle: 1.5px stroke zinc-900, from center (50, 55) to tip at calculated angle
  - Center dot: 3px zinc-900 circle
- Below arc (14px bold):
  - Left gauge: "买入" (emerald-600)
  - Middle gauge: "强烈买入" (emerald-600 + slightly larger)
  - Right gauge: "买入" (emerald-600)

Middle gauge has subtle indigo-50 background tint behind its cell to mark it as our AI signature.

### §13 季节性 (Seasonality)
Title "季节性 ›" (14px bold) · subtitle "过去模型在这个月份的预测准确率" (10px zinc-500).

12-column horizontal grid, 2px gap. Each cell (10px vertical padding):
- Month label (9px zinc-800)
- Accuracy % (9px, colored)

Color bands:
- >60% → emerald-100 bg, emerald-700 text
- 50–60% → amber-100 bg, amber-700 text
- <50% → rose-100 bg, rose-700 text

Sample values (month: %):
1月 68% · 2月 62% · 3月 48% · 4月 72% · 5月 55% · 6月 64% · 7月 66% · 8月 50% · 9月 61% · 10月 63% · 11月 56% · 12月 69%

### §14 CTA Row
2 equal buttons, 10px gap, 40px tall:
- Primary (left): indigo-600 filled · white bold 12px · "🛒 基于此信号纸上下单"
- Secondary (right): white bg + 1px zinc-300 border · zinc-900 12px · "🧪 训练新模型 (AAPL 专属)"

## Right Rail Watchlist (280px sticky column)

`#fafafa` background · 1px zinc-200 left border · 12px padding. Sticks to viewport top on scroll.

### Header
"Watchlist" (12px bold zinc-900) on left + "+ ⚙" icons on right (zinc-500)

### ▼ INDICES (9px uppercase zinc-500 header)
3 rows, 4px gap, each highlighted for current (10px padding 4px, white bg 1px zinc-200 border 3px radius for active):
- 🔶 VIX — 17.48 (rose-600 mono 10px, right-aligned)
- 💵 DXY — 98.378 (emerald-600)
- 📊 NDQ — 26,672 (emerald-600)

Each row: icon + ticker + right-aligned price

### ▼ YOUR HOLDINGS
4 rows:
- 🍎 AAPL — 270.23 (emerald)
- 🚗 TSLA — 400.62 (emerald)
- 💻 MSFT — 465.12 (emerald)
- 🛒 AMZN — 215.33 (rose)

### Current ticker card (separated by 1px zinc-200 border-top, 12px padding-top)
- "🎯 AAPL · 当前" (11px bold zinc-900)
- "270.23" (18px bold mono zinc-900)
- "+2.59%" (11px emerald-600)
- 8px gap
- "AI 预测（5天）" (9px zinc-500)
- "↗ 看涨 · 高置信度" (11px bold emerald-600)

### 表现 mini-cards
"表现" label (10px bold zinc-900), then 3-column grid (4px gap):
- 1W · "+5.2%" — emerald-100 bg, emerald-700 text, 6px padding, centered
- 1M · "+8.1%" — same
- YTD · "+15.3%" — same

Each tile: % bold 9px on top, period label 9px below.

## Floating RAG Button (fixed, above right rail)

Fixed position: 24px from bottom, 304px from right (to avoid right rail). 56px circle, indigo-600 bg, white "❓" centered. shadow-lg. On hover: scale 1.05, shadow-xl.

## Sample Data (use these for realism)

AAPL at 270.23, +2.59 (+2.59%). Bullish high-confidence prediction prob_up=0.68. Realistic Apple news headlines (not placeholder). Emerald-dominant SHAP bars. All 3 gauges lean toward buy side (middle AI gauge to 强烈买入 at ~75% arc fill).

## Produce

1. **Main mockup** — Dashboard 概览 tab, all 14 content sections + right rail + floating button, desktop at 1440px wide, full-fidelity light theme
2. **Loading state variant** (smaller, below) — same layout but AI Insight Band (§4) and Gauges (§12) show "Computing…" skeleton state
3. **Tablet layout** (smaller, below) — at 900px wide, right rail collapses to horizontal drawer at top, grids reflow (News 3-col, Related 3×2)

## Do Not

- Use dark theme
- Drop the right rail
- Drop any of the 14 content sections
- Invent features not in the list (no options chain, no watchlist alerts, no deep financials)
- Drift from the exact hex values — stay on the specified tokens
