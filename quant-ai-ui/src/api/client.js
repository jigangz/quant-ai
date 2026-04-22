const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

/**
 * Base fetch wrapper
 */
async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
    },
    ...options,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }

  return res.json();
}

// ===================================
// Market Data
// ===================================
export function getMarket(tickerOrOpts, lookback) {
  let ticker, qs;
  if (tickerOrOpts && typeof tickerOrOpts === "object") {
    const { ticker: t, period, lookback: lb } = tickerOrOpts;
    ticker = t;
    qs = period ? `&period=${encodeURIComponent(period)}` : lb ? `&lookback=${lb}` : "";
  } else {
    ticker = tickerOrOpts;
    qs = lookback ? `&lookback=${lookback}` : "";
  }
  return request(`/data/market?ticker=${encodeURIComponent(ticker)}${qs}`);
}

// ===================================
// Prediction
// ===================================
export function predict(payload) {
  return request(`/predict`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// V4 Pivot · Phase 2 · Volatility Prediction
export function predictVolatility({ ticker, model_id = null, horizon_days = 5 }) {
  return request(`/predict/volatility`, {
    method: "POST",
    body: JSON.stringify({ ticker, model_id, horizon_days }),
  });
}

// ===================================
// SHAP Explain
// ===================================
export function explain(ticker) {
  return request(`/explain?ticker=${ticker}`);
}

// ===================================
// Search
// ===================================
export function search(q) {
  return request(`/search?q=${encodeURIComponent(q)}`);
}

// ===================================
// Training
// ===================================
export function train(payload) {
  return request(`/train`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getRunStatus(runId) {
  return request(`/runs/${runId}`);
}

export function listRuns(limit = 20) {
  return request(`/runs?limit=${limit}`);
}

// ===================================
// Models
// ===================================
export function listModels(status = "active", limit = 50) {
  return request(`/models?status=${status}&limit=${limit}`);
}

export function getModel(modelId) {
  return request(`/models/${modelId}`);
}

export function promoteModel(modelId) {
  return request(`/models/${modelId}/promote`, {
    method: "POST",
  });
}

export function demoteModel() {
  return request(`/models/promoted`, {
    method: "DELETE",
  });
}

export function getPromotedModel() {
  return request(`/models/promoted`);
}

export function listModelTypes() {
  return request(`/models/types`);
}

// ===================================
// Features
// ===================================
export function listFeatureGroups() {
  return request(`/features/groups`);
}

// ===================================
// Strategies
// ===================================
export function listStrategies() {
  return request(`/api/strategies`);
}

export function getStrategy(name) {
  return request(`/api/strategies/${name}`);
}

export function generateSignals(name, payload) {
  return request(`/api/strategies/${name}/signals`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function runStrategyBacktest(name, payload) {
  return request(`/api/strategies/${name}/backtest`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// ===================================
// Paper Trading
// ===================================
export function placeOrder(payload) {
  return request(`/api/trading/orders`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function listOrders(status = "all", limit = 50) {
  return request(`/api/trading/orders?status=${status}&limit=${limit}`);
}

export function cancelOrder(orderId) {
  return request(`/api/trading/orders/${orderId}`, { method: "DELETE" });
}

export function getPortfolio() {
  return request(`/api/trading/portfolio`);
}

export function resetPortfolio() {
  return request(`/api/trading/portfolio/reset`, { method: "POST" });
}

export function getTrades(limit = 50) {
  return request(`/api/trading/trades?limit=${limit}`);
}

// ===================================
// Screener (uses market data)
// ===================================
export function getMarketMulti(tickers) {
  return Promise.all(tickers.map((t) => getMarket(t).catch(() => null)));
}

// === Optimization ===

export async function optimizeModel(request) {
  const res = await fetch(`${BASE}/api/optimize/model`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function optimizeStrategy(request) {
  const res = await fetch(`${BASE}/api/optimize/strategy`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ===================================
// Dashboard V2 — Agents + RAG
// ===================================
function post(path, body) {
  return request(path, { method: "POST", body: JSON.stringify(body) });
}

export function agentTechnical({ ticker, model_id = null, include_shap = true, top_features = 5 }) {
  return post("/agents/technical", { ticker, model_id, include_shap, top_features });
}

export function agentSummary({ tickers, model_id = null }) {
  return post("/agents/summary", { tickers, model_id });
}

export function ragAnswer({ query, top_k = 5 }) {
  return post("/rag/answer", { query, top_k });
}

// G3 is now implemented backend-side (V4 P2). Pass ticker + optional label_type
// as query params; backend filters. Falls back to client-side filter if the
// backend is still on pre-V4 schema.
export function getModelsForTicker(
  ticker,
  { status = "active", labelType = null } = {},
) {
  const params = new URLSearchParams({ status, limit: "50", ticker });
  if (labelType) params.set("label_type", labelType);
  return request(`/models?${params.toString()}`).then((res) => {
    const models = Array.isArray(res) ? res : (res.models ?? []);
    // Defensive: pre-V4 backends ignore `ticker`, so filter locally too.
    return models.filter((m) => (m.tickers ?? []).includes(ticker));
  });
}

// G2 (docs/backend-gaps.md): no /tickers/{ticker}/related endpoint.
// MVP: static sector-peer map until backend exposes a sector API.
const SECTOR_PEERS = {
  AAPL: ["MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
  MSFT: ["AAPL", "GOOGL", "AMZN", "NVDA", "META", "CRM"],
  GOOGL: ["AAPL", "MSFT", "AMZN", "META", "NVDA", "NFLX"],
  AMZN: ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "WMT"],
  NVDA: ["AAPL", "MSFT", "AMD", "GOOGL", "META", "TSM"],
  TSLA: ["F", "GM", "RIVN", "NIO", "LCID", "XPEV"],
  META: ["GOOGL", "AAPL", "AMZN", "SNAP", "PINS", "NFLX"],
  JPM: ["BAC", "WFC", "C", "GS", "MS", "USB"],
  V: ["MA", "AXP", "PYPL", "SQ", "DFS", "COF"],
  WMT: ["TGT", "COST", "AMZN", "KR", "HD", "LOW"],
};

export function getRelatedStocks(ticker, { limit = 6 } = {}) {
  const peers = SECTOR_PEERS[ticker] ?? [];
  return Promise.resolve(peers.slice(0, limit));
}

// G1 (docs/backend-gaps.md): no /models/{id}/accuracy endpoint yet.
// MVP: return null placeholder; UI shows "数据积累中".
export function getSeasonalAccuracy() {
  return Promise.resolve({ monthly: null, overall: null });
}

export function getSentiment({ ticker, days = 30 }) {
  return request(`/data/sentiment?ticker=${encodeURIComponent(ticker)}&days=${days}`);
}

