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
export function getMarket(ticker) {
  return request(`/data/market?ticker=${ticker}`);
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
// RAG
// ===================================
export function ragAnswer(query, topK = 5) {
  return request(`/rag/answer`, {
    method: "POST",
    body: JSON.stringify({ query, top_k: topK }),
  });
}
