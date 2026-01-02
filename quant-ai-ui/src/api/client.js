const BASE = import.meta.env.VITE_API_BASE;

/**
 *  fetch 
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

/**
 * ===== Market data =====
 */
export function getMarket(ticker) {
  return request(`/data/market?ticker=${ticker}`);
}

/**
 * ===== Prediction =====
 * v1  stub / baseline
 * v2  multi-horizon
 */
export function predict(payload) {
  return request(`/predict`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * ===== SHAP explain =====
 */
export function explain(ticker) {
  return request(`/explain?ticker=${ticker}`);
}

/**
 * ===== Vector search =====
 */
export function search(q) {
  return request(`/search?q=${encodeURIComponent(q)}`);
}
