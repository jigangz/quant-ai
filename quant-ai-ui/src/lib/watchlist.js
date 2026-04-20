const KEY = "quant-ai:watchlist";
const DEFAULT = ["AAPL", "TSLA", "MSFT", "AMZN"];

export function loadWatchlist() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return DEFAULT;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : DEFAULT;
  } catch {
    return DEFAULT;
  }
}

export function saveWatchlist(tickers) {
  try {
    localStorage.setItem(KEY, JSON.stringify(tickers));
  } catch (e) {
    console.warn("Failed to save watchlist", e);
  }
}

export function addTicker(ticker) {
  const current = loadWatchlist();
  if (current.includes(ticker)) return current;
  const next = [...current, ticker.toUpperCase()];
  saveWatchlist(next);
  return next;
}

export function removeTicker(ticker) {
  const current = loadWatchlist();
  const next = current.filter((t) => t !== ticker);
  saveWatchlist(next);
  return next;
}
