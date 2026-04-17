export const fmtPrice = (v, { dp = 2 } = {}) =>
  v == null || isNaN(v) ? "—" : Number(v).toFixed(dp);

export const fmtPct = (v, { dp = 2 } = {}) =>
  v == null || isNaN(v) ? "—" : `${v >= 0 ? "+" : ""}${(Number(v) * 100).toFixed(dp)}%`;

export const fmtVolume = (v) => {
  if (v == null || isNaN(v)) return "—";
  const n = Number(v);
  if (n >= 1e9) return (n / 1e9).toFixed(1) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
};

export const fmtDate = (iso) => {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("en-US", {
    year: "numeric", month: "short", day: "numeric",
  });
};

export const fmtDatetime = (iso) => {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("en-US", {
    month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
};

export const classForDelta = (v) =>
  v > 0 ? "text-up" : v < 0 ? "text-down" : "text-muted";
