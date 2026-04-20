const RANGES = [
  { key: "1D", label: "1天", days: 1 },
  { key: "5D", label: "5天", days: 5 },
  { key: "1M", label: "1月", days: 30 },
  { key: "6M", label: "6月", days: 180 },
  { key: "YTD", label: "YTD", days: null },
  { key: "1Y", label: "1年", days: 365 },
  { key: "5Y", label: "5年", days: 5 * 365 },
  { key: "10Y", label: "10年", days: 10 * 365 },
  { key: "ALL", label: "全部", days: null },
];

function computePerf(candles, range) {
  if (!candles || candles.length === 0) return null;
  const sorted = [...candles].sort((a, b) => new Date(a.date) - new Date(b.date));
  const last = sorted[sorted.length - 1];
  let first;
  if (range.key === "YTD") {
    const yearStart = new Date(new Date(last.date).getFullYear(), 0, 1);
    first = sorted.find((c) => new Date(c.date) >= yearStart) ?? sorted[0];
  } else if (range.key === "ALL") {
    first = sorted[0];
  } else {
    const cutoff = new Date(new Date(last.date).getTime() - range.days * 24 * 3600 * 1000);
    first = sorted.find((c) => new Date(c.date) >= cutoff) ?? sorted[0];
  }
  if (!first?.close || !last?.close) return null;
  return ((last.close - first.close) / first.close) * 100;
}

export function PerformancePills({ candles = [], activeRange = "6M", onChange = () => {} }) {
  return (
    <div className="grid grid-cols-9 gap-0.5 text-[9px] text-center mt-1">
      {RANGES.map((r) => {
        const perf = computePerf(candles, r);
        const color = perf == null ? "text-muted" : perf >= 0 ? "text-up" : "text-down";
        const isActive = r.key === activeRange;
        return (
          <button
            key={r.key}
            onClick={() => onChange(r.key)}
            className={`py-1.5 rounded transition-colors ${isActive ? "bg-surface-muted" : "hover:bg-surface-muted/50"}`}
          >
            <div className="text-muted">{r.label}</div>
            <div className={color}>{perf == null ? "—" : `${perf >= 0 ? "+" : ""}${perf.toFixed(2)}%`}</div>
          </button>
        );
      })}
    </div>
  );
}
