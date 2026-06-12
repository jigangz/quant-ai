import { Link } from "react-router-dom";

function signalLabel(signal) {
  if (!signal) return { text: "—", color: "text-muted" };
  const dir = signal.direction === "bullish" ? "Bullish" : signal.direction === "bearish" ? "Bearish" : "Neutral";
  const conf = signal.confidence === "high" ? "High" : signal.confidence === "medium" ? "Med" : signal.confidence === "low" ? "Low" : "";
  const color = signal.direction === "bullish" ? "text-up" : signal.direction === "bearish" ? "text-down" : "text-muted";
  return { text: `🤖 ${dir}${conf ? " · " + conf : ""}`, color };
}

export function RelatedStocks({ peers = [] }) {
  if (!peers.length) return null;
  return (
    <section className="mb-4">
      <h3 className="text-sm font-bold text-foreground">Related Stocks</h3>
      <p className="text-[10px] text-muted mb-2">Sector peers + AI prediction signals</p>
      <div className="grid grid-cols-6 gap-2">
        {peers.map((p) => {
          const sig = signalLabel(p.signal);
          return (
            <Link
              key={p.ticker}
              to={`/dashboard?ticker=${p.ticker}`}
              className="border border-surface-border rounded-md p-2 hover:bg-surface-muted transition-colors"
            >
              <div className="text-xs font-bold text-foreground">{p.ticker}</div>
              <div className="text-[9px] text-muted">{p.name}</div>
              <div className="font-mono text-[10px] mt-1 text-foreground">${p.price?.toFixed(2) ?? "—"}</div>
              <div className={`text-[9px] mt-1 ${sig.color}`}>{sig.text}</div>
            </Link>
          );
        })}
      </div>
    </section>
  );
}
