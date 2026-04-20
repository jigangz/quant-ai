export function SymbolHeader({ ticker, name, exchange, price, change = 0, changePct = 0, lastUpdate, modelSource }) {
  const upOrDown = change >= 0 ? "text-up" : "text-down";
  const sign = change >= 0 ? "+" : "";
  return (
    <div className="flex gap-4 mb-4">
      <div className="w-[52px] h-[52px] bg-up rounded-full flex items-center justify-center text-white text-xl font-bold flex-shrink-0">
        {ticker?.[0] ?? "?"}
      </div>
      <div className="flex-1">
        <div className="text-2xl font-bold text-foreground">{name ?? ticker}</div>
        <div className="text-xs text-muted flex items-center gap-2 mt-0.5">
          <span className="bg-surface-muted px-1.5 py-0.5 rounded text-[10px]">{exchange}</span>
        </div>
        <div className="mt-2 flex items-baseline gap-2">
          <span className="text-[28px] font-bold font-mono text-foreground">{price?.toFixed(2)}</span>
          <span className="text-xs text-muted">USD</span>
          <span className={`text-sm ${upOrDown}`}>
            {sign}{change?.toFixed(2)} {sign}{changePct?.toFixed(2)}%
          </span>
        </div>
        {lastUpdate && <div className="text-[10px] text-muted mt-1">在 {lastUpdate} 收盘</div>}
        {modelSource && (
          <div className="text-[10px] text-muted mt-1">
            🔁 Model: <a href={`/training?tab=runs&id=${modelSource.runId}`} className="hover:underline">run #{modelSource.runId} · git {modelSource.gitSha}</a>
          </div>
        )}
      </div>
    </div>
  );
}
