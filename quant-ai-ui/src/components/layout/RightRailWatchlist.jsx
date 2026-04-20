import { useState, useEffect } from "react";
import { Plus, Settings } from "lucide-react";
import { loadWatchlist, addTicker } from "@/lib/watchlist";
import { useAgentTechnical } from "@/api/queries";

const INDICES = [
  { ticker: "VIX", label: "🔶", price: "17.48", change: -2.56 },
  { ticker: "DXY", label: "💵", price: "98.38", change: 0.15 },
  { ticker: "NDQ", label: "📊", price: "26,672", change: 1.29 },
];

function TickerRow({ ticker, price, change, icon, highlighted }) {
  const color = change >= 0 ? "text-up" : "text-down";
  return (
    <div className={`grid grid-cols-[auto_1fr_auto] gap-2 items-center py-1 px-1 text-xs ${highlighted ? "bg-surface border border-surface-border rounded" : ""}`}>
      <span>{icon ?? "·"}</span>
      <span className="text-foreground font-medium">{ticker}</span>
      <span className={`font-mono ${color}`}>{price}</span>
    </div>
  );
}

export function RightRailWatchlist({ currentTicker }) {
  const [holdings, setHoldings] = useState(loadWatchlist());
  const [addingMode, setAddingMode] = useState(false);
  const [newTicker, setNewTicker] = useState("");
  const aiForCurrent = useAgentTechnical(currentTicker);

  useEffect(() => {
    if (currentTicker && !holdings.includes(currentTicker)) {
      setHoldings(addTicker(currentTicker));
    }
  }, [currentTicker, holdings]);

  const handleAdd = (e) => {
    e.preventDefault();
    if (!newTicker.trim()) return;
    setHoldings(addTicker(newTicker.trim().toUpperCase()));
    setNewTicker("");
    setAddingMode(false);
  };

  const aiData = aiForCurrent.data;
  const direction = aiData?.prediction === 1 ? "看涨" : aiData?.prediction === 0 ? "看跌" : "中性";
  const confidence = aiData?.confidence ?? "—";

  return (
    <aside className="w-[280px] bg-surface-muted border-l border-surface-border p-3 sticky top-12 self-start h-[calc(100vh-3rem)] overflow-y-auto">
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-xs font-bold text-foreground">Watchlist</h3>
        <div className="flex gap-1 text-muted">
          <button aria-label="add" onClick={() => setAddingMode(true)} className="hover:text-foreground"><Plus size={14} /></button>
          <button aria-label="settings" className="hover:text-foreground"><Settings size={14} /></button>
        </div>
      </div>

      {addingMode && (
        <form onSubmit={handleAdd} className="mb-2">
          <input
            type="text"
            value={newTicker}
            onChange={(e) => setNewTicker(e.target.value)}
            autoFocus
            placeholder="Ticker (e.g. NVDA)"
            className="w-full px-2 py-1 text-xs bg-surface border border-surface-border rounded focus:outline-none"
          />
        </form>
      )}

      <div className="text-[9px] uppercase text-muted tracking-wider mb-1">▼ INDICES</div>
      <div className="space-y-0.5 mb-3">
        {INDICES.map((i) => (
          <TickerRow key={i.ticker} ticker={i.ticker} price={i.price} change={i.change} icon={i.label} />
        ))}
      </div>

      <div className="text-[9px] uppercase text-muted tracking-wider mb-1">▼ YOUR HOLDINGS</div>
      <div className="space-y-0.5 mb-3">
        {holdings.map((t) => (
          <TickerRow key={t} ticker={t} price="—" change={0} />
        ))}
      </div>

      {currentTicker && (
        <div className="border-t border-surface-border pt-3">
          <div className="text-xs font-bold text-foreground">🎯 {currentTicker} · 当前</div>
          {aiForCurrent.isLoading ? (
            <div className="text-xs text-muted mt-1">Loading...</div>
          ) : (
            <>
              <div className="text-[9px] text-muted mt-2">AI 预测（5 天）</div>
              <div className={`text-xs font-bold ${aiData?.prediction === 1 ? "text-up" : "text-down"}`}>
                {aiData?.prediction === 1 ? "↗" : "↘"} {direction} · {confidence === "high" ? "高置信度" : confidence === "medium" ? "中置信度" : "低置信度"}
              </div>
            </>
          )}
        </div>
      )}
    </aside>
  );
}
