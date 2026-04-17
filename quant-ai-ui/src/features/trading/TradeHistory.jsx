import { useTrades } from "../../api/queries";
import { Badge } from "../../components/ui/badge";
import { fmtPrice, fmtDatetime } from "../../lib/formatters";

export default function TradeHistory() {
  const { data, isLoading } = useTrades(20);
  if (isLoading) return <div className="text-sm text-muted p-4">Loading...</div>;
  const trades = data?.trades || data || [];
  if (trades.length === 0) return <div className="text-sm text-muted p-4">No trades yet.</div>;

  return (
    <div className="overflow-hidden rounded-xl border border-surface-border bg-surface-card">
      <table className="w-full text-sm">
        <thead className="bg-surface-muted text-xs uppercase text-muted">
          <tr>
            <th className="px-4 py-2 text-left">Time</th>
            <th className="px-4 py-2 text-left">Ticker</th>
            <th className="px-4 py-2 text-left">Side</th>
            <th className="px-4 py-2 text-right">Qty</th>
            <th className="px-4 py-2 text-right">Price</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t) => (
            <tr key={t.trade_id || t.id} className="border-t border-surface-border">
              <td className="px-4 py-2 text-muted text-xs">{fmtDatetime(t.timestamp || t.executed_at)}</td>
              <td className="px-4 py-2 font-semibold">{t.ticker}</td>
              <td className="px-4 py-2">
                <Badge variant={t.side === "buy" ? "success" : "destructive"}>{t.side}</Badge>
              </td>
              <td className="px-4 py-2 text-right font-mono">{t.quantity}</td>
              <td className="px-4 py-2 text-right font-mono">${fmtPrice(t.price)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
