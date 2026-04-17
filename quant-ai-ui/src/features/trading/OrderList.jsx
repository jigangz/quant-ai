import { Button } from "../../components/ui/button";
import { Badge } from "../../components/ui/badge";
import { useOrders, useCancelOrder } from "../../api/queries";
import { fmtPrice, fmtDatetime } from "../../lib/formatters";

export default function OrderList() {
  const { data, isLoading } = useOrders("all");
  const cancel = useCancelOrder();
  if (isLoading) return <div className="text-sm text-muted p-4">Loading orders...</div>;
  const orders = (data?.orders || data || []).slice(0, 20);
  if (orders.length === 0) return <div className="text-sm text-muted p-4">No orders yet.</div>;

  return (
    <div className="overflow-hidden rounded-xl border border-surface-border bg-surface-card">
      <table className="w-full text-sm">
        <thead className="bg-surface-muted text-xs uppercase text-muted">
          <tr>
            <th className="px-4 py-2 text-left">Ticker</th>
            <th className="px-4 py-2 text-left">Side</th>
            <th className="px-4 py-2 text-right">Qty</th>
            <th className="px-4 py-2 text-right">Price</th>
            <th className="px-4 py-2 text-left">Status</th>
            <th className="px-4 py-2 text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          {orders.map((o) => (
            <tr key={o.order_id || o.id} className="border-t border-surface-border">
              <td className="px-4 py-2 font-semibold">{o.ticker}</td>
              <td className="px-4 py-2">
                <Badge variant={o.side === "buy" ? "success" : "destructive"}>{o.side}</Badge>
              </td>
              <td className="px-4 py-2 text-right font-mono">{o.quantity}</td>
              <td className="px-4 py-2 text-right font-mono">
                {o.limit_price ? `$${fmtPrice(o.limit_price)}` : "market"}
              </td>
              <td className="px-4 py-2">
                <Badge variant="outline">{o.status}</Badge>
              </td>
              <td className="px-4 py-2 text-right">
                {(o.status === "pending" || o.status === "open") && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => cancel.mutate(o.order_id || o.id)}
                  >
                    Cancel
                  </Button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
