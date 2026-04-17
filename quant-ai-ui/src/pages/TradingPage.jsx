import PageHeader from "../components/PageHeader";
import { Card, CardHeader, CardTitle, CardContent } from "../components/ui/card";
import OrderForm from "../features/trading/OrderForm";
import PortfolioCard from "../features/trading/PortfolioCard";
import OrderList from "../features/trading/OrderList";
import TradeHistory from "../features/trading/TradeHistory";
import { useLivePrices } from "../features/trading/useLivePrices";
import { useLiveStore } from "../stores/liveStore";
import { Badge } from "../components/ui/badge";

export default function TradingPage() {
  useLivePrices();
  const status = useLiveStore((s) => s.connectionStatus);
  const variant = status === "connected" ? "success" : status === "error" ? "destructive" : "warning";

  return (
    <div>
      <PageHeader
        title="Paper Trading"
        subtitle="Place orders, track positions, view trade history"
        actions={<Badge variant={variant}>WS {status}</Badge>}
      />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <PortfolioCard />
          <Card>
            <CardHeader><CardTitle>Open Orders</CardTitle></CardHeader>
            <CardContent><OrderList /></CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle>Recent Trades</CardTitle></CardHeader>
            <CardContent><TradeHistory /></CardContent>
          </Card>
        </div>
        <div>
          <Card>
            <CardHeader><CardTitle>Place Order</CardTitle></CardHeader>
            <CardContent><OrderForm /></CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
