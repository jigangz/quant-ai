import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";
import { usePortfolio, useResetPortfolio } from "../../api/queries";
import { LoadingOverlay } from "../../components/LoadingSpinner";
import ErrorState from "../../components/ErrorState";
import ConfirmDialog from "../../components/ConfirmDialog";
import { Button } from "../../components/ui/button";
import { fmtPrice, fmtPct, classForDelta } from "../../lib/formatters";
import { RefreshCw } from "lucide-react";

export default function PortfolioCard() {
  const { data, isLoading, error, refetch } = usePortfolio();
  const reset = useResetPortfolio();

  if (isLoading) return <LoadingOverlay label="Loading portfolio..." />;
  if (error) return <ErrorState error={error} onRetry={refetch} />;

  const cash = data?.cash ?? 0;
  const equity = data?.total_equity ?? data?.equity ?? 0;
  const pnl = data?.day_pnl ?? 0;
  const pnlPct = data?.day_pnl_pct ?? 0;
  const positions = data?.positions || [];

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between pb-3">
        <CardTitle>Portfolio</CardTitle>
        <ConfirmDialog
          trigger={<Button size="sm" variant="outline"><RefreshCw className="h-3 w-3" /> Reset</Button>}
          title="Reset portfolio?"
          description="All positions and trade history will be cleared. This cannot be undone."
          confirmLabel="Reset"
          destructive
          onConfirm={() => reset.mutate()}
        />
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <p className="text-xs text-muted uppercase">Cash</p>
            <p className="text-lg font-mono font-semibold">${fmtPrice(cash)}</p>
          </div>
          <div>
            <p className="text-xs text-muted uppercase">Equity</p>
            <p className="text-lg font-mono font-semibold">${fmtPrice(equity)}</p>
          </div>
          <div>
            <p className="text-xs text-muted uppercase">Day P&L</p>
            <p className={`text-lg font-mono font-semibold ${classForDelta(pnl)}`}>
              {pnl >= 0 ? "+" : ""}${fmtPrice(pnl)} ({fmtPct(pnlPct)})
            </p>
          </div>
          <div>
            <p className="text-xs text-muted uppercase">Positions</p>
            <p className="text-lg font-mono font-semibold">{positions.length}</p>
          </div>
        </div>
        {positions.length > 0 && (
          <div className="border-t border-surface-border pt-3 space-y-1">
            {positions.map((p) => (
              <div key={p.ticker} className="flex justify-between text-sm">
                <span className="font-semibold">{p.ticker}</span>
                <span className="text-muted">{p.quantity} @ ${fmtPrice(p.avg_cost)}</span>
                <span className={`font-mono ${classForDelta(p.unrealized_pnl)}`}>{fmtPct(p.unrealized_pnl_pct)}</span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
