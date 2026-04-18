import { useState, useMemo } from "react";
import PageHeader from "../components/PageHeader";
import { Card, CardContent } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { LoadingOverlay } from "../components/LoadingSpinner";
import ErrorState from "../components/ErrorState";
import EmptyState from "../components/EmptyState";
import { useScreenerTickers } from "../api/queries";
import ScreenerTable from "../features/screener/ScreenerTable";
import { RefreshCw, BarChart2 } from "lucide-react";
import { cn } from "../lib/utils";

const SORT_OPTIONS = [
  { value: "change_pct", label: "Change %" },
  { value: "volume", label: "Volume" },
];

export default function ScreenerPage() {
  const [sortBy, setSortBy] = useState("change_pct");

  const { data, isLoading, error, refetch, isFetching } = useScreenerTickers();

  // Transform [{ticker, data: {rows: [...]}}] → [{ticker, close, change, change_pct, volume}]
  const rows = useMemo(() => {
    if (!data) return [];
    return data
      .map(({ ticker, data: tickerData }) => {
        const ohlcv = tickerData?.rows || [];
        if (ohlcv.length === 0) return null;
        const last = ohlcv[ohlcv.length - 1];
        const prev = ohlcv.length >= 2 ? ohlcv[ohlcv.length - 2] : last;
        const change = last.close - prev.close;
        const change_pct = prev.close ? change / prev.close : 0;
        return {
          ticker,
          close: last.close,
          change,
          change_pct,
          volume: last.volume,
        };
      })
      .filter(Boolean);
  }, [data]);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Stock Screener"
        subtitle="Real-time market overview for top tickers"
        actions={
          <div className="flex items-center gap-2">
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="h-9 rounded-md border border-surface-border bg-surface-muted px-3 text-sm text-foreground"
            >
              {SORT_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
            <Button variant="outline" size="sm" onClick={() => refetch()} disabled={isFetching}>
              <RefreshCw className={cn("h-4 w-4", isFetching && "animate-spin")} />
              Refresh
            </Button>
          </div>
        }
      />

      {error && <ErrorState message={error.message} onRetry={() => refetch()} />}

      <Card className="relative">
        {isLoading && <LoadingOverlay label="Loading tickers…" />}
        <CardContent className="p-0">
          {!isLoading && rows.length === 0 && !error ? (
            <EmptyState
              icon={BarChart2}
              title="No tickers available"
              description="Market data could not be loaded."
            />
          ) : (
            <ScreenerTable rows={rows} sortBy={sortBy} />
          )}
        </CardContent>
      </Card>
    </div>
  );
}
