import { useState } from "react";
import { useNavigate } from "react-router-dom";
import PageHeader from "../components/PageHeader";
import { Card, CardContent } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Select } from "../components/ui/select";
import { LoadingOverlay } from "../components/LoadingSpinner";
import ErrorState from "../components/ErrorState";
import { useScreenerTickers } from "../api/queries";
import { fmtPrice, fmtPct, classForDelta } from "../lib/formatters";
import { RefreshCw } from "lucide-react";
import { cn } from "../lib/utils";

const SORT_OPTIONS = [
  { value: "change_pct", label: "Change %" },
  { value: "volume", label: "Volume" },
];

export default function ScreenerPage() {
  const [sortBy, setSortBy] = useState("change_pct");
  const navigate = useNavigate();

  const { data, isLoading, error, refetch, isFetching } = useScreenerTickers();

  const rows = (data || [])
    .filter(Boolean)
    .sort((a, b) =>
      sortBy === "change_pct"
        ? (b.change_pct ?? 0) - (a.change_pct ?? 0)
        : (b.volume ?? 0) - (a.volume ?? 0)
    );

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

      <Card className="relative">
        {isLoading && <LoadingOverlay label="Loading tickers…" />}
        {error && <ErrorState message={error.message} onRetry={() => refetch()} />}
        <CardContent className="p-0">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-surface-border text-muted">
                <th className="text-left px-4 py-3 font-medium">Ticker</th>
                <th className="text-right px-4 py-3 font-medium">Last</th>
                <th className="text-right px-4 py-3 font-medium">Change</th>
                <th className="text-right px-4 py-3 font-medium">Change %</th>
                <th className="text-right px-4 py-3 font-medium">Volume</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr
                  key={row.ticker}
                  onClick={() => navigate(`/dashboard?ticker=${row.ticker}`)}
                  className="border-b border-surface-border hover:bg-surface-hover cursor-pointer transition-colors"
                >
                  <td className="px-4 py-3 font-medium text-foreground">{row.ticker}</td>
                  <td className="px-4 py-3 text-right">${fmtPrice(row.close)}</td>
                  <td className={cn("px-4 py-3 text-right font-medium", classForDelta(row.change ?? 0))}>
                    {row.change >= 0 ? "+" : ""}{fmtPrice(row.change ?? 0)}
                  </td>
                  <td className={cn("px-4 py-3 text-right font-medium", classForDelta(row.change_pct ?? 0))}>
                    {fmtPct(row.change_pct ?? 0)}
                  </td>
                  <td className="px-4 py-3 text-right text-muted">
                    {row.volume?.toLocaleString() ?? "—"}
                  </td>
                </tr>
              ))}
              {!isLoading && rows.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-4 py-8 text-center text-muted">
                    No data available
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  );
}
