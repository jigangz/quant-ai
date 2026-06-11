import { useQuery } from "@tanstack/react-query";
import { agentSummary } from "@/api/client";

/** Portfolio-wide AI summary over POST /agents/summary (P6 Sub 3). */
export function usePortfolioSummary(tickers, opts = {}) {
  return useQuery({
    // Sorted key: ["AAPL","MSFT"] and ["MSFT","AAPL"] are the same portfolio
    queryKey: ["portfolio-summary", [...tickers].sort().join(",")],
    queryFn: () => agentSummary({ tickers }),
    enabled: tickers.length > 0,
    staleTime: 60_000,
    ...opts,
  });
}
