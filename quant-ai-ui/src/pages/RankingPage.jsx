import { useState } from "react";
import PageHeader from "../components/PageHeader";
import { Card, CardContent } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { LoadingOverlay } from "../components/LoadingSpinner";
import ErrorState from "../components/ErrorState";
import EmptyState from "../components/EmptyState";
import { useRanking } from "../api/queries";
import BacktestCard from "../features/ranking/BacktestCard";
import { RefreshCw, Trophy, Info } from "lucide-react";
import { cn } from "../lib/utils";

const TOP_OPTIONS = [10, 20, 30, 50];

export default function RankingPage() {
  const [topN, setTopN] = useState(20);
  const { data, isLoading, error, refetch, isFetching } = useRanking(topN);

  const rankings = data?.success ? data.rankings || [] : [];
  const maxScore = rankings.length ? rankings[0].score : 1;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Strength Ranking"
        subtitle={
          data?.as_of
            ? `Cross-sectional Top names as of ${data.as_of} · ${data.scored}/${data.universe_size} scored`
            : "Cross-sectional Top-N strong stocks (xs_strong model)"
        }
        actions={
          <div className="flex items-center gap-2">
            <select
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
              className="h-9 rounded-md border border-surface-border bg-surface-muted px-3 text-sm text-foreground"
            >
              {TOP_OPTIONS.map((n) => (
                <option key={n} value={n}>Top {n}</option>
              ))}
            </select>
            <Button variant="outline" size="sm" onClick={() => refetch()} disabled={isFetching}>
              <RefreshCw className={cn("h-4 w-4", isFetching && "animate-spin")} />
              Refresh
            </Button>
          </div>
        }
      />

      {/* Honest-by-design disclaimer: this is a relative rank, not a forecast. */}
      <div className="flex items-start gap-2 rounded-lg border border-surface-border bg-surface-muted/50 px-4 py-3 text-sm text-muted">
        <Info className="h-4 w-4 mt-0.5 shrink-0 text-accent" />
        <span>
          {data?.score_semantics ||
            "Strength score (0–1) is the model's relative cross-sectional rank — the probability a name is in the next-5d top 30% by return. It is NOT a price target, expected return, or buy recommendation."}
        </span>
      </div>

      {error && <ErrorState message={error.message} onRetry={() => refetch()} />}

      <Card className="relative">
        {isLoading && <LoadingOverlay label="Scoring the universe…" />}
        <CardContent className="p-0">
          {!isLoading && rankings.length === 0 && !error ? (
            <EmptyState
              icon={Trophy}
              title="No ranking available"
              description={
                data && !data.success
                  ? data.error || "No xs_strong model is published yet."
                  : "The strength ranking could not be loaded."
              }
            />
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-border text-muted text-xs uppercase tracking-wide">
                  <th className="text-left font-medium px-4 py-3 w-16">#</th>
                  <th className="text-left font-medium px-4 py-3">Ticker</th>
                  <th className="text-left font-medium px-4 py-3">Strength</th>
                  <th className="text-right font-medium px-4 py-3 w-28">Score</th>
                  <th className="text-right font-medium px-4 py-3 w-28">Percentile</th>
                </tr>
              </thead>
              <tbody>
                {rankings.map((r) => (
                  <tr
                    key={r.ticker}
                    className="border-b border-surface-border/60 hover:bg-surface-hover transition-colors"
                  >
                    <td className="px-4 py-3 text-muted tabular-nums">{r.rank}</td>
                    <td className="px-4 py-3 font-semibold text-foreground">{r.ticker}</td>
                    <td className="px-4 py-3">
                      <div className="h-2 w-full max-w-[220px] rounded-full bg-surface-muted overflow-hidden">
                        <div
                          className="h-full rounded-full bg-accent"
                          style={{ width: `${Math.max(4, (r.score / maxScore) * 100)}%` }}
                        />
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums text-foreground">
                      {r.score.toFixed(3)}
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums text-muted">
                      {r.percentile.toFixed(1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardContent>
      </Card>

      <BacktestCard />
    </div>
  );
}
