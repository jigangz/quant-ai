import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { X, Plus, Briefcase } from "lucide-react";
import PortfolioSummary from "@/features/portfolio/PortfolioSummary";
import { usePortfolioSummary } from "@/features/portfolio/portfolioQueries";
import { loadWatchlist } from "@/lib/watchlist";
import { LoadingOverlay } from "@/components/LoadingSpinner";
import ErrorState from "@/components/ErrorState";
import EmptyState from "@/components/EmptyState";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

export default function PortfolioPage() {
  const navigate = useNavigate();
  const [tickers, setTickers] = useState(() => loadWatchlist());
  const [draft, setDraft] = useState("");
  const query = usePortfolioSummary(tickers);

  const addTicker = (e) => {
    e.preventDefault();
    const t = draft.trim().toUpperCase();
    if (t && !tickers.includes(t)) setTickers((prev) => [...prev, t]);
    setDraft("");
  };

  const removeTicker = (t) => setTickers((prev) => prev.filter((x) => x !== t));

  return (
    <div className="p-6 space-y-4 max-w-7xl mx-auto">
      <header>
        <h1 className="text-2xl font-semibold text-foreground">Portfolio</h1>
        <p className="text-sm text-muted">
          AI signal distribution across your watchlist — bullish/bearish split, per-ticker
          probability, one click into the full Dashboard analysis.
        </p>
      </header>

      <div className="rounded-lg border border-surface-border bg-surface-card p-4 space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          {tickers.map((t) => (
            <span
              key={t}
              className="inline-flex items-center gap-1 rounded-full bg-surface-muted px-2.5 py-1 text-xs font-medium text-foreground"
            >
              {t}
              <button
                type="button"
                aria-label={`Remove ${t}`}
                onClick={() => removeTicker(t)}
                className="text-muted hover:text-down"
              >
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
          {tickers.length === 0 && (
            <span className="text-xs text-muted">No tickers — add one below.</span>
          )}
        </div>
        <form onSubmit={addTicker} className="flex items-center gap-2">
          <Input
            value={draft}
            onChange={(e) => setDraft(e.target.value.toUpperCase())}
            placeholder="Add ticker (e.g. NVDA)"
            aria-label="Add ticker"
            className="w-44"
          />
          <Button type="submit" size="sm" variant="outline">
            <Plus className="h-4 w-4 mr-1" /> Add
          </Button>
          <span className="text-[11px] text-muted">Seeded from your Dashboard watchlist.</span>
        </form>
      </div>

      {tickers.length === 0 ? (
        <EmptyState
          icon={Briefcase}
          title="Build a portfolio to analyze"
          description="Add a few tickers above — the AI summarizes bullish/bearish signals across all of them at once."
        />
      ) : query.isLoading ? (
        <LoadingOverlay label="Scoring your portfolio..." />
      ) : query.isError ? (
        <ErrorState error={query.error} onRetry={() => query.refetch()} />
      ) : query.data && !query.data.success ? (
        <EmptyState
          icon={Briefcase}
          title="No promoted model yet"
          description={query.data.error || "Train and promote a model first — then the portfolio agent can score your tickers."}
          actionLabel="Open Training"
          onAction={() => navigate("/training")}
        />
      ) : (
        <PortfolioSummary data={query.data} />
      )}
    </div>
  );
}
