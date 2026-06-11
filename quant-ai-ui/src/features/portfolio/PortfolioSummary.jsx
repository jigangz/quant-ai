import { Link } from "react-router-dom";
import { TrendingUp, TrendingDown, Minus, ArrowRight } from "lucide-react";
import { cn } from "@/lib/utils";

const SIGNAL_STYLES = {
  bullish: { icon: TrendingUp, chip: "bg-up/10 text-up" },
  bearish: { icon: TrendingDown, chip: "bg-down/10 text-down" },
  neutral: { icon: Minus, chip: "bg-surface-muted text-muted" },
};

function SignalChip({ signal }) {
  const s = SIGNAL_STYLES[signal] ?? SIGNAL_STYLES.neutral;
  const Icon = s.icon;
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium", s.chip)}>
      <Icon className="h-3 w-3" />
      {signal}
    </span>
  );
}

function DistributionBar({ bullish, neutral, bearish }) {
  const total = bullish + neutral + bearish || 1;
  const pct = (n) => `${(n / total) * 100}%`;
  return (
    <div>
      <div
        className="flex h-3 w-full overflow-hidden rounded-full bg-surface-muted"
        role="img"
        aria-label={`${bullish} bullish, ${neutral} neutral, ${bearish} bearish`}
      >
        <div className="bg-up" style={{ width: pct(bullish) }} />
        <div className="bg-surface-border" style={{ width: pct(neutral) }} />
        <div className="bg-down" style={{ width: pct(bearish) }} />
      </div>
      <div className="mt-2 flex gap-4 text-xs text-muted">
        <span><span className="font-semibold text-up">{bullish}</span> bullish</span>
        <span><span className="font-semibold text-foreground">{neutral}</span> neutral</span>
        <span><span className="font-semibold text-down">{bearish}</span> bearish</span>
      </div>
    </div>
  );
}

/**
 * Presentational view over a successful /agents/summary response.
 * Page owns loading / error / empty states.
 */
export default function PortfolioSummary({ data }) {
  if (!data?.success) return null;
  const analyses = data.analyses ?? [];
  const bullish = data.bullish_count ?? 0;
  const bearish = data.bearish_count ?? 0;
  const neutral = Math.max(analyses.length - bullish - bearish, 0);

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-surface-border bg-surface-card p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-medium text-muted">Overall signal</h2>
          <SignalChip signal={data.overall_signal === "mixed" ? "neutral" : data.overall_signal} />
        </div>
        <DistributionBar bullish={bullish} neutral={neutral} bearish={bearish} />
        {data.summary && <p className="text-sm text-foreground">{data.summary}</p>}
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {analyses.map((a) => (
          <div
            key={a.ticker}
            className="rounded-lg border border-surface-border bg-surface-card p-4 flex flex-col gap-2"
          >
            <div className="flex items-center justify-between">
              <span className="font-semibold text-foreground">{a.ticker}</span>
              <SignalChip signal={a.signal} />
            </div>
            <div className="text-xs text-muted">
              P(up) <span className="font-medium text-foreground tabular-nums">{(a.probability * 100).toFixed(0)}%</span>
              {a.top_driver && <> · driver: {a.top_driver}</>}
            </div>
            <Link
              to={`/dashboard?ticker=${a.ticker}`}
              className="mt-auto inline-flex items-center gap-1 text-xs text-accent hover:underline"
            >
              Analyze <ArrowRight className="h-3 w-3" />
            </Link>
          </div>
        ))}
      </div>
    </div>
  );
}
