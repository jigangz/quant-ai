import { Card, CardContent } from "../../components/ui/card";
import { LoadingOverlay } from "../../components/LoadingSpinner";
import { useRankingBacktest } from "../../api/queries";
import { Info } from "lucide-react";

/** Two-line equity curve (strategy net vs benchmark), normalized to start = 1. */
function EquityCurve({ strategy = [], benchmark = [] }) {
  const all = [...strategy, ...benchmark];
  if (all.length < 2) return null;
  const W = 560;
  const H = 150;
  const pad = 8;
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const span = hi - lo || 1;
  const n = Math.max(strategy.length, benchmark.length);
  const x = (i) => pad + (i / (n - 1)) * (W - 2 * pad);
  const y = (v) => H - pad - ((v - lo) / span) * (H - 2 * pad);
  const path = (arr) =>
    arr.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" preserveAspectRatio="none">
      <line
        x1={pad} y1={y(1)} x2={W - pad} y2={y(1)}
        stroke="currentColor" className="text-surface-border" strokeDasharray="3 3"
      />
      <path d={path(benchmark)} fill="none" stroke="currentColor" className="text-muted" strokeWidth="1.5" />
      <path d={path(strategy)} fill="none" stroke="currentColor" className="text-accent" strokeWidth="2.25" />
    </svg>
  );
}

const ROWS = [
  ["Sharpe", "sharpe", ""],
  ["CAGR", "cagr", "%"],
  ["Max DD", "max_drawdown", "%"],
];

export default function BacktestCard() {
  const { data, isLoading, error } = useRankingBacktest({ topPct: 0.1 });
  const ok = data?.success;
  const s = data?.strategy;
  const b = data?.benchmark;

  return (
    <Card className="relative">
      {isLoading && <LoadingOverlay label="Running out-of-sample backtest…" />}
      <CardContent className="p-4 space-y-3">
        <div className="flex items-baseline justify-between gap-3">
          <h3 className="font-semibold text-foreground">Out-of-sample backtest</h3>
          {ok && (
            <span className="text-xs text-muted text-right">
              Top-decile · net of {data.cost_bps}bps · {data.oos_start} → {data.oos_end} ·{" "}
              {data.n_rebalances} rebalances
            </span>
          )}
        </div>

        {error || (data && !ok) ? (
          <p className="text-sm text-muted">
            {(data && data.error) || "Backtest unavailable — publish an xs_strong model first."}
          </p>
        ) : ok ? (
          <>
            <EquityCurve strategy={s.equity} benchmark={b.equity} />

            <div className="flex items-center gap-4 text-xs text-muted">
              <span className="flex items-center gap-1.5">
                <span className="inline-block w-3.5 h-[3px] rounded bg-accent" /> Strategy (net)
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block w-3.5 h-[2px] rounded bg-muted" /> Equal-weight universe
              </span>
            </div>

            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-muted">
                  <th className="text-left font-normal py-1" />
                  <th className="text-right font-normal">Strategy (net)</th>
                  <th className="text-right font-normal">EW universe</th>
                </tr>
              </thead>
              <tbody>
                {ROWS.map(([label, key, suf]) => (
                  <tr key={key} className="border-t border-surface-border/40">
                    <td className="py-1 text-muted">{label}</td>
                    <td className="py-1 text-right tabular-nums font-medium text-foreground">
                      {s[key]}{suf}
                    </td>
                    <td className="py-1 text-right tabular-nums text-muted">{b[key]}{suf}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <p className="flex items-start gap-1.5 text-xs text-muted pt-1">
              <Info className="h-3.5 w-3.5 mt-0.5 shrink-0 text-accent" />
              {data.caveat}
            </p>
          </>
        ) : null}
      </CardContent>
    </Card>
  );
}
