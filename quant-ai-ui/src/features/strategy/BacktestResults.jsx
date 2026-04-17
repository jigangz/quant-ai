import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";
import { classForDelta, fmtPct } from "../../lib/formatters";

export default function BacktestResults({ result }) {
  if (!result) return <div className="text-sm text-muted">Run a backtest to see results.</div>;

  const metrics = result.metrics || result;

  const kpi = [
    { label: "Sharpe Ratio", value: metrics.sharpe_ratio?.toFixed(2) ?? "—" },
    { label: "Total Return", value: fmtPct(metrics.total_return), color: classForDelta(metrics.total_return) },
    { label: "Max Drawdown", value: fmtPct(metrics.max_drawdown), color: classForDelta(metrics.max_drawdown) },
    { label: "Win Rate", value: fmtPct(metrics.win_rate, { dp: 1 }) },
  ];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {kpi.map((k) => (
          <Card key={k.label}>
            <CardContent className="p-4">
              <p className="text-xs text-muted uppercase tracking-wider">{k.label}</p>
              <p className={`text-2xl font-bold font-mono mt-1 ${k.color || "text-foreground"}`}>{k.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>
      {metrics.equity_curve && (
        <Card>
          <CardHeader>
            <CardTitle>Equity curve</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted">
              {metrics.equity_curve.length} data points · final equity{" "}
              <span className="font-mono text-foreground">
                {metrics.equity_curve[metrics.equity_curve.length - 1]?.toFixed(2)}
              </span>
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
