import { useNavigate } from "react-router-dom";
import { cn } from "../../lib/utils";
import { fmtPrice, fmtPct, fmtVolume, classForDelta } from "../../lib/formatters";
import Sparkline from "./Sparkline";

export default function ScreenerTable({ rows = [], sortBy }) {
  const navigate = useNavigate();

  const sorted = [...rows]
    .filter(Boolean)
    .sort((a, b) =>
      sortBy === "volume"
        ? (b.volume ?? 0) - (a.volume ?? 0)
        : (b.change_pct ?? 0) - (a.change_pct ?? 0)
    );

  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b border-surface-border text-muted">
          <th className="text-left px-4 py-3 font-medium">Ticker</th>
          <th className="text-right px-4 py-3 font-medium">Last</th>
          <th className="text-right px-4 py-3 font-medium">Change</th>
          <th className="text-right px-4 py-3 font-medium">Change %</th>
          <th className="text-center px-4 py-3 font-medium">30D</th>
          <th className="text-right px-4 py-3 font-medium">Volume</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((row) => (
          <tr
            key={row.ticker}
            onClick={() => navigate(`/dashboard?ticker=${row.ticker}`)}
            className="border-b border-surface-border hover:bg-surface-hover cursor-pointer transition-colors"
          >
            <td className="px-4 py-3 font-medium text-foreground">{row.ticker}</td>
            <td className="px-4 py-3 text-right">${fmtPrice(row.close)}</td>
            <td className={cn("px-4 py-3 text-right font-medium", classForDelta(row.change ?? 0))}>
              {(row.change ?? 0) >= 0 ? "+" : ""}{fmtPrice(row.change ?? 0)}
            </td>
            <td className={cn("px-4 py-3 text-right font-medium", classForDelta(row.change_pct ?? 0))}>
              {fmtPct(row.change_pct ?? 0)}
            </td>
            <td className="px-4 py-3">
              <div className="flex justify-center">
                <Sparkline values={row.series ?? []} />
              </div>
            </td>
            <td className="px-4 py-3 text-right text-muted">
              {fmtVolume(row.volume)}
            </td>
          </tr>
        ))}
        {sorted.length === 0 && (
          <tr>
            <td colSpan={6} className="px-4 py-8 text-center text-muted">
              No data available
            </td>
          </tr>
        )}
      </tbody>
    </table>
  );
}
