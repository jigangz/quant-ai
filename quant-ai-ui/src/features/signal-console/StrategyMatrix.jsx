import { useMetaLabelModels } from "@/api/signalQueries";

const STRATEGIES = ["ma_cross", "rsi_strategy", "bollinger_breakout", "sentiment_driven"];

function Cell({ ticker, strategy, model, onSelect, onTrain }) {
  if (!model) {
    return (
      <td className="px-2 py-2">
        <button
          type="button"
          onClick={() => onTrain?.({ ticker, strategy })}
          className="w-full text-[10px] text-slate-500 hover:text-emerald-300 border border-dashed border-slate-700 rounded px-1 py-2"
        >
          — Train meta
        </button>
      </td>
    );
  }
  const auc = model.extras.meta_label.cv.metrics.auc_mean;
  const er = model.extras.meta_label.cv.metrics.expected_R_when_trade;
  const variant = auc < 0.5 ? "warn" : auc >= 0.60 ? "good" : "neutral";
  const bg = variant === "warn" ? "bg-amber-500/10" : variant === "good" ? "bg-emerald-500/10" : "bg-slate-700/30";
  return (
    <td className="px-2 py-2">
      <button
        type="button"
        data-cell
        data-variant={variant}
        onClick={() => onSelect({ ticker, strategy, model_id: model.model_id })}
        className={`w-full text-xs rounded px-2 py-2 text-left ${bg} hover:ring-1 hover:ring-emerald-500/50`}
      >
        <div className="font-medium">AUC {auc.toFixed(2)}{variant === "warn" ? " ⚠" : ""}</div>
        <div className="text-[10px] text-slate-400">E[R] {er >= 0 ? "+" : ""}{er.toFixed(2)}</div>
      </button>
    </td>
  );
}

function TickerRow({ ticker, onSelect, onTrain }) {
  const { data = [] } = useMetaLabelModels(ticker);
  const byStrategy = {};
  for (const m of data) {
    const s = m?.extras?.meta_label?.primary?.strategy_name;
    if (s) byStrategy[s] = m;
  }
  return (
    <tr>
      <td className="px-3 py-2 text-sm font-medium text-slate-200 border-r border-slate-800">{ticker}</td>
      {STRATEGIES.map((s) => (
        <Cell
          key={s}
          ticker={ticker}
          strategy={s}
          model={byStrategy[s]}
          onSelect={onSelect}
          onTrain={onTrain}
        />
      ))}
    </tr>
  );
}

export default function StrategyMatrix({ tickers, onSelect, onTrain }) {
  if (!tickers || tickers.length === 0) {
    return (
      <div className="p-6 text-sm text-slate-500 text-center bg-slate-900/40 rounded-lg">
        Select one or more tickers from the watchlist above.
      </div>
    );
  }
  return (
    <div className="overflow-x-auto bg-slate-900/40 rounded-lg">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-[10px] uppercase tracking-wide text-slate-400 border-b border-slate-800">
            <th className="px-3 py-2 text-left">Ticker</th>
            {STRATEGIES.map((s) => <th key={s} className="px-2 py-2 text-left">{s}</th>)}
          </tr>
        </thead>
        <tbody>
          {tickers.map((t) => (
            <TickerRow key={t} ticker={t} onSelect={onSelect} onTrain={onTrain} />
          ))}
        </tbody>
      </table>
    </div>
  );
}
