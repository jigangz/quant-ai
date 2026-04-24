import { useEffect, useState } from "react";
import { useSignalScorePreview } from "@/api/signalQueries";

export default function SignalDetail({ selection }) {
  const preview = useSignalScorePreview();
  const [resp, setResp] = useState(null);

  useEffect(() => {
    setResp(null);
    if (!selection?.model_id) return;
    preview.mutate(
      { ticker: selection.ticker, meta_model_id: selection.model_id, strategy_name: selection.strategy },
      { onSuccess: setResp },
    );
  }, [selection?.model_id]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!selection) {
    return (
      <div className="p-6 text-sm text-slate-500 bg-slate-900/40 rounded-lg">
        Select a cell in the matrix to see signal detail.
      </div>
    );
  }

  if (preview.isPending || !resp) {
    return (
      <div className="p-6 text-sm text-slate-400 bg-slate-900/40 rounded-lg">
        Loading signal score for {selection.ticker} × {selection.strategy}...
      </div>
    );
  }

  if (!resp.triggered) {
    return (
      <div className="p-6 bg-slate-900/40 rounded-lg space-y-2">
        <div className="text-xs text-slate-400">{selection.ticker} × {selection.strategy}</div>
        <div className="text-sm text-amber-300">
          Strategy silent at latest close — {resp.reason || "did not trigger"}
        </div>
      </div>
    );
  }

  const score = resp.reliability_score ?? 0;
  const action = resp.recommended_action?.toUpperCase() ?? "—";
  const actionColor = action === "TRADE" ? "text-emerald-400" : action === "SKIP" ? "text-rose-400" : "text-amber-300";
  const sizing = resp.sizing_hint ?? {};

  return (
    <div className="p-6 bg-slate-900/40 rounded-lg space-y-4">
      <div>
        <div className="text-xs text-slate-400">
          {selection.ticker} × {selection.strategy}
        </div>
        <div className="text-sm text-slate-500">Model: {resp.meta_model?.id}</div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <Metric label="Reliability score" value={score.toFixed(2)} />
        <Metric
          label="Expected R"
          value={`${resp.expected_R >= 0 ? "+" : ""}${(resp.expected_R ?? 0).toFixed(2)}`}
        />
        <Metric label="Signal" value={resp.signal > 0 ? "Long (+1)" : "Short (-1)"} />
        <Metric label="CV AUC" value={(resp.meta_model?.cv_auc ?? 0).toFixed(2)} />
      </div>

      <div className={`text-lg font-bold ${actionColor}`}>Action: {action}</div>

      {sizing.half_kelly_fraction !== undefined && (
        <div className="text-xs text-slate-400 space-y-1">
          <div>Sizing hint (half-Kelly): {(sizing.half_kelly_fraction * 100).toFixed(1)}% of capital</div>
          <div>Raw Kelly: {(sizing.raw_kelly * 100).toFixed(1)}% · Cap: {(sizing.cap * 100).toFixed(0)}%</div>
        </div>
      )}

      <div className="text-[10px] text-slate-500 pt-2 border-t border-slate-800">
        Primary: {resp.meta_model?.primary_source} · Timestamp: {resp.timestamp}
      </div>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="bg-slate-800/50 rounded px-3 py-2">
      <div className="text-[10px] uppercase text-slate-500">{label}</div>
      <div className="text-lg font-semibold text-slate-100">{value}</div>
    </div>
  );
}
