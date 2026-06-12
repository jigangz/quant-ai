import { useState } from "react";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";
import { Label } from "../../components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../components/ui/select";
import ErrorState from "../../components/ErrorState";
import { usePlaceOrder } from "../../api/queries";
import { useMetaLabelModels, useSignalScorePreview } from "@/api/signalQueries";

export default function OrderForm() {
  const [ticker, setTicker] = useState("AAPL");
  const [side, setSide] = useState("buy");
  const [type, setType] = useState("market");
  const [qty, setQty] = useState(10);
  const [price, setPrice] = useState("");
  const place = usePlaceOrder();

  // Meta-label filter state
  const [metaEnabled, setMetaEnabled] = useState(false);
  const [metaModelId, setMetaModelId] = useState("");
  const [metaThreshold, setMetaThreshold] = useState(0.55);
  const [metaScore, setMetaScore] = useState(null);
  const metaModels = useMetaLabelModels(ticker, { enabled: metaEnabled && !!ticker });
  const preview = useSignalScorePreview();

  const onPreview = () => {
    if (!metaModelId || !ticker) return;
    preview.mutate(
      { ticker, meta_model_id: metaModelId, signal: side === "buy" ? 1 : -1 },
      { onSuccess: setMetaScore },
    );
  };

  const submit = async (e) => {
    e.preventDefault();
    const payload = { ticker, side, order_type: type, quantity: Number(qty) };
    if (type === "limit") payload.limit_price = Number(price);
    if (metaEnabled && metaModelId) {
      payload.meta_model_id = metaModelId;
      payload.score_threshold = metaThreshold;
    }
    await place.mutateAsync(payload);
  };

  return (
    <form onSubmit={submit} className="space-y-3">
      <div>
        <Label htmlFor="ticker">Ticker</Label>
        <Input id="ticker" value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label>Side</Label>
          <Select value={side} onValueChange={setSide}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="buy">Buy</SelectItem>
              <SelectItem value="sell">Sell</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label>Type</Label>
          <Select value={type} onValueChange={setType}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="market">Market</SelectItem>
              <SelectItem value="limit">Limit</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label htmlFor="qty">Quantity</Label>
          <Input id="qty" type="number" min="1" value={qty} onChange={(e) => setQty(e.target.value)} />
        </div>
        {type === "limit" && (
          <div>
            <Label htmlFor="price">Limit Price</Label>
            <Input id="price" type="number" step="0.01" value={price} onChange={(e) => setPrice(e.target.value)} />
          </div>
        )}
      </div>
      <div className="border-t border-slate-800 pt-4">
        <label className="flex items-center gap-2 text-sm cursor-pointer">
          <input
            type="checkbox"
            checked={metaEnabled}
            onChange={(e) => setMetaEnabled(e.target.checked)}
            aria-label="Use meta-label filter"
          />
          <span>Use meta-label filter</span>
        </label>
        {metaEnabled && (
          <div className="mt-3 space-y-3 pl-6">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Meta model</label>
              <select
                data-testid="meta-model-select"
                value={metaModelId}
                onChange={(e) => setMetaModelId(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm"
              >
                <option value="">— select a model —</option>
                {(metaModels.data || []).map((m) => {
                  const prim = m.extras?.meta_label?.primary?.strategy_name || "—";
                  const auc = m.extras?.meta_label?.cv?.metrics?.auc_mean ?? 0;
                  return (
                    <option key={m.model_id} value={m.model_id}>
                      {prim} · {m.model_id.slice(0, 12)}... · AUC {auc.toFixed(2)}
                      {auc < 0.5 ? " ⚠" : ""}
                    </option>
                  );
                })}
              </select>
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1" htmlFor="meta-threshold">
                Threshold: {metaThreshold.toFixed(2)}
              </label>
              <input
                id="meta-threshold"
                type="range"
                min="0.45"
                max="0.85"
                step="0.01"
                value={metaThreshold}
                onChange={(e) => setMetaThreshold(parseFloat(e.target.value))}
                aria-label="Threshold"
                className="w-full"
              />
            </div>
            <button
              type="button"
              onClick={onPreview}
              disabled={!metaModelId}
              className="px-3 py-1 text-sm bg-emerald-600/20 border border-emerald-600/40 rounded hover:bg-emerald-600/30 disabled:opacity-50"
            >
              Preview score
            </button>
            {metaScore?.triggered && (
              <div className="p-3 bg-slate-800/50 rounded text-xs space-y-1">
                <div>
                  Score: <span className="font-semibold">{metaScore.reliability_score.toFixed(2)}</span>
                  {" · "}E[R]: {(metaScore.expected_R ?? 0).toFixed(2)}
                </div>
                <div className="uppercase font-bold text-sm">
                  Action: <span className={
                    metaScore.recommended_action === "trade" ? "text-emerald-400"
                      : metaScore.recommended_action === "skip" ? "text-rose-400" : "text-amber-400"
                  }>{metaScore.recommended_action}</span>
                </div>
                {metaScore.sizing_hint && (
                  <div className="text-slate-400">
                    Sizing hint: {(metaScore.sizing_hint.half_kelly_fraction * 100).toFixed(1)}% of capital
                  </div>
                )}
              </div>
            )}
            {metaScore && !metaScore.triggered && (
              <div className="text-xs text-amber-400">
                Strategy silent at latest close — {metaScore.reason || "did not trigger"}
              </div>
            )}
          </div>
        )}
      </div>
      <Button type="submit" disabled={place.isPending} className="w-full">
        {place.isPending ? "Placing..." : `Place ${side.toUpperCase()}`}
      </Button>
      {place.error && <ErrorState error={place.error} />}
      {place.data && <div className="text-sm text-up">Order placed · id: {place.data.order_id || place.data.id}</div>}
    </form>
  );
}
