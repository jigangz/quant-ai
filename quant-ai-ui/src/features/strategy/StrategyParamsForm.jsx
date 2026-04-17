import { useEffect, useState } from "react";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";
import { Label } from "../../components/ui/label";
import ErrorState from "../../components/ErrorState";
import { useStrategy, useStrategyBacktest } from "../../api/queries";

export default function StrategyParamsForm({ name, onResult }) {
  const { data: strategy, isLoading } = useStrategy(name);
  const [params, setParams] = useState({});
  const [ticker, setTicker] = useState("AAPL");
  const backtest = useStrategyBacktest();

  useEffect(() => {
    if (strategy?.default_params) setParams(strategy.default_params);
    else if (strategy?.parameters) {
      const defaults = {};
      Object.entries(strategy.parameters).forEach(([k, v]) => {
        defaults[k] = v.default ?? v;
      });
      setParams(defaults);
    }
  }, [strategy]);

  const runBacktest = async () => {
    const payload = { ticker, parameters: params };
    const result = await backtest.mutateAsync({ name, payload });
    onResult?.(result);
  };

  if (isLoading) return <div className="text-sm text-muted">Loading parameters...</div>;
  if (!strategy) return null;

  const paramEntries = Object.entries(strategy.parameters || {});

  return (
    <div className="space-y-4">
      <div>
        <Label htmlFor="ticker">Ticker</Label>
        <Input id="ticker" value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} />
      </div>
      {paramEntries.map(([key, def]) => (
        <div key={key}>
          <Label htmlFor={key}>{key}</Label>
          <Input
            id={key}
            type="number"
            step="any"
            value={params[key] ?? ""}
            onChange={(e) => setParams({ ...params, [key]: parseFloat(e.target.value) })}
          />
          {def.description && <p className="text-xs text-muted mt-1">{def.description}</p>}
        </div>
      ))}
      <Button onClick={runBacktest} disabled={backtest.isPending}>
        {backtest.isPending ? "Running..." : "Run Backtest"}
      </Button>
      {backtest.error && <ErrorState error={backtest.error} />}
    </div>
  );
}
