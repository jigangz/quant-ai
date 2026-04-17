import { useSearchParams } from "react-router-dom";
import { useState, useMemo } from "react";
import PageHeader from "../components/PageHeader";
import TickerSearch from "../components/TickerSearch";
import { LoadingOverlay } from "../components/LoadingSpinner";
import ErrorState from "../components/ErrorState";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import CandlestickChart from "../features/charts/CandlestickChart";
import { useMarket, usePredict, useExplain } from "../api/queries";
import { fmtPrice, fmtPct, classForDelta } from "../lib/formatters";
import { TrendingUp, TrendingDown, Sparkles } from "lucide-react";

export default function DashboardPage() {
  const [searchParams] = useSearchParams();
  const ticker = (searchParams.get("ticker") || "AAPL").toUpperCase();
  const [prediction, setPrediction] = useState(null);

  const { data: marketData, isLoading: marketLoading, error: marketError } = useMarket(ticker);
  const { data: explainData } = useExplain(ticker);
  const predictMutation = usePredict();

  const candles = useMemo(() => {
    if (!marketData?.rows) return [];
    return marketData.rows.map((r) => ({
      time: r.date,
      open: r.open,
      high: r.high,
      low: r.low,
      close: r.close,
    }));
  }, [marketData]);

  const lastRow = marketData?.rows?.[marketData.rows.length - 1];
  const prevRow = marketData?.rows?.[marketData.rows.length - 2];
  const change = lastRow && prevRow ? lastRow.close - prevRow.close : 0;
  const changePct = prevRow ? change / prevRow.close : 0;

  const handlePredict = async () => {
    try {
      const result = await predictMutation.mutateAsync({ ticker, horizons: [5] });
      setPrediction(result);
    } catch (e) {
      // error shown below via mutation.error
    }
  };

  return (
    <div>
      <PageHeader
        title={`${ticker}`}
        subtitle={lastRow ? `$${fmtPrice(lastRow.close)}` : "—"}
        actions={<TickerSearch defaultTicker={ticker} />}
      />

      {marketLoading && <LoadingOverlay label="Loading market data..." />}
      {marketError && <ErrorState error={marketError} />}

      {!marketLoading && marketData && (
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          <Card className="lg:col-span-3">
            <CardHeader className="flex-row items-center justify-between gap-4 pb-2">
              <CardTitle>Price</CardTitle>
              <div className={`flex items-center gap-1 text-sm font-mono ${classForDelta(change)}`}>
                {change >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                {fmtPrice(Math.abs(change))} ({fmtPct(changePct)})
              </div>
            </CardHeader>
            <CardContent className="p-2">
              <CandlestickChart data={candles} height={420} />
            </CardContent>
          </Card>

          <div className="lg:col-span-2 space-y-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle>Prediction</CardTitle>
              </CardHeader>
              <CardContent>
                {prediction ? (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Badge variant={prediction.prediction === 1 ? "success" : "destructive"}>
                        {prediction.prediction === 1 ? "BULLISH" : "BEARISH"}
                      </Badge>
                      <span className="text-sm text-muted">
                        prob_up: {fmtPct(prediction.probability_up ?? prediction.prob_up ?? 0, { dp: 1 })}
                      </span>
                    </div>
                    <div className="text-xs text-muted">
                      Horizon: {prediction.horizon || 5} days · Model: {prediction.model_id || "promoted"}
                    </div>
                  </div>
                ) : (
                  <Button onClick={handlePredict} disabled={predictMutation.isPending} className="w-full">
                    <Sparkles className="h-4 w-4" />
                    {predictMutation.isPending ? "Predicting..." : "Run prediction"}
                  </Button>
                )}
                {predictMutation.error && <ErrorState className="mt-3" error={predictMutation.error} />}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle>SHAP Top Features</CardTitle>
              </CardHeader>
              <CardContent>
                {explainData?.top_features?.length ? (
                  <ul className="space-y-2">
                    {explainData.top_features.slice(0, 6).map((f, i) => {
                      const max = explainData.top_features[0].mean_abs_shap;
                      const pct = (f.mean_abs_shap / max) * 100;
                      return (
                        <li key={i} className="text-sm">
                          <div className="flex justify-between mb-1">
                            <span className="font-mono text-foreground">{f.feature}</span>
                            <span className="text-muted">{f.mean_abs_shap.toFixed(4)}</span>
                          </div>
                          <div className="h-1.5 bg-surface-muted rounded-full overflow-hidden">
                            <div className="h-full bg-accent" style={{ width: `${pct}%` }} />
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                ) : (
                  <p className="text-sm text-muted">No SHAP data available.</p>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
