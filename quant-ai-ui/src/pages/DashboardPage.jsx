import { useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ThemeScope } from "@/theme/ThemeScope";
import { getMarket, getRelatedStocks, agentSummary } from "@/api/client";
import { useAgentTechnical, useModelMeta, useModelsForTicker, useSeasonalAccuracy, useSentiment } from "@/api/queries";
import { SymbolHeader } from "@/features/dashboard/SymbolHeader";
import { SymbolTabs } from "@/features/dashboard/SymbolTabs";
import { AIInsightBand } from "@/features/dashboard/AIInsightBand";
import { ChartSection } from "@/features/dashboard/ChartSection";
import { KeyDataGrid } from "@/features/dashboard/KeyDataGrid";
import { AboutBlock } from "@/features/dashboard/AboutBlock";
import { RelatedStocks } from "@/features/dashboard/RelatedStocks";
import { NewsGrid } from "@/features/dashboard/NewsGrid";
import { ModelComparison } from "@/features/dashboard/ModelComparison";
import { GaugesSection } from "@/features/dashboard/GaugesSection";
import { VolatilityCard } from "@/features/dashboard/VolatilityCard";
import { SeasonalityHeatmap } from "@/features/dashboard/SeasonalityHeatmap";
import { CTARow } from "@/features/dashboard/CTARow";
import { RightRailWatchlist } from "@/components/layout/RightRailWatchlist";

function useMarket(ticker) {
  return useQuery({
    queryKey: ["market", ticker, "6mo"],
    queryFn: () => getMarket(ticker),
    enabled: !!ticker,
    staleTime: 60 * 1000,
  });
}

function useRelatedWithSignals(ticker) {
  return useQuery({
    queryKey: ["related+signals", ticker],
    queryFn: async () => {
      const peers = await getRelatedStocks(ticker);
      if (!peers.length) return [];
      const [markets, summary] = await Promise.all([
        Promise.all(peers.map((t) => getMarket(t).catch(() => []))),
        agentSummary({ tickers: peers }).catch(() => ({ analyses: [] })),
      ]);
      return peers.map((t, i) => {
        const raw = markets[i];
        const candles = Array.isArray(raw) ? raw : (raw?.rows ?? []);
        const latest = candles[candles.length - 1];
        const analysis = (summary.analyses ?? []).find((a) => a.ticker === t);
        return {
          ticker: t,
          name: t,
          price: latest?.close,
          signal: analysis
            ? {
                direction: analysis.prediction === "up" || analysis.prediction === 1 ? "bullish" : "bearish",
                confidence: "medium",
              }
            : null,
        };
      });
    },
    enabled: !!ticker,
    staleTime: 5 * 60 * 1000,
  });
}

export default function DashboardPage() {
  const [params] = useSearchParams();
  const ticker = params.get("ticker") ?? "AAPL";
  const modelId = params.get("modelId");

  const marketQ = useMarket(ticker);
  const aiQ = useAgentTechnical(ticker, modelId);
  const newsQ = useSentiment(ticker, 30);
  const modelMetaQ = useModelMeta(aiQ.data?.model_id);
  const modelsForTickerQ = useModelsForTicker(ticker);
  const relatedQ = useRelatedWithSignals(ticker);
  const seasonalQ = useSeasonalAccuracy(ticker, aiQ.data?.model_id);

  const candles = marketQ.data ?? [];
  const sorted = [...candles].sort((a, b) => new Date(a.date) - new Date(b.date));
  const latest = sorted[sorted.length - 1];
  const prev = sorted[sorted.length - 2];
  const change = latest && prev ? latest.close - prev.close : 0;
  const changePct = latest && prev ? (change / prev.close) * 100 : 0;

  return (
    <ThemeScope value="light" className="min-h-full">
      <div className="grid grid-cols-[1fr_280px]">
        <div className="p-5 max-w-[1200px]">
          <nav className="text-[11px] text-muted mb-2">
            市场 / 美国 / 股票 / {ticker}
          </nav>
          <SymbolHeader
            ticker={ticker}
            name={ticker}
            exchange="NASDAQ"
            price={latest?.close}
            change={change}
            changePct={changePct}
            lastUpdate={latest?.date}
            modelSource={
              modelMetaQ.data
                ? { runId: modelMetaQ.data.training_run_id, gitSha: modelMetaQ.data.git_sha?.slice(0, 7) }
                : null
            }
          />
          <SymbolTabs active="概览" />
          <AIInsightBand data={aiQ.data} isLoading={aiQ.isLoading} error={aiQ.error} />
          <ChartSection candles={candles} isLoading={marketQ.isLoading} />
          <KeyDataGrid latestCandle={latest} prevClose={prev?.close} />
          <AboutBlock ticker={ticker} modelMeta={modelMetaQ.data} />
          <RelatedStocks peers={relatedQ.data ?? []} />
          <NewsGrid items={newsQ.data?.news ?? []} />
          <ModelComparison models={modelsForTickerQ.data ?? []} promotedId={aiQ.data?.model_id} />
          <GaugesSection
            prediction={aiQ.data?.prediction}
            probability={aiQ.data?.probability}
            confidence={aiQ.data?.confidence}
            signals={aiQ.data?.signals ?? []}
          />
          <VolatilityCard ticker={ticker} />
          <SeasonalityHeatmap monthly={seasonalQ.data?.monthly} />
          <CTARow ticker={ticker} prediction={aiQ.data?.prediction} />
        </div>
        <RightRailWatchlist currentTicker={ticker} />
      </div>
    </ThemeScope>
  );
}
