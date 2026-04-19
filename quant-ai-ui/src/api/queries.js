import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import * as api from "./client";

const SCREENER_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "WMT"];

// ===== Market =====
// Backend /data/market returns a flat array of OHLCV rows (newest first).
// Normalize to { rows: [...] } ascending by date so Lightweight Charts and
// Screener table can read it uniformly.
const normalizeMarket = (raw) => {
  if (!raw) return { rows: [] };
  if (Array.isArray(raw)) {
    const rows = [...raw].sort((a, b) => String(a.date).localeCompare(String(b.date)));
    return { rows };
  }
  if (Array.isArray(raw.rows)) {
    const rows = [...raw.rows].sort((a, b) => String(a.date).localeCompare(String(b.date)));
    return { ...raw, rows };
  }
  return { rows: [] };
};

export const useMarket = (ticker, opts = {}) =>
  useQuery({
    queryKey: ["market", ticker],
    queryFn: () => api.getMarket(ticker),
    select: normalizeMarket,
    enabled: !!ticker,
    staleTime: 30_000,
    ...opts,
  });

export const useScreenerTickers = () =>
  useQuery({
    queryKey: ["screener", SCREENER_TICKERS],
    queryFn: async () => {
      // lookback=5 → only need last 2 close prices to compute change; 5 gives safety buffer
      const results = await Promise.all(
        SCREENER_TICKERS.map((t) => api.getMarket(t, 5).catch(() => null))
      );
      return results
        .map((r, idx) => ({ ticker: SCREENER_TICKERS[idx], data: normalizeMarket(r) }))
        .filter((x) => x.data.rows.length > 0);
    },
    staleTime: 60_000,
  });

// ===== Prediction =====
export const usePredict = () =>
  useMutation({
    mutationFn: (payload) => api.predict(payload),
  });

// ===== Explain =====
export const useExplain = (ticker) =>
  useQuery({
    queryKey: ["explain", ticker],
    queryFn: () => api.explain(ticker),
    enabled: !!ticker,
  });

export const useSimilarCases = (query) =>
  useQuery({
    queryKey: ["search", query],
    queryFn: () => api.search(query),
    enabled: !!query,
  });

// ===== Training =====
export const useTrain = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.train,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["runs"] });
    },
  });
};

export const useRuns = (limit = 20) =>
  useQuery({
    queryKey: ["runs", limit],
    queryFn: () => api.listRuns(limit),
    refetchInterval: 10_000,
  });

export const useRunStatus = (runId) =>
  useQuery({
    queryKey: ["runs", runId],
    queryFn: () => api.getRunStatus(runId),
    enabled: !!runId,
    refetchInterval: (q) => {
      const status = q.state.data?.status;
      if (status === "success" || status === "failed") return false;
      return 2000;
    },
  });

// ===== Models =====
export const useModels = () =>
  useQuery({
    queryKey: ["models"],
    queryFn: () => api.listModels(),
  });

export const useModelTypes = () =>
  useQuery({
    queryKey: ["model-types"],
    queryFn: api.listModelTypes,
    staleTime: 5 * 60_000,
  });

export const usePromoteModel = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.promoteModel,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["models"] });
      qc.invalidateQueries({ queryKey: ["promoted-model"] });
    },
  });
};

export const useDemoteModel = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.demoteModel,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["models"] });
      qc.invalidateQueries({ queryKey: ["promoted-model"] });
    },
  });
};

export const usePromotedModel = () =>
  useQuery({
    queryKey: ["promoted-model"],
    queryFn: api.getPromotedModel,
  });

// ===== Features =====
export const useFeatureGroups = () =>
  useQuery({
    queryKey: ["feature-groups"],
    queryFn: api.listFeatureGroups,
    staleTime: 5 * 60_000,
  });

// ===== Strategies =====
export const useStrategies = () =>
  useQuery({
    queryKey: ["strategies"],
    queryFn: api.listStrategies,
    staleTime: 5 * 60_000,
  });

export const useStrategy = (name) =>
  useQuery({
    queryKey: ["strategy", name],
    queryFn: () => api.getStrategy(name),
    enabled: !!name,
  });

export const useGenerateSignals = () =>
  useMutation({ mutationFn: ({ name, payload }) => api.generateSignals(name, payload) });

export const useStrategyBacktest = () =>
  useMutation({ mutationFn: ({ name, payload }) => api.runStrategyBacktest(name, payload) });

// ===== Trading =====
export const usePortfolio = () =>
  useQuery({
    queryKey: ["portfolio"],
    queryFn: api.getPortfolio,
    refetchInterval: 5000,
  });

export const useOrders = (status = "all") =>
  useQuery({
    queryKey: ["orders", status],
    queryFn: () => api.listOrders(status),
    refetchInterval: 5000,
  });

export const useTrades = (limit = 20) =>
  useQuery({
    queryKey: ["trades", limit],
    queryFn: () => api.getTrades(limit),
    refetchInterval: 10_000,
  });

export const usePlaceOrder = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.placeOrder,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["portfolio"] });
      qc.invalidateQueries({ queryKey: ["orders"] });
      qc.invalidateQueries({ queryKey: ["trades"] });
    },
  });
};

export const useCancelOrder = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.cancelOrder,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["orders"] });
    },
  });
};

export const useResetPortfolio = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.resetPortfolio,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["portfolio"] });
      qc.invalidateQueries({ queryKey: ["orders"] });
      qc.invalidateQueries({ queryKey: ["trades"] });
    },
  });
};

// ===== Optimization =====
export const useOptimizeModel = () => useMutation({ mutationFn: api.optimizeModel });
export const useOptimizeStrategy = () => useMutation({ mutationFn: api.optimizeStrategy });
