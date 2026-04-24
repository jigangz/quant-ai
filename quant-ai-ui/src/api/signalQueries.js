import { useQuery, useMutation } from "@tanstack/react-query";
import * as api from "./client";

/** Meta-label models for a specific ticker. */
export function useMetaLabelModels(ticker, opts = {}) {
  return useQuery({
    queryKey: ["meta-label-models", ticker],
    queryFn: () => api.getMetaLabelModels(ticker),
    enabled: !!ticker,
    staleTime: 30_000,
    ...opts,
  });
}

/** Coverage for a single strategy (Strategy card badge + Signal Console). */
export function useMetaCoverage(strategyName, opts = {}) {
  return useQuery({
    queryKey: ["meta-coverage", strategyName],
    queryFn: () => api.getMetaCoverage(strategyName),
    enabled: !!strategyName,
    staleTime: 60_000,
    retry: (failureCount, error) => {
      if (String(error.message).includes("API error 404")) return false;
      return failureCount < 2;
    },
    ...opts,
  });
}

/** Mutation — manual score preview. */
export function useSignalScorePreview() {
  return useMutation({
    mutationFn: (payload) => api.postSignalScore(payload),
  });
}

/** Mutation — train a new meta-label model from Signal Console CTA. */
export function useMetaLabelTrain() {
  return useMutation({
    mutationFn: (payload) => api.postMetaLabelTrain(payload),
  });
}
