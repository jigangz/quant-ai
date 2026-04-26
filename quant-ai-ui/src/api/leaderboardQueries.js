import { useQuery, useMutation } from "@tanstack/react-query";
import * as api from "./client";

export function useLeaderboard(labelType, opts = {}) {
  return useQuery({
    queryKey: ["leaderboard", labelType],
    queryFn: () => api.getModels({ label_type: labelType, status: "active" }),
    enabled: !!labelType,
    staleTime: 60_000,
    ...opts,
  });
}

export function useModelAccuracy(modelId, opts = {}) {
  return useQuery({
    queryKey: ["model-accuracy", modelId],
    queryFn: () => api.getModelAccuracy(modelId, { window_days: 30 }),
    enabled: !!modelId,
    staleTime: 30_000,
    retry: (failureCount, error) => {
      if (String(error.message).includes("API error 404")) return false;
      return failureCount < 2;
    },
    ...opts,
  });
}

export function useAblationRun() {
  return useMutation({
    mutationFn: (payload) => api.postAblationRun(payload),
  });
}
