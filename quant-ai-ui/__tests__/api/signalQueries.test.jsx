import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { makeQueryWrapper } from "../_helpers/queryWrapper";

vi.mock("@/api/client", () => ({
  getMetaLabelModels: vi.fn(),
  getMetaCoverage: vi.fn(),
  postSignalScore: vi.fn(),
  postMetaLabelTrain: vi.fn(),
}));

import * as client from "@/api/client";
import { useMetaLabelModels, useMetaCoverage, useSignalScorePreview } from "@/api/signalQueries";

beforeEach(() => { vi.clearAllMocks(); });

describe("useMetaLabelModels", () => {
  it("calls getMetaLabelModels with ticker and returns data", async () => {
    client.getMetaLabelModels.mockResolvedValue([
      { model_id: "meta_a", extras: { meta_label: { primary: { strategy_name: "rsi_strategy" } } } },
    ]);
    const { result } = renderHook(() => useMetaLabelModels("AAPL"), { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(result.current.data).toBeDefined());
    expect(client.getMetaLabelModels).toHaveBeenCalledWith("AAPL");
    expect(result.current.data).toHaveLength(1);
  });
});

describe("useMetaCoverage", () => {
  it("calls getMetaCoverage with strategy name and returns data", async () => {
    client.getMetaCoverage.mockResolvedValue({ count: 3, max_auc: 0.62, avg_auc: 0.55, tickers: ["MSFT"] });
    const { result } = renderHook(() => useMetaCoverage("rsi_strategy"), { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(result.current.data).toBeDefined());
    expect(result.current.data.count).toBe(3);
  });
});

describe("useSignalScorePreview", () => {
  it("exposes a mutate function that calls postSignalScore", async () => {
    client.postSignalScore.mockResolvedValue({ triggered: true, reliability_score: 0.71, signal: 1 });
    const { result } = renderHook(() => useSignalScorePreview(), { wrapper: makeQueryWrapper() });
    await result.current.mutateAsync({ ticker: "AAPL", meta_model_id: "meta_a", signal: 1 });
    expect(client.postSignalScore).toHaveBeenCalledWith({ ticker: "AAPL", meta_model_id: "meta_a", signal: 1 });
  });
});
