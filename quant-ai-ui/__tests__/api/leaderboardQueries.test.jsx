import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { makeQueryWrapper } from "../_helpers/queryWrapper";

vi.mock("@/api/client", () => ({
  getModels: vi.fn(),
  getModelAccuracy: vi.fn(),
  postAblationRun: vi.fn(),
}));

import * as client from "@/api/client";
import {
  useLeaderboard,
  useModelAccuracy,
  useAblationRun,
} from "@/api/leaderboardQueries";

beforeEach(() => vi.clearAllMocks());

describe("useLeaderboard", () => {
  it("calls getModels with label_type filter", async () => {
    client.getModels.mockResolvedValue([
      { id: "m1", label_type: "direction", metrics: { test_auc: 0.6 } },
    ]);
    const { result } = renderHook(() => useLeaderboard("direction"), { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(result.current.data).toBeDefined());
    expect(client.getModels).toHaveBeenCalledWith({ label_type: "direction", status: "active" });
  });
});

describe("useModelAccuracy", () => {
  it("fetches accuracy when modelId given", async () => {
    client.getModelAccuracy.mockResolvedValue({
      model_id: "m1", stats: { hit_rate: 0.6 }, by_ticker: [],
    });
    const { result } = renderHook(() => useModelAccuracy("m1"), { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(result.current.data).toBeDefined());
    expect(client.getModelAccuracy).toHaveBeenCalledWith("m1", { window_days: 30 });
  });
});

describe("useAblationRun", () => {
  it("returns mutation that posts ablation run", async () => {
    client.postAblationRun.mockResolvedValue({ ticker: "MSFT", matrix: {} });
    const { result } = renderHook(() => useAblationRun(), { wrapper: makeQueryWrapper() });
    await result.current.mutateAsync({ ticker: "MSFT", targets: ["direction"], feature_sets: [] });
    expect(client.postAblationRun).toHaveBeenCalled();
  });
});
