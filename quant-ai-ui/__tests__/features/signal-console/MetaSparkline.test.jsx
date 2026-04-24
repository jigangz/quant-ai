import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { makeQueryWrapper } from "../../_helpers/queryWrapper";
import MetaSparkline from "@/features/signal-console/MetaSparkline";

vi.mock("@/api/client", () => ({
  getMetaLabelModels: vi.fn(),
  postSignalScore: vi.fn(),
  getMetaCoverage: vi.fn(),
  postMetaLabelTrain: vi.fn(),
}));
import * as client from "@/api/client";

beforeEach(() => vi.clearAllMocks());

describe("MetaSparkline", () => {
  it("renders sparkline when meta-model exists for the ticker", async () => {
    client.getMetaLabelModels.mockResolvedValue([
      {
        model_id: "meta_m",
        extras: {
          meta_label: {
            primary: { strategy_name: "rsi_strategy" },
            cv: { metrics: { auc_mean: 0.62 } },
            barrier: {},
          },
        },
      },
    ]);
    client.postSignalScore.mockResolvedValue({
      triggered: true,
      reliability_score: 0.6,
      signal: 1,
      expected_R: 0.4,
      recommended_action: "trade",
      sizing_hint: {},
      meta_model: {},
    });
    const { container } = render(<MetaSparkline ticker="MSFT" />, {
      wrapper: makeQueryWrapper(),
    });
    await waitFor(() => expect(container.querySelector("svg,canvas")).toBeTruthy());
  });

  it("renders nothing when no meta-model for ticker", async () => {
    client.getMetaLabelModels.mockResolvedValue([]);
    const { container } = render(<MetaSparkline ticker="NVDA" />, {
      wrapper: makeQueryWrapper(),
    });
    await waitFor(() => {
      // Resolved query with empty data → component returns null
      expect(container.textContent.trim()).toBe("");
    });
  });
});
