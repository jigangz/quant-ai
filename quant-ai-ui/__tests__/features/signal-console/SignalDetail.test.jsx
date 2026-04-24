import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { makeQueryWrapper } from "../../_helpers/queryWrapper";
import SignalDetail from "@/features/signal-console/SignalDetail";

vi.mock("@/api/client", () => ({
  postSignalScore: vi.fn(),
  getMetaLabelModels: vi.fn(), getMetaCoverage: vi.fn(), postMetaLabelTrain: vi.fn(),
}));
import * as client from "@/api/client";

beforeEach(() => vi.clearAllMocks());

describe("SignalDetail", () => {
  it("renders score, expected_R, recommended_action when triggered", async () => {
    client.postSignalScore.mockResolvedValue({
      triggered: true, signal: 1, reliability_score: 0.71, expected_R: 0.54,
      recommended_action: "trade",
      sizing_hint: { half_kelly_fraction: 0.18, raw_kelly: 0.36, cap: 0.25 },
      meta_model: { id: "meta_a", primary_source: "strategy:rsi_strategy", cv_auc: 0.62 },
      timestamp: "2026-04-24T20:00:00Z",
    });
    render(
      <SignalDetail selection={{ ticker: "AAPL", strategy: "rsi_strategy", model_id: "meta_a" }} />,
      { wrapper: makeQueryWrapper() },
    );
    await waitFor(() => expect(screen.getByText(/0\.71/)).toBeInTheDocument());
    expect(screen.getByText(/TRADE/i)).toBeInTheDocument();
    expect(screen.getByText(/\+0\.54/)).toBeInTheDocument();
  });

  it("shows 'silent' message when triggered=false", async () => {
    client.postSignalScore.mockResolvedValue({
      triggered: false, signal: 0, reason: "rsi_strategy did not trigger at latest close",
      timestamp: "2026-04-24T20:00:00Z",
    });
    render(
      <SignalDetail selection={{ ticker: "AAPL", strategy: "rsi_strategy", model_id: "meta_a" }} />,
      { wrapper: makeQueryWrapper() },
    );
    await waitFor(() => expect(screen.getByText(/silent|did not trigger/i)).toBeInTheDocument());
  });

  it("renders nothing when selection is null", () => {
    const { container } = render(<SignalDetail selection={null} />, { wrapper: makeQueryWrapper() });
    expect(container.textContent.toLowerCase()).toContain("select a cell");
  });
});
