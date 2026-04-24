import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { makeQueryWrapper } from "../../_helpers/queryWrapper";
import StrategyMatrix from "@/features/signal-console/StrategyMatrix";

vi.mock("@/api/client", () => ({
  getMetaLabelModels: vi.fn(),
  getMetaCoverage: vi.fn(), postSignalScore: vi.fn(), postMetaLabelTrain: vi.fn(),
}));
import * as client from "@/api/client";

const MODEL_AAPL_RSI = {
  model_id: "meta_aapl_c", metadata: { ticker: "AAPL", label_type: "meta_label" },
  extras: { meta_label: {
    primary: { source: "strategy", strategy_name: "rsi_strategy" },
    cv: { metrics: { auc_mean: 0.42, precision_at_50: 0.42, expected_R_when_trade: -0.05 } },
    barrier: { tp_k: 2, sl_k: 1 }, event_count: 492,
  }},
};
const MODEL_MSFT_RSI = {
  model_id: "meta_msft_a", metadata: { ticker: "MSFT", label_type: "meta_label" },
  extras: { meta_label: {
    primary: { source: "strategy", strategy_name: "rsi_strategy" },
    cv: { metrics: { auc_mean: 0.62, precision_at_50: 0.55, expected_R_when_trade: 0.02 } },
    barrier: { tp_k: 2, sl_k: 1 }, event_count: 483,
  }},
};

beforeEach(() => {
  vi.clearAllMocks();
  client.getMetaLabelModels.mockImplementation((ticker) => {
    if (ticker === "AAPL") return Promise.resolve([MODEL_AAPL_RSI]);
    if (ticker === "MSFT") return Promise.resolve([MODEL_MSFT_RSI]);
    return Promise.resolve([]);
  });
});

describe("StrategyMatrix", () => {
  it("renders a row per ticker with 4 strategy columns", async () => {
    render(<StrategyMatrix tickers={["AAPL", "MSFT"]} onSelect={() => {}} />, { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(screen.getByText("AAPL")).toBeInTheDocument());
    ["ma_cross", "rsi_strategy", "bollinger_breakout", "sentiment_driven"].forEach((s) => {
      expect(screen.getByText(s)).toBeInTheDocument();
    });
    expect(screen.getByText("MSFT")).toBeInTheDocument();
  });

  it("shows 'Train' CTA for cells without a model", async () => {
    render(<StrategyMatrix tickers={["AAPL"]} onSelect={() => {}} />, { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(screen.getAllByText(/Train/i).length).toBeGreaterThan(0));
  });

  it("calls onSelect with cell details when a cell is clicked", async () => {
    const onSelect = vi.fn();
    render(<StrategyMatrix tickers={["MSFT"]} onSelect={onSelect} />, { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(screen.getByText(/0\.62/)).toBeInTheDocument());
    fireEvent.click(screen.getByText(/0\.62/).closest("[data-cell]"));
    expect(onSelect).toHaveBeenCalledWith(
      expect.objectContaining({ ticker: "MSFT", strategy: "rsi_strategy", model_id: "meta_msft_a" })
    );
  });

  it("applies warn data-variant for cells with AUC < 0.5", async () => {
    const { container } = render(<StrategyMatrix tickers={["AAPL"]} onSelect={() => {}} />, { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(container.querySelector("[data-variant='warn']")).toBeTruthy());
  });
});
