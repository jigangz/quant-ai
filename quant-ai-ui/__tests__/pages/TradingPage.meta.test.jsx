import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { makeQueryWrapper } from "../_helpers/queryWrapper";
import TradingPage from "@/pages/TradingPage";

vi.mock("@/api/client", () => ({
  getMetaLabelModels: vi.fn(),
  postSignalScore: vi.fn(),
  getMetaCoverage: vi.fn().mockResolvedValue({ count: 0 }),
  postMetaLabelTrain: vi.fn(),
  getMarket: vi.fn().mockResolvedValue([]),
  predict: vi.fn(),
  getPortfolio: vi.fn().mockResolvedValue({ positions: [], cash: 10000, total_value: 10000 }),
  getOrders: vi.fn().mockResolvedValue({ orders: [] }),
  getTrades: vi.fn().mockResolvedValue({ trades: [] }),
  placeOrder: vi.fn().mockResolvedValue({ order_id: "o1" }),
  cancelOrder: vi.fn().mockResolvedValue({}),
  resetPortfolio: vi.fn().mockResolvedValue({}),
  getPortfolioHistory: vi.fn().mockResolvedValue([]),
}));
import * as client from "@/api/client";

beforeEach(() => vi.clearAllMocks());

describe("TradingPage meta-label integration", () => {
  it("renders meta-label checkbox (off by default)", async () => {
    render(<MemoryRouter><TradingPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    const cb = await screen.findByLabelText(/meta-label/i);
    expect(cb).not.toBeChecked();
  });

  it("expands meta section with dropdown + threshold slider when checked", async () => {
    client.getMetaLabelModels.mockResolvedValue([
      { model_id: "meta_a", extras: { meta_label: {
        primary: { strategy_name: "rsi_strategy" },
        cv: { metrics: { auc_mean: 0.62 } }, barrier: {},
      }}, metadata: { ticker: "AAPL" }},
    ]);
    render(<MemoryRouter><TradingPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    const cb = await screen.findByLabelText(/meta-label/i);
    fireEvent.click(cb);
    await waitFor(() => expect(screen.getByLabelText(/threshold/i)).toBeInTheDocument());
  });

  it("calls postSignalScore when preview button clicked", async () => {
    client.getMetaLabelModels.mockResolvedValue([
      { model_id: "meta_a", extras: { meta_label: {
        primary: { strategy_name: "rsi_strategy" },
        cv: { metrics: { auc_mean: 0.62 } }, barrier: {},
      }}, metadata: { ticker: "AAPL" }},
    ]);
    client.postSignalScore.mockResolvedValue({
      triggered: true, signal: 1, reliability_score: 0.71,
      expected_R: 0.54, recommended_action: "trade",
      sizing_hint: { half_kelly_fraction: 0.18, raw_kelly: 0.36, cap: 0.25 },
      meta_model: { id: "meta_a", primary_source: "strategy:rsi_strategy", cv_auc: 0.62 },
    });
    render(<MemoryRouter><TradingPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    const cb = await screen.findByLabelText(/meta-label/i);
    fireEvent.click(cb);
    // wait for dropdown to appear, then select the model
    await waitFor(() => expect(screen.getByLabelText(/threshold/i)).toBeInTheDocument());
    const metaSelect = await screen.findByTestId("meta-model-select");
    fireEvent.change(metaSelect, { target: { value: "meta_a" } });
    const previewBtn = await screen.findByRole("button", { name: /preview score/i });
    fireEvent.click(previewBtn);
    await waitFor(() => expect(client.postSignalScore).toHaveBeenCalled());
  });

  it("displays score + sizing after preview", async () => {
    client.getMetaLabelModels.mockResolvedValue([
      { model_id: "meta_a", extras: { meta_label: {
        primary: { strategy_name: "rsi_strategy" },
        cv: { metrics: { auc_mean: 0.62 } }, barrier: {},
      }}, metadata: { ticker: "AAPL" }},
    ]);
    client.postSignalScore.mockResolvedValue({
      triggered: true, signal: 1, reliability_score: 0.71,
      expected_R: 0.54, recommended_action: "trade",
      sizing_hint: { half_kelly_fraction: 0.18, raw_kelly: 0.36, cap: 0.25 },
      meta_model: { id: "meta_a", primary_source: "strategy:rsi_strategy", cv_auc: 0.62 },
    });
    render(<MemoryRouter><TradingPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    fireEvent.click(await screen.findByLabelText(/meta-label/i));
    await waitFor(() => expect(screen.getByLabelText(/threshold/i)).toBeInTheDocument());
    const metaSelect = await screen.findByTestId("meta-model-select");
    fireEvent.change(metaSelect, { target: { value: "meta_a" } });
    fireEvent.click(await screen.findByRole("button", { name: /preview score/i }));
    await waitFor(() => expect(screen.getByText(/0\.71/)).toBeInTheDocument());
    expect(screen.getAllByText(/trade/i).length).toBeGreaterThan(0);
  });
});
