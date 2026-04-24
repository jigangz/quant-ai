import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { makeQueryWrapper } from "../../_helpers/queryWrapper";
import StrategyCard from "@/features/strategy/StrategyCard";

vi.mock("@/api/client", () => ({
  getMetaCoverage: vi.fn(),
  getMetaLabelModels: vi.fn(), postSignalScore: vi.fn(), postMetaLabelTrain: vi.fn(),
}));
import * as client from "@/api/client";

beforeEach(() => vi.clearAllMocks());

const FIXTURE_STRATEGY = {
  name: "rsi_strategy",
  description: "RSI oversold/overbought trigger strategy",
  version: "1.0.0",
};

describe("StrategyCard meta badge", () => {
  it("shows badge when coverage exists", async () => {
    client.getMetaCoverage.mockResolvedValue({
      count: 3, max_auc: 0.62, avg_auc: 0.55, tickers: ["MSFT", "GOOGL", "AAPL"],
    });
    render(
      <MemoryRouter>
        <StrategyCard strategy={FIXTURE_STRATEGY} />
      </MemoryRouter>,
      { wrapper: makeQueryWrapper() },
    );
    await waitFor(() => expect(screen.getByText(/Meta/i)).toBeInTheDocument());
  });

  it("hides badge when coverage is empty", async () => {
    client.getMetaCoverage.mockResolvedValue({
      count: 0, max_auc: null, avg_auc: null, tickers: [], models: [],
    });
    render(
      <MemoryRouter>
        <StrategyCard strategy={FIXTURE_STRATEGY} />
      </MemoryRouter>,
      { wrapper: makeQueryWrapper() },
    );
    // Resolves with count=0 → badge renders null → no "Meta" label
    await waitFor(() => expect(screen.queryByText(/Meta ✓/i)).toBeNull());
  });
});
