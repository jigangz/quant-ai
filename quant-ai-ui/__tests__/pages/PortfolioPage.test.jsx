import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { makeQueryWrapper } from "../_helpers/queryWrapper";
import PortfolioPage from "@/pages/PortfolioPage";

vi.mock("@/api/client", () => ({
  agentSummary: vi.fn(),
}));

import * as client from "@/api/client";

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear(); // watchlist falls back to its default seed
});

const OK = {
  success: true,
  overall_signal: "mixed",
  bullish_count: 1,
  bearish_count: 1,
  summary: "Mixed picture across the watchlist.",
  analyses: [
    { ticker: "AAPL", prediction: "up", probability: 0.6, signal: "bullish", top_driver: "momentum" },
    { ticker: "TSLA", prediction: "down", probability: 0.4, signal: "bearish", top_driver: "momentum" },
  ],
};

function renderPage() {
  return render(<MemoryRouter><PortfolioPage /></MemoryRouter>, {
    wrapper: makeQueryWrapper(),
  });
}

describe("PortfolioPage", () => {
  it("seeds ticker chips from the watchlist default and queries the summary agent", async () => {
    client.agentSummary.mockResolvedValue(OK);
    renderPage();
    // default watchlist seed = AAPL, TSLA, MSFT, AMZN
    expect(screen.getByLabelText("Remove AAPL")).toBeInTheDocument();
    await waitFor(() => expect(client.agentSummary).toHaveBeenCalled());
    expect(client.agentSummary.mock.calls[0][0].tickers).toEqual(
      expect.arrayContaining(["AAPL", "TSLA", "MSFT", "AMZN"])
    );
  });

  it("renders the summary once loaded", async () => {
    client.agentSummary.mockResolvedValue(OK);
    renderPage();
    await waitFor(() =>
      expect(screen.getByText(/Mixed picture across the watchlist/)).toBeInTheDocument()
    );
  });

  it("removing a chip re-queries without that ticker", async () => {
    client.agentSummary.mockResolvedValue(OK);
    renderPage();
    await waitFor(() => expect(client.agentSummary).toHaveBeenCalled());
    fireEvent.click(screen.getByLabelText("Remove TSLA"));
    await waitFor(() => {
      const lastCall = client.agentSummary.mock.calls.at(-1)[0];
      expect(lastCall.tickers).not.toContain("TSLA");
    });
  });

  it("shows the no-model empty state when backend has no promoted model", async () => {
    client.agentSummary.mockResolvedValue({ success: false, error: "No model available" });
    renderPage();
    await waitFor(() =>
      expect(screen.getByText(/No promoted model yet/)).toBeInTheDocument()
    );
    expect(screen.getByRole("button", { name: /open training/i })).toBeInTheDocument();
  });
});
