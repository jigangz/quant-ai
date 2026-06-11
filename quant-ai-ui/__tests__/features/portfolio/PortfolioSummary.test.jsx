import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import PortfolioSummary from "@/features/portfolio/PortfolioSummary";

const DATA = {
  success: true,
  overall_signal: "bullish",
  bullish_count: 2,
  bearish_count: 1,
  summary: "2 of 4 tickers look bullish on the 5-day horizon.",
  analyses: [
    { ticker: "AAPL", prediction: "up", probability: 0.62, signal: "bullish", top_driver: "momentum" },
    { ticker: "MSFT", prediction: "up", probability: 0.58, signal: "bullish", top_driver: "momentum" },
    { ticker: "TSLA", prediction: "down", probability: 0.41, signal: "bearish", top_driver: "momentum" },
    { ticker: "AMZN", prediction: "up", probability: 0.5, signal: "neutral", top_driver: "momentum" },
  ],
};

function renderIt(data) {
  return render(
    <MemoryRouter>
      <PortfolioSummary data={data} />
    </MemoryRouter>
  );
}

describe("PortfolioSummary", () => {
  it("renders the distribution counts (bullish/neutral/bearish)", () => {
    renderIt(DATA);
    expect(
      screen.getByLabelText("2 bullish, 1 neutral, 1 bearish")
    ).toBeInTheDocument();
  });

  it("renders a card per ticker with an Analyze deep-link to the dashboard", () => {
    renderIt(DATA);
    for (const t of ["AAPL", "MSFT", "TSLA", "AMZN"]) {
      expect(screen.getByText(t)).toBeInTheDocument();
    }
    const links = screen.getAllByRole("link", { name: /analyze/i });
    expect(links).toHaveLength(4);
    expect(links[0]).toHaveAttribute("href", "/dashboard?ticker=AAPL");
  });

  it("renders the narrative summary", () => {
    renderIt(DATA);
    expect(screen.getByText(/2 of 4 tickers look bullish/)).toBeInTheDocument();
  });

  it("renders nothing when the response is unsuccessful", () => {
    const { container } = renderIt({ success: false, error: "No model available" });
    expect(container).toBeEmptyDOMElement();
  });
});
