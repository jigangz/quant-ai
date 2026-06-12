import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, it, expect } from "vitest";
import { RelatedStocks } from "@/features/dashboard/RelatedStocks";

const mockPeers = [
  { ticker: "MSFT", name: "Microsoft", price: 465.12, signal: { direction: "bullish", confidence: "high" } },
  { ticker: "AMZN", name: "Amazon", price: 215.33, signal: { direction: "bearish", confidence: "low" } },
];

describe("RelatedStocks", () => {
  it("renders peer cards", () => {
    render(
      <MemoryRouter>
        <RelatedStocks peers={mockPeers} />
      </MemoryRouter>
    );
    expect(screen.getByText("MSFT")).toBeInTheDocument();
    expect(screen.getByText("Microsoft")).toBeInTheDocument();
    expect(screen.getByText(/465.12/)).toBeInTheDocument();
    expect(screen.getByText(/Bullish/)).toBeInTheDocument();
    expect(screen.getByText(/Bearish/)).toBeInTheDocument();
  });

  it("hides section when no peers", () => {
    const { container } = render(<RelatedStocks peers={[]} />);
    expect(container.firstChild).toBeNull();
  });
});
