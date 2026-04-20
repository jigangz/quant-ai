import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { SymbolHeader } from "@/features/dashboard/SymbolHeader";

describe("SymbolHeader", () => {
  it("renders ticker, company name, exchange, price, change", () => {
    render(
      <SymbolHeader
        ticker="AAPL"
        name="Apple Inc."
        exchange="NASDAQ"
        price={270.23}
        change={2.59}
        changePct={2.59}
        lastUpdate="2026-04-19 GMT-7 13:15"
      />
    );
    expect(screen.getByText("Apple Inc.")).toBeInTheDocument();
    expect(screen.getByText("NASDAQ")).toBeInTheDocument();
    expect(screen.getByText(/270.23/)).toBeInTheDocument();
    expect(screen.getByText(/\+2.59/)).toBeInTheDocument();
  });

  it("shows price in up color when change positive", () => {
    render(<SymbolHeader ticker="AAPL" price={270.23} change={2.59} changePct={2.59} />);
    const change = screen.getByText(/\+2.59/);
    expect(change.className).toMatch(/text-up/);
  });

  it("shows price in down color when change negative", () => {
    render(<SymbolHeader ticker="AAPL" price={270.23} change={-2.59} changePct={-2.59} />);
    const change = screen.getByText(/-2.59/);
    expect(change.className).toMatch(/text-down/);
  });
});
