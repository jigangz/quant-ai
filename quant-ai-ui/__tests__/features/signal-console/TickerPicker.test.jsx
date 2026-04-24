import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import TickerPicker from "@/features/signal-console/TickerPicker";

beforeEach(() => {
  localStorage.clear();
  localStorage.setItem(
    "quant-ai:watchlist",
    JSON.stringify(["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"])
  );
});

describe("TickerPicker", () => {
  it("loads tickers from localStorage on mount", () => {
    render(<TickerPicker selected={["AAPL"]} onChange={() => {}} />);
    expect(screen.getByText("AAPL")).toBeInTheDocument();
    expect(screen.getByText("MSFT")).toBeInTheDocument();
  });

  it("toggles a ticker when clicked (multi-select)", () => {
    const onChange = vi.fn();
    render(<TickerPicker selected={["AAPL"]} onChange={onChange} />);
    fireEvent.click(screen.getByText("MSFT"));
    expect(onChange).toHaveBeenCalledWith(["AAPL", "MSFT"]);
  });

  it("caps selection at 10 tickers", () => {
    const big = Array.from({ length: 12 }, (_, i) => `T${i}`);
    localStorage.setItem("quant-ai:watchlist", JSON.stringify(big));
    const onChange = vi.fn();
    render(<TickerPicker selected={big.slice(0, 10)} onChange={onChange} />);
    fireEvent.click(screen.getByText("T10"));
    // Already at cap, onChange should NOT be called with 11 items
    expect(onChange).not.toHaveBeenCalled();
  });
});
