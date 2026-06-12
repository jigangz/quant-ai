import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { GaugesSection } from "@/features/dashboard/GaugesSection";

describe("GaugesSection", () => {
  it("renders 3 gauges", () => {
    render(<GaugesSection prediction={1} probability={{ up: 0.68 }} confidence="high" signals={[]} />);
    expect(screen.getByText("Oscillators (RSI/MACD)")).toBeInTheDocument();
    expect(screen.getByText(/AI Model Summary/)).toBeInTheDocument();
    expect(screen.getByText("Moving Averages")).toBeInTheDocument();
  });

  it("maps bullish prediction+high to Strong Buy for AI gauge", () => {
    render(<GaugesSection prediction={1} probability={{ up: 0.8 }} confidence="high" signals={[]} />);
    const labels = screen.getAllByText(/Buy|Sell|Neutral|Strong/);
    expect(labels.some((el) => el.textContent === "Strong Buy")).toBe(true);
  });
});
