import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { GaugesSection } from "@/features/dashboard/GaugesSection";

describe("GaugesSection", () => {
  it("renders 3 gauges", () => {
    render(<GaugesSection prediction={1} probability={{ up: 0.68 }} confidence="high" signals={[]} />);
    expect(screen.getByText("震荡指标 (RSI/MACD)")).toBeInTheDocument();
    expect(screen.getByText(/AI 模型总结/)).toBeInTheDocument();
    expect(screen.getByText("移动平均线")).toBeInTheDocument();
  });

  it("maps bullish prediction+high to 强烈买入 for AI gauge", () => {
    render(<GaugesSection prediction={1} probability={{ up: 0.8 }} confidence="high" signals={[]} />);
    const labels = screen.getAllByText(/买入|卖出|中立|强烈/);
    expect(labels.some((el) => el.textContent === "强烈买入")).toBe(true);
  });
});
