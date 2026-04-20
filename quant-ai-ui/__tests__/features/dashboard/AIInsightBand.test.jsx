import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { AIInsightBand } from "@/features/dashboard/AIInsightBand";

const mockTechnical = {
  prediction: 1,
  probability: { up: 0.68, down: 0.32 },
  confidence: "high",
  summary: "主因 RSI 超卖反弹 + MA 金叉 + 正面新闻情绪。",
  top_features: [
    { name: "RSI 14", contribution: 0.28, direction: "up" },
    { name: "MA 10", contribution: 0.21, direction: "up" },
    { name: "情绪", contribution: 0.12, direction: "up" },
  ],
  horizon: 5,
};

describe("AIInsightBand", () => {
  it("renders 3 cards", () => {
    render(<AIInsightBand data={mockTechnical} />);
    expect(screen.getByText(/AI 预测/)).toBeInTheDocument();
    expect(screen.getByText(/为什么这么说/)).toBeInTheDocument();
    expect(screen.getByText(/SHAP Top 3/)).toBeInTheDocument();
  });

  it("renders bullish direction and high confidence", () => {
    render(<AIInsightBand data={mockTechnical} />);
    expect(screen.getByText(/看涨/)).toBeInTheDocument();
    expect(screen.getByText(/高/)).toBeInTheDocument();
  });

  it("renders top 3 features", () => {
    render(<AIInsightBand data={mockTechnical} />);
    expect(screen.getByText("RSI 14")).toBeInTheDocument();
    expect(screen.getByText("MA 10")).toBeInTheDocument();
    expect(screen.getByText("情绪")).toBeInTheDocument();
  });

  it("shows loading skeleton when data null", () => {
    render(<AIInsightBand data={null} isLoading />);
    expect(screen.getByTestId("ai-band-skeleton")).toBeInTheDocument();
  });
});
