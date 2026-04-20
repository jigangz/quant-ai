import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi } from "vitest";

vi.mock("@/api/client", async () => {
  const actual = await vi.importActual("@/api/client");
  return {
    ...actual,
    getMarket: vi.fn(async () => [
      { date: "2026-04-01", open: 260, high: 265, low: 258, close: 263, volume: 45e6 },
      { date: "2026-04-19", open: 268, high: 272, low: 267, close: 270.23, volume: 48e6 },
    ]),
    agentTechnical: vi.fn(async () => ({
      prediction: 1,
      probability: { up: 0.68, down: 0.32 },
      confidence: "high",
      summary: "主因 RSI 超卖反弹 + MA 金叉。",
      top_features: [{ name: "RSI 14", contribution: 0.28 }],
      signals: [{ indicator: "RSI", signal: "bullish", strength: "strong" }],
      model_id: "xgb-42",
      horizon: 5,
    })),
    agentSummary: vi.fn(async () => ({ analyses: [] })),
    getModel: vi.fn(async () => ({
      id: "xgb-42",
      model_type: "xgboost",
      metrics: { val_auc: 0.62 },
      training_run_id: 42,
      git_sha: "abc1234",
    })),
    getModelsForTicker: vi.fn(async () => []),
    getRelatedStocks: vi.fn(async () => ["MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]),
    getSeasonalAccuracy: vi.fn(async () => ({ monthly: null, overall: null })),
    getSentiment: vi.fn(async () => ({ news: [] })),
    ragAnswer: vi.fn(),
  };
});

import DashboardPage from "@/pages/DashboardPage";

const renderPage = (url = "/dashboard?ticker=AAPL") => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[url]}>
        <DashboardPage />
      </MemoryRouter>
    </QueryClientProvider>
  );
};

describe("DashboardPage", () => {
  it("renders 14 sections on happy path", async () => {
    renderPage();
    // Wait for AI data to load (appears after agentTechnical query resolves)
    await waitFor(() => expect(screen.getAllByText(/AI 预测/).length).toBeGreaterThan(0), { timeout: 5000 });
    expect(screen.getAllByText(/为什么这么说/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/SHAP Top 3/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/图表/).length).toBeGreaterThan(0);
    expect(screen.getAllByText("关键数据点").length).toBeGreaterThan(0);
    expect(screen.getAllByText("相关股票").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/新闻/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/历史模型对此股的预测/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/技术指标/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/季节性/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/纸上下单/).length).toBeGreaterThan(0);
  });

  it("wraps content in ThemeScope light", async () => {
    const { container } = renderPage();
    await waitFor(() => expect(screen.getAllByText("AAPL").length).toBeGreaterThan(0));
    expect(container.querySelector("[data-theme='light']")).not.toBeNull();
  });
});
