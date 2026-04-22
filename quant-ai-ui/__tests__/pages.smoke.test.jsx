import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

// Mock all API queries to prevent network calls
vi.mock("../src/api/queries", () => ({
  useMarket: () => ({ data: null, isLoading: false, error: null }),
  useScreenerTickers: () => ({ data: [], isLoading: false, error: null, refetch: vi.fn(), isFetching: false }),
  useExplain: () => ({ data: null, isLoading: false, error: null }),
  useSimilarCases: () => ({ data: null, isLoading: false, error: null }),
  useRuns: () => ({ data: [], isLoading: false, error: null }),
  useRunStatus: () => ({ data: null }),
  useModels: () => ({ data: [], isLoading: false, error: null }),
  useModelTypes: () => ({ data: [], isLoading: false }),
  usePromotedModel: () => ({ data: null, isLoading: false }),
  useFeatureGroups: () => ({ data: null }),
  useStrategies: () => ({ data: { strategies: [] }, isLoading: false, error: null }),
  useStrategy: () => ({ data: null, isLoading: false, error: null }),
  usePortfolio: () => ({ data: null, isLoading: false, error: null }),
  useOrders: () => ({ data: { orders: [] }, isLoading: false, error: null }),
  useTrades: () => ({ data: { trades: [] }, isLoading: false, error: null }),
  usePredict: () => ({ mutate: vi.fn(), isPending: false }),
  useTrain: () => ({ mutate: vi.fn(), isPending: false }),
  usePromoteModel: () => ({ mutate: vi.fn(), isPending: false }),
  useDemoteModel: () => ({ mutate: vi.fn(), isPending: false }),
  useGenerateSignals: () => ({ mutate: vi.fn(), isPending: false }),
  useStrategyBacktest: () => ({ mutate: vi.fn(), isPending: false }),
  usePlaceOrder: () => ({ mutate: vi.fn(), isPending: false }),
  useCancelOrder: () => ({ mutate: vi.fn(), isPending: false }),
  useResetPortfolio: () => ({ mutate: vi.fn(), isPending: false }),
  useOptimizeModel: () => ({ mutate: vi.fn(), isPending: false }),
  useOptimizeStrategy: () => ({ mutate: vi.fn(), isPending: false }),
  useAgentTechnical: () => ({ data: null, isLoading: false, error: null }),
  useModelMeta: () => ({ data: null, isLoading: false }),
  useModelsForTicker: () => ({ data: [], isLoading: false }),
  useRelatedStocks: () => ({ data: [], isLoading: false }),
  useSeasonalAccuracy: () => ({ data: null, isLoading: false }),
  useRelatedStocksSignals: () => ({ data: null, isLoading: false }),
  useSentiment: () => ({ data: null, isLoading: false }),
  // V4 P2 hooks
  usePredictedVolatility: () => ({ data: null, isLoading: false, error: null }),
  useModelsForTickerByLabelType: () => ({ data: [], isLoading: false }),
}));

// Mock the live prices hook (named export)
vi.mock("../src/features/trading/useLivePrices", () => ({
  useLivePrices: () => {},
}));

// Mock the live store
vi.mock("../src/stores/liveStore", () => ({
  useLiveStore: () => "disconnected",
}));

function wrapper(ui, initialEntry = "/") {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[initialEntry]}>
        {ui}
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe("Page smoke tests", () => {
  it("ScreenerPage renders Stock Screener heading", async () => {
    const { default: ScreenerPage } = await import("../src/pages/ScreenerPage");
    wrapper(<ScreenerPage />);
    expect(screen.getByRole("heading", { name: /stock screener/i })).toBeInTheDocument();
  });

  it("DashboardPage renders ticker heading", async () => {
    const { default: DashboardPage } = await import("../src/pages/DashboardPage");
    wrapper(<DashboardPage />, "/dashboard?ticker=AAPL");
  });

  it("TrainingPage renders Training heading", async () => {
    const { default: TrainingPage } = await import("../src/pages/TrainingPage");
    wrapper(<TrainingPage />);
    expect(screen.getByRole("heading", { name: /training/i })).toBeInTheDocument();
  });

  it("StrategyPage renders Strategy heading", async () => {
    const { default: StrategyPage } = await import("../src/pages/StrategyPage");
    wrapper(<StrategyPage />);
    expect(screen.getByRole("heading", { name: /strategy/i })).toBeInTheDocument();
  });

  it("TradingPage renders Paper Trading heading", async () => {
    const { default: TradingPage } = await import("../src/pages/TradingPage");
    wrapper(<TradingPage />);
    expect(screen.getByRole("heading", { name: /paper trading/i })).toBeInTheDocument();
  });

  it("ExplainPage renders Model Explainability heading", async () => {
    const { default: ExplainPage } = await import("../src/pages/ExplainPage");
    wrapper(<ExplainPage />);
    expect(screen.getByRole("heading", { name: /model explainability/i })).toBeInTheDocument();
  });
});
