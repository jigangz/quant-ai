import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { RightRailWatchlist } from "@/components/layout/RightRailWatchlist";

vi.mock("@/api/client", () => ({
  get: vi.fn(async () => []),
  post: vi.fn(async () => ({ analyses: [] })),
  agentSummary: vi.fn(async () => ({ analyses: [] })),
  getMarket: vi.fn(async () => []),
  agentTechnical: vi.fn(async () => null),
}));

const renderRail = (props = {}) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <RightRailWatchlist currentTicker="AAPL" {...props} />
    </QueryClientProvider>
  );
};

beforeEach(() => localStorage.clear());

describe("RightRailWatchlist", () => {
  it("renders Watchlist heading + INDICES + YOUR HOLDINGS sections", () => {
    renderRail();
    expect(screen.getByText("Watchlist")).toBeInTheDocument();
    expect(screen.getByText(/INDICES/)).toBeInTheDocument();
    expect(screen.getByText(/YOUR HOLDINGS/i)).toBeInTheDocument();
  });

  it("includes current ticker in holdings", () => {
    renderRail();
    expect(screen.getAllByText("AAPL").length).toBeGreaterThan(0);
  });

  it("persists added ticker in localStorage", async () => {
    const user = userEvent.setup();
    renderRail();
    await user.click(screen.getByRole("button", { name: /add/i }));
    const input = screen.getByPlaceholderText(/ticker/i);
    await user.type(input, "NVDA{enter}");
    expect(localStorage.getItem("quant-ai:watchlist")).toContain("NVDA");
  });
});
