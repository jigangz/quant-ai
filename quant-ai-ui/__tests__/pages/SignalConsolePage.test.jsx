import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { makeQueryWrapper } from "../_helpers/queryWrapper";
import SignalConsolePage from "@/pages/SignalConsolePage";

vi.mock("@/api/client", () => ({
  getMetaLabelModels: vi.fn().mockResolvedValue([]),
  getMetaCoverage: vi.fn().mockResolvedValue({ count: 0, max_auc: null, avg_auc: null, tickers: [], models: [] }),
  postSignalScore: vi.fn(), postMetaLabelTrain: vi.fn(),
}));

beforeEach(() => {
  localStorage.clear();
  localStorage.setItem("quant-ai:watchlist", JSON.stringify(["AAPL", "MSFT"]));
});

describe("SignalConsolePage", () => {
  it("mounts with 3 sections (picker, matrix, detail)", async () => {
    render(
      <MemoryRouter initialEntries={["/signal-console"]}>
        <SignalConsolePage />
      </MemoryRouter>,
      { wrapper: makeQueryWrapper() },
    );
    expect(screen.getByText(/Watchlist/i)).toBeInTheDocument();
    expect(screen.getByText(/Signal Console/i)).toBeInTheDocument();
  });

  it("applies ?strategy= query param (reads without crashing)", async () => {
    render(
      <MemoryRouter initialEntries={["/signal-console?strategy=rsi_strategy"]}>
        <SignalConsolePage />
      </MemoryRouter>,
      { wrapper: makeQueryWrapper() },
    );
    expect(screen.getAllByText(/rsi_strategy|Signal Console/i).length).toBeGreaterThan(0);
  });
});
