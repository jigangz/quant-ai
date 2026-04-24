import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { makeQueryWrapper } from "../_helpers/queryWrapper";
import MetaLabelCoverageBadge from "@/components/MetaLabelCoverageBadge";

vi.mock("@/api/client", () => ({
  getMetaCoverage: vi.fn(),
  getMetaLabelModels: vi.fn(),
  postSignalScore: vi.fn(),
  postMetaLabelTrain: vi.fn(),
}));
import * as client from "@/api/client";

beforeEach(() => vi.clearAllMocks());

function makeWrapper() {
  const QueryWrapper = makeQueryWrapper();
  return ({ children }) => (
    <QueryWrapper>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryWrapper>
  );
}

describe("MetaLabelCoverageBadge", () => {
  it("renders count and max AUC when coverage exists", async () => {
    client.getMetaCoverage.mockResolvedValue({
      count: 3, max_auc: 0.619, avg_auc: 0.549, tickers: ["MSFT", "GOOGL", "AAPL"],
    });
    render(<MetaLabelCoverageBadge strategyName="rsi_strategy" />, { wrapper: makeWrapper() });
    await waitFor(() => expect(screen.getByText(/Meta/i)).toBeInTheDocument());
    expect(screen.getByText(/Meta/i).textContent).toMatch(/3/);
    expect(screen.getByText(/Meta/i).textContent).toMatch(/0\.62/);
  });

  it("renders nothing when coverage is zero (count=0 or 404)", async () => {
    client.getMetaCoverage.mockResolvedValue({
      count: 0, max_auc: null, avg_auc: null, tickers: [],
    });
    const { container } = render(
      <MetaLabelCoverageBadge strategyName="rsi_strategy" />,
      { wrapper: makeWrapper() },
    );
    await waitFor(() => {
      expect(container.textContent.trim()).toBe("");
    });
  });

  it("applies warning tint when avg_auc < 0.5", async () => {
    client.getMetaCoverage.mockResolvedValue({
      count: 1, max_auc: 0.42, avg_auc: 0.42, tickers: ["AAPL"],
    });
    const { container } = render(
      <MetaLabelCoverageBadge strategyName="rsi_strategy" />,
      { wrapper: makeWrapper() },
    );
    await waitFor(() => expect(container.querySelector("[data-variant='warn']")).toBeTruthy());
  });
});
