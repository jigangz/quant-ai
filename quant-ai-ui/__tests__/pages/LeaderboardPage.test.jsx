import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { makeQueryWrapper } from "../_helpers/queryWrapper";
import LeaderboardPage from "@/pages/LeaderboardPage";

vi.mock("@/api/client", () => ({
  getModels: vi.fn(),
  getModelAccuracy: vi.fn().mockResolvedValue({ stats: { hit_rate: null } }),
  postAblationRun: vi.fn(),
}));

import * as client from "@/api/client";

beforeEach(() => {
  vi.clearAllMocks();
  client.getModels.mockResolvedValue([
    { id: "m1", name: "Direction Model 1", model_type: "xgboost",
      label_type: "direction", tickers: ["MSFT"],
      metrics: { test_auc: 0.62 }, created_at: "2026-04-20" },
    { id: "m2", name: "Direction Model 2", model_type: "logistic",
      label_type: "direction", tickers: ["AAPL"],
      metrics: { test_auc: 0.55 }, created_at: "2026-04-19" },
  ]);
});

describe("LeaderboardPage", () => {
  it("renders 3 tabs (direction/vol/meta)", () => {
    render(<MemoryRouter><LeaderboardPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    expect(screen.getByText(/direction/i)).toBeInTheDocument();
    expect(screen.getByText(/volatility/i)).toBeInTheDocument();
    expect(screen.getByText(/meta-label|meta_label/i)).toBeInTheDocument();
  });

  it("renders model rows with metrics", async () => {
    render(<MemoryRouter><LeaderboardPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(screen.getByText(/Direction Model 1/i)).toBeInTheDocument());
    expect(screen.getByText(/0\.62/)).toBeInTheDocument();
  });

  it("sorts by primary metric desc (best first)", async () => {
    render(<MemoryRouter><LeaderboardPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    await waitFor(() => expect(screen.getByText(/Direction Model 1/i)).toBeInTheDocument());
    const rows = document.querySelectorAll("tbody tr");
    // First row should be "Direction Model 1" (auc 0.62) before "Direction Model 2" (0.55)
    expect(rows[0]?.textContent).toMatch(/Direction Model 1/);
  });

  it("switches tab to volatility on click and re-queries", async () => {
    render(<MemoryRouter><LeaderboardPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    fireEvent.click(screen.getByText(/volatility/i));
    await waitFor(() => expect(client.getModels).toHaveBeenCalledWith(
      expect.objectContaining({ label_type: "volatility" })
    ));
  });
});
