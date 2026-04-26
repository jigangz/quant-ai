import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { makeQueryWrapper } from "../_helpers/queryWrapper";
import AblationPage from "@/pages/AblationPage";

vi.mock("@/api/client", () => ({
  getModels: vi.fn(),
  getModelAccuracy: vi.fn(),
  postAblationRun: vi.fn(),
}));

import * as client from "@/api/client";

beforeEach(() => vi.clearAllMocks());

describe("AblationPage", () => {
  it("renders form with ticker input and run button", () => {
    render(<MemoryRouter><AblationPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    expect(screen.getByLabelText(/ticker/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /run ablation/i })).toBeInTheDocument();
  });

  it("posts ablation run with default 3 targets and 2 feature sets", async () => {
    client.postAblationRun.mockResolvedValue({
      ticker: "MSFT", matrix: {}, summary: { interpretation: "" },
      elapsed_seconds: 1.2,
    });
    render(<MemoryRouter><AblationPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    fireEvent.change(screen.getByLabelText(/ticker/i), { target: { value: "MSFT" } });
    fireEvent.click(screen.getByRole("button", { name: /run ablation/i }));
    await waitFor(() => expect(client.postAblationRun).toHaveBeenCalled());
    const payload = client.postAblationRun.mock.calls[0][0];
    expect(payload.ticker).toBe("MSFT");
    expect(payload.targets).toContain("direction");
    expect(payload.feature_sets.length).toBe(2);
  });

  it("renders matrix after successful run", async () => {
    client.postAblationRun.mockResolvedValue({
      ticker: "MSFT",
      matrix: { direction: { "ta_basic": { auc: 0.5 }, "ta_basic + sentiment": { auc: 0.6 } } },
      summary: { interpretation: "ok" },
      elapsed_seconds: 1.2,
    });
    render(<MemoryRouter><AblationPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    fireEvent.change(screen.getByLabelText(/ticker/i), { target: { value: "MSFT" } });
    fireEvent.click(screen.getByRole("button", { name: /run ablation/i }));
    await waitFor(() => expect(screen.getByText(/0\.6/)).toBeInTheDocument());
  });

  it("renders summary interpretation after run", async () => {
    client.postAblationRun.mockResolvedValue({
      ticker: "MSFT", matrix: {},
      summary: { interpretation: "Sentiment lifts AUC by 6.8 points on direction." },
      elapsed_seconds: 1.2,
    });
    render(<MemoryRouter><AblationPage /></MemoryRouter>, { wrapper: makeQueryWrapper() });
    fireEvent.change(screen.getByLabelText(/ticker/i), { target: { value: "MSFT" } });
    fireEvent.click(screen.getByRole("button", { name: /run ablation/i }));
    await waitFor(() => expect(screen.getByText(/lifts AUC/i)).toBeInTheDocument());
  });
});
