import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import Tour from "@/features/onboarding/Tour";

beforeEach(() => localStorage.clear());

function renderTour() {
  return render(
    <MemoryRouter>
      <Tour />
    </MemoryRouter>
  );
}

describe("Tour", () => {
  it("shows step 1 (Screener) on first visit", () => {
    renderTour();
    expect(screen.getByRole("dialog", { name: /product tour/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Screener" })).toBeInTheDocument();
    expect(screen.getByText(/1\/4/)).toBeInTheDocument();
  });

  it("Next advances through steps and the last step offers Get started", () => {
    renderTour();
    fireEvent.click(screen.getByRole("button", { name: /next/i }));
    expect(screen.getByRole("heading", { name: "Dashboard" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /next/i }));
    expect(screen.getByRole("heading", { name: "Portfolio" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /next/i }));
    expect(
      screen.getByRole("heading", { name: "Leaderboard & Ablation" })
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /get started/i })).toBeInTheDocument();
  });

  it("Skip dismisses forever (persists across re-renders)", () => {
    const { unmount } = renderTour();
    fireEvent.click(screen.getByRole("button", { name: /skip/i }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    unmount();
    renderTour();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
