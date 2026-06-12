import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, it, expect } from "vitest";
import { TopNavBar } from "@/components/layout/TopNavBar";

describe("TopNavBar", () => {
  const renderNav = () =>
    render(
      <MemoryRouter>
        <TopNavBar />
      </MemoryRouter>
    );

  it("shows the Quant AI brand", () => {
    renderNav();
    expect(screen.getByText("Quant AI")).toBeInTheDocument();
  });

  it("shows search input with Ctrl+K placeholder", () => {
    renderNav();
    expect(screen.getByPlaceholderText(/Search/)).toBeInTheDocument();
  });

  it("shows navigation links", () => {
    renderNav();
    expect(screen.getByText("Markets")).toBeInTheDocument();
    expect(screen.getByText("Research")).toBeInTheDocument();
    expect(screen.getByText("Models")).toBeInTheDocument();
  });

  it("renders Leaderboard link", () => {
    renderNav();
    expect(screen.getByText(/Leaderboard/)).toBeInTheDocument();
  });

  it("renders Ablation link", () => {
    renderNav();
    expect(screen.getByText(/Ablation/)).toBeInTheDocument();
  });
});
