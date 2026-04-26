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
    expect(screen.getByPlaceholderText(/搜索/)).toBeInTheDocument();
  });

  it("shows navigation links", () => {
    renderNav();
    expect(screen.getByText("市场")).toBeInTheDocument();
    expect(screen.getByText("研究")).toBeInTheDocument();
    expect(screen.getByText("模型")).toBeInTheDocument();
  });

  it("renders Leaderboard link", () => {
    renderNav();
    expect(screen.getByText(/榜单/)).toBeInTheDocument();
  });

  it("renders Ablation link", () => {
    renderNav();
    expect(screen.getByText(/消融/)).toBeInTheDocument();
  });
});
