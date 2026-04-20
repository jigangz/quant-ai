import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { Gauge } from "@/features/dashboard/Gauge";

describe("Gauge", () => {
  it("renders label and scoreLabel", () => {
    render(<Gauge label="震荡指标" score={1} scoreLabel="买入" />);
    expect(screen.getByText("震荡指标")).toBeInTheDocument();
    expect(screen.getByText("买入")).toBeInTheDocument();
  });

  it("has accessible role=meter", () => {
    render(<Gauge label="AI" score={2} scoreLabel="强烈买入" />);
    const meter = screen.getByRole("meter");
    expect(meter).toHaveAttribute("aria-valuemin", "-2");
    expect(meter).toHaveAttribute("aria-valuemax", "2");
    expect(meter).toHaveAttribute("aria-valuenow", "2");
  });

  it("clamps out-of-bound scores", () => {
    render(<Gauge label="test" score={10} scoreLabel="x" />);
    expect(screen.getByRole("meter")).toHaveAttribute("aria-valuenow", "2");
  });
});
