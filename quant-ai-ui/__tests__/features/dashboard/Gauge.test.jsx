import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { Gauge } from "@/features/dashboard/Gauge";

describe("Gauge", () => {
  it("renders label and scoreLabel", () => {
    render(<Gauge label="Oscillators" score={1} scoreLabel="Buy" />);
    expect(screen.getByText("Oscillators")).toBeInTheDocument();
    expect(screen.getByText("Buy")).toBeInTheDocument();
  });

  it("has accessible role=meter", () => {
    render(<Gauge label="AI" score={2} scoreLabel="Strong Buy" />);
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
