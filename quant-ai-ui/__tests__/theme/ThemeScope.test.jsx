import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { ThemeScope } from "@/theme/ThemeScope";

describe("ThemeScope", () => {
  it("applies data-theme attribute to its wrapper", () => {
    render(
      <ThemeScope value="light">
        <span data-testid="child">content</span>
      </ThemeScope>
    );
    const wrapper = screen.getByTestId("child").parentElement;
    expect(wrapper).toHaveAttribute("data-theme", "light");
  });

  it("accepts dark as well", () => {
    render(
      <ThemeScope value="dark">
        <span data-testid="child">content</span>
      </ThemeScope>
    );
    const wrapper = screen.getByTestId("child").parentElement;
    expect(wrapper).toHaveAttribute("data-theme", "dark");
  });

  it("renders children", () => {
    render(
      <ThemeScope value="light">
        <div>hello</div>
      </ThemeScope>
    );
    expect(screen.getByText("hello")).toBeInTheDocument();
  });
});
