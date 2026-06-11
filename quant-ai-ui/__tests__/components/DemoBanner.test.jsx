import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import DemoBanner from "@/components/DemoBanner";

beforeEach(() => localStorage.clear());

describe("DemoBanner", () => {
  it("shows the cold-start notice on first visit", () => {
    render(<DemoBanner />);
    expect(screen.getByRole("note", { name: /demo notice/i })).toBeInTheDocument();
    expect(screen.getByText(/~30s/)).toBeInTheDocument();
  });

  it("dismiss hides it and persists across re-renders", () => {
    const { unmount } = render(<DemoBanner />);
    fireEvent.click(screen.getByLabelText(/dismiss demo notice/i));
    expect(screen.queryByRole("note")).not.toBeInTheDocument();
    unmount();
    render(<DemoBanner />);
    expect(screen.queryByRole("note")).not.toBeInTheDocument();
  });
});
