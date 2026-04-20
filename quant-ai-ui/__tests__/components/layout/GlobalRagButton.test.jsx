import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect } from "vitest";
import { GlobalRagButton } from "@/components/layout/GlobalRagButton";

const renderWithClient = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <GlobalRagButton />
    </QueryClientProvider>
  );
};

describe("GlobalRagButton", () => {
  it("renders floating ❓ button", () => {
    renderWithClient();
    expect(screen.getByRole("button", { name: /RAG/ })).toBeInTheDocument();
  });

  it("opens dialog on click", async () => {
    const user = userEvent.setup();
    renderWithClient();
    await user.click(screen.getByRole("button", { name: /RAG/ }));
    expect(screen.getByPlaceholderText(/问问题/)).toBeInTheDocument();
  });
});
