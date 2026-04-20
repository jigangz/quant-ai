import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { MigrationBanner } from "@/theme/MigrationBanner";

describe("MigrationBanner", () => {
  it("renders nothing when current path is in migrated list", () => {
    const { container } = render(
      <MigrationBanner currentPath="/dashboard" migratedPaths={["/dashboard"]} />
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders banner with unmigrated page names when current path is not migrated", () => {
    render(
      <MigrationBanner
        currentPath="/training"
        migratedPaths={["/dashboard"]}
        allPaths={[
          { path: "/dashboard", label: "Dashboard" },
          { path: "/training", label: "Training" },
          { path: "/strategy", label: "Strategy" },
        ]}
      />
    );
    expect(screen.getByText(/Training/)).toBeInTheDocument();
    expect(screen.getByText(/Strategy/)).toBeInTheDocument();
  });
});
