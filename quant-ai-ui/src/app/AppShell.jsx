import { useEffect } from "react";
import { Outlet, useLocation } from "react-router-dom";
import Sidebar from "./Sidebar";
import ErrorBoundary from "../components/ErrorBoundary";
import { TopNavBar } from "@/components/layout/TopNavBar";
import { GlobalRagButton } from "@/components/layout/GlobalRagButton";
import DemoBanner from "@/components/DemoBanner";
import Tour from "@/features/onboarding/Tour";

// Pages that manage their own full-bleed layout (e.g. Dashboard's 1fr/280px
// grid with a right-rail watchlist). Everything else gets the standard padded,
// max-width, centered content container below — one source of truth for page gutters.
const FULL_BLEED_PATHS = ["/dashboard"];

export default function AppShell() {
  const { pathname } = useLocation();
  const onDashboard = pathname.startsWith("/dashboard");
  const fullBleed = FULL_BLEED_PATHS.some((p) => pathname.startsWith(p));

  // Unified light design-language across every page (the "Research" look).
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", "light");
  }, [pathname]);

  return (
    <div className="min-h-screen flex bg-background text-foreground">
      <Sidebar />
      <div className="flex-1 flex flex-col md:ml-16 min-w-0">
        <TopNavBar />
        <DemoBanner />
        <main className="flex-1 overflow-y-auto">
          <ErrorBoundary>
            {fullBleed ? (
              <Outlet />
            ) : (
              <div className="mx-auto w-full max-w-7xl px-4 sm:px-6 lg:px-8 py-6">
                <Outlet />
              </div>
            )}
          </ErrorBoundary>
        </main>
      </div>
      <GlobalRagButton bottom={24} right={onDashboard ? 304 : 24} />
      <Tour />
    </div>
  );
}
