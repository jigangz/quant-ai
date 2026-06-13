import { useEffect } from "react";
import { Outlet, useLocation } from "react-router-dom";
import Sidebar from "./Sidebar";
import ErrorBoundary from "../components/ErrorBoundary";
import { TopNavBar } from "@/components/layout/TopNavBar";
import { GlobalRagButton } from "@/components/layout/GlobalRagButton";
import DemoBanner from "@/components/DemoBanner";
import Tour from "@/features/onboarding/Tour";

export default function AppShell() {
  const { pathname } = useLocation();
  const onDashboard = pathname.startsWith("/dashboard");

  // Unified light design-language across every page (the "Research" look).
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", "light");
  }, [pathname]);

  return (
    <div className="min-h-screen flex bg-background text-foreground">
      <Sidebar />
      <div className="flex-1 flex flex-col md:ml-16">
        <TopNavBar />
        <DemoBanner />
        <main className="flex-1 overflow-y-auto">
          <ErrorBoundary>
            <Outlet />
          </ErrorBoundary>
        </main>
      </div>
      <GlobalRagButton bottom={24} right={onDashboard ? 304 : 24} />
      <Tour />
    </div>
  );
}
