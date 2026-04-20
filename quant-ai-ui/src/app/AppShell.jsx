import { Outlet, useLocation } from "react-router-dom";
import Sidebar from "./Sidebar";
import ErrorBoundary from "../components/ErrorBoundary";
import { TopNavBar } from "@/components/layout/TopNavBar";
import { MigrationBanner } from "@/theme/MigrationBanner";
import { GlobalRagButton } from "@/components/layout/GlobalRagButton";

const MIGRATED_PATHS = ["/dashboard"];
const ALL_PATHS = [
  { path: "/dashboard", label: "Dashboard" },
  { path: "/screener", label: "Screener" },
  { path: "/training", label: "Training" },
  { path: "/strategy", label: "Strategy" },
  { path: "/trading", label: "Paper Trading" },
  { path: "/explain", label: "Explain" },
];

export default function AppShell() {
  const { pathname } = useLocation();
  const onDashboard = pathname.startsWith("/dashboard");
  return (
    <div className="min-h-screen flex bg-background text-foreground">
      <Sidebar />
      <div className="flex-1 flex flex-col md:ml-16">
        <TopNavBar />
        <MigrationBanner currentPath={pathname} migratedPaths={MIGRATED_PATHS} allPaths={ALL_PATHS} />
        <main className="flex-1 overflow-y-auto">
          <ErrorBoundary>
            <Outlet />
          </ErrorBoundary>
        </main>
      </div>
      <GlobalRagButton bottom={24} right={onDashboard ? 304 : 24} />
    </div>
  );
}
