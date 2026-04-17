import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";
import ErrorBoundary from "../components/ErrorBoundary";

export default function AppShell() {
  return (
    <div className="flex min-h-screen bg-surface">
      <Sidebar />
      <main className="flex-1 md:ml-16 px-6 py-6 max-w-none">
        <ErrorBoundary>
          <Outlet />
        </ErrorBoundary>
      </main>
    </div>
  );
}
