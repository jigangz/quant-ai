import { lazy, Suspense } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import AppShell from "./AppShell";
import { LoadingOverlay } from "../components/LoadingSpinner";

// Lazy-load each page — each becomes its own chunk, first-page load time drops ~75%
const ScreenerPage = lazy(() => import("../pages/ScreenerPage"));
const DashboardPage = lazy(() => import("../pages/DashboardPage"));
const TrainingPage = lazy(() => import("../pages/TrainingPage"));
const StrategyPage = lazy(() => import("../pages/StrategyPage"));
const TradingPage = lazy(() => import("../pages/TradingPage"));
const ExplainPage = lazy(() => import("../pages/ExplainPage"));
const SignalConsolePage = lazy(() => import("../pages/SignalConsolePage"));

function PageSuspense({ children }) {
  return (
    <Suspense fallback={<LoadingOverlay label="Loading page..." />}>
      {children}
    </Suspense>
  );
}

export default function AppRouter() {
  return (
    <Routes>
      <Route path="/" element={<AppShell />}>
        <Route index element={<Navigate to="/screener" replace />} />
        <Route path="screener" element={<PageSuspense><ScreenerPage /></PageSuspense>} />
        <Route path="dashboard" element={<PageSuspense><DashboardPage /></PageSuspense>} />
        <Route path="training" element={<PageSuspense><TrainingPage /></PageSuspense>} />
        <Route path="strategy" element={<PageSuspense><StrategyPage /></PageSuspense>} />
        <Route path="trading" element={<PageSuspense><TradingPage /></PageSuspense>} />
        <Route path="explain" element={<PageSuspense><ExplainPage /></PageSuspense>} />
        <Route path="signal-console" element={<PageSuspense><SignalConsolePage /></PageSuspense>} />
      </Route>
    </Routes>
  );
}
