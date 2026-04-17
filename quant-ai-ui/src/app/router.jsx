import { Routes, Route, Navigate } from "react-router-dom";
import AppShell from "./AppShell";
import ScreenerPage from "../pages/ScreenerPage";
import DashboardPage from "../pages/DashboardPage";
import TrainingPage from "../pages/TrainingPage";
import StrategyPage from "../pages/StrategyPage";
import TradingPage from "../pages/TradingPage";
import ExplainPage from "../pages/ExplainPage";

export default function AppRouter() {
  return (
    <Routes>
      <Route path="/" element={<AppShell />}>
        <Route index element={<Navigate to="/screener" replace />} />
        <Route path="screener" element={<ScreenerPage />} />
        <Route path="dashboard" element={<DashboardPage />} />
        <Route path="training" element={<TrainingPage />} />
        <Route path="strategy" element={<StrategyPage />} />
        <Route path="trading" element={<TradingPage />} />
        <Route path="explain" element={<ExplainPage />} />
      </Route>
    </Routes>
  );
}
