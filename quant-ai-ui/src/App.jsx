import { Routes, Route, NavLink, Navigate } from "react-router-dom";
import DashboardPage from "./pages/DashboardPage";
import ExplainPage from "./pages/ExplainPage";
import TrainingPage from "./pages/TrainingPage";
import ScreenerPage from "./pages/ScreenerPage";
import StrategyPage from "./pages/StrategyPage";
import TradingPage from "./pages/TradingPage";

const NAV = [
  { to: "/screener", label: "Screener" },
  { to: "/dashboard", label: "Dashboard" },
  { to: "/training", label: "Training" },
  { to: "/strategy", label: "Strategy" },
  { to: "/trading", label: "Trading" },
  { to: "/explain", label: "Explain" },
];

function App() {
  return (
    <div className="min-h-screen bg-surface">
      <nav className="flex items-center justify-between px-6 py-3 bg-surface-card border-b border-gray-700">
        <span className="text-lg font-bold text-white">Quant AI</span>
        <div className="flex gap-1">
          {NAV.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `px-3 py-1.5 rounded text-sm transition ${
                  isActive
                    ? "bg-accent text-white"
                    : "text-gray-400 hover:text-white hover:bg-surface-hover"
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </div>
      </nav>
      <main className="max-w-7xl mx-auto px-6 py-6">
        <Routes>
          <Route path="/" element={<Navigate to="/screener" replace />} />
          <Route path="/screener" element={<ScreenerPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/strategy" element={<StrategyPage />} />
          <Route path="/trading" element={<TradingPage />} />
          <Route path="/explain" element={<ExplainPage />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
