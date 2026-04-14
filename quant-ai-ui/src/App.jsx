import { Routes, Route, NavLink, Navigate } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Explain from "./pages/Explain";
import Training from "./pages/Training";
import Screener from "./pages/Screener";
import Strategy from "./pages/Strategy";
import Trading from "./pages/Trading";

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
          <Route path="/screener" element={<Screener />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/training" element={<Training />} />
          <Route path="/strategy" element={<Strategy />} />
          <Route path="/trading" element={<Trading />} />
          <Route path="/explain" element={<Explain />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
