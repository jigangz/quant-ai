import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  LayoutDashboard,
  TrendingUp,
  FlaskConical,
  Cpu,
  ChevronLeft,
  ChevronRight,
  Activity,
  Sparkles,
  Wallet,
} from 'lucide-react';

const navItems = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/prediction', label: 'Prediction', icon: TrendingUp },
  { path: '/backtest', label: 'Backtest', icon: FlaskConical },
  { path: '/strategy', label: 'Strategy', icon: Sparkles },
  { path: '/trading', label: 'Trading', icon: Wallet },
  { path: '/models', label: 'Models', icon: Cpu },
];

export default function Sidebar() {
  const [expanded, setExpanded] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <aside
      className={`fixed left-0 top-0 h-screen bg-dark-card border-r border-dark-border z-50 flex flex-col transition-all duration-200 ${
        expanded ? 'w-[220px]' : 'w-[60px]'
      }`}
    >
      {/* Logo */}
      <div className="flex items-center h-14 px-4 border-b border-dark-border gap-2">
        <Activity className="w-6 h-6 text-accent flex-shrink-0" />
        {expanded && (
          <span className="text-sm font-bold text-white whitespace-nowrap">QuantAI</span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4">
        {navItems.map(({ path, label, icon: Icon }) => {
          const active = location.pathname === path;
          return (
            <button
              key={path}
              onClick={() => navigate(path)}
              className={`w-full flex items-center gap-3 px-4 py-3 text-sm transition-colors ${
                active
                  ? 'text-accent bg-accent/10 border-r-2 border-accent'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-dark-hover'
              }`}
              title={label}
            >
              <Icon className="w-5 h-5 flex-shrink-0" />
              {expanded && <span className="whitespace-nowrap">{label}</span>}
            </button>
          );
        })}
      </nav>

      {/* Toggle */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center justify-center h-10 border-t border-dark-border text-gray-500 hover:text-gray-300 transition-colors"
      >
        {expanded ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
      </button>
    </aside>
  );
}
