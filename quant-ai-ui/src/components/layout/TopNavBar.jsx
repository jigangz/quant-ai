import { Link } from "react-router-dom";
import { Search } from "lucide-react";

const NAV_ITEMS = [
  { label: "市场", to: "/screener" },
  { label: "研究", to: "/dashboard" },
  { label: "模型", to: "/training" },
  { label: "信号", to: "/signal-console" },
  { label: "榜单", to: "/leaderboard" },
  { label: "消融", to: "/ablation" },
  { label: "更多", to: "#" },
];

export function TopNavBar() {
  return (
    <header className="h-12 bg-surface border-b border-surface-border flex items-center gap-4 px-4">
      <Link to="/" className="text-accent font-bold text-base">Quant AI</Link>
      <nav className="flex items-center gap-4">
        {NAV_ITEMS.map((n) => (
          <Link
            key={n.label}
            to={n.to}
            className="text-[13px] text-muted hover:text-foreground transition-colors"
          >
            {n.label}
          </Link>
        ))}
      </nav>
      <div className="flex-1 flex justify-center">
        <div className="relative w-60">
          <Search size={13} className="absolute left-2 top-1/2 -translate-y-1/2 text-muted" />
          <input
            type="text"
            placeholder="🔍 搜索 (Ctrl+K)"
            className="w-full pl-7 pr-2 py-1 rounded text-xs bg-surface-muted border border-surface-border placeholder:text-muted focus:outline-none focus:ring-1 focus:ring-accent"
          />
        </div>
      </div>
      <button className="bg-accent text-accent-foreground text-xs px-3 py-1 rounded">升级</button>
      <div className="w-8 h-8 bg-surface-muted rounded-full" aria-label="User" />
    </header>
  );
}
