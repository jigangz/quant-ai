import { NavLink } from "react-router-dom";
import {
  BarChart2,
  LayoutDashboard,
  Brain,
  TrendingUp,
  LineChart,
  Lightbulb,
  Briefcase,
} from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "../components/ui/tooltip";
import { cn } from "../lib/utils";

const NAV_ITEMS = [
  { to: "/screener", label: "Screener", icon: BarChart2 },
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/portfolio", label: "Portfolio", icon: Briefcase },
  { to: "/training", label: "Training", icon: Brain },
  { to: "/strategy", label: "Strategy", icon: TrendingUp },
  { to: "/trading", label: "Trading", icon: LineChart },
  { to: "/explain", label: "Explain", icon: Lightbulb },
];

export default function Sidebar() {
  return (
    <TooltipProvider delayDuration={200}>
      <aside className="hidden md:flex flex-col w-16 h-screen fixed left-0 top-0 bg-surface-card border-r border-surface-border z-10">
        <div className="flex items-center justify-center h-14 border-b border-surface-border">
          <span className="text-accent font-bold text-lg">Q</span>
        </div>
        <nav className="flex flex-col items-center gap-1 py-3 flex-1">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <Tooltip key={to}>
              <TooltipTrigger asChild>
                <NavLink
                  to={to}
                  className={({ isActive }) =>
                    cn(
                      "flex items-center justify-center h-10 w-10 rounded-lg transition-colors",
                      isActive
                        ? "bg-accent text-white"
                        : "text-muted hover:text-foreground hover:bg-surface-hover"
                    )
                  }
                >
                  <Icon className="h-5 w-5" />
                  <span className="sr-only">{label}</span>
                </NavLink>
              </TooltipTrigger>
              <TooltipContent side="right">{label}</TooltipContent>
            </Tooltip>
          ))}
        </nav>
      </aside>
    </TooltipProvider>
  );
}
