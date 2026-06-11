import { useState } from "react";
import { useLeaderboard } from "@/api/leaderboardQueries";
import LeaderboardTable from "@/features/leaderboard/LeaderboardTable";
import { LoadingOverlay } from "@/components/LoadingSpinner";

const TABS = [
  { id: "direction", label: "Direction" },
  { id: "volatility", label: "Volatility" },
  { id: "meta_label", label: "Meta-Label" },
];

export default function LeaderboardPage() {
  const [active, setActive] = useState("direction");
  const { data: models = [], isLoading } = useLeaderboard(active);

  return (
    <div className="p-6 space-y-4 max-w-7xl mx-auto">
      <header>
        <h1 className="text-2xl font-semibold">Leaderboard</h1>
        <p className="text-sm text-muted">
          Active models per V4 multi-task target, sorted by primary metric. Live hit rate from prediction_log
          (30-day window).
        </p>
      </header>

      <nav className="flex gap-2 border-b border-surface-border">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setActive(t.id)}
            className={`px-3 py-2 text-sm border-b-2 transition-colors ${
              active === t.id
                ? "border-accent text-accent"
                : "border-transparent text-muted hover:text-foreground"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {isLoading ? (
        <LoadingOverlay label="Loading models..." />
      ) : (
        <LeaderboardTable models={models} labelType={active} />
      )}
    </div>
  );
}
