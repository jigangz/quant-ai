import { useState } from "react";
import { useLeaderboard } from "@/api/leaderboardQueries";
import LeaderboardTable from "@/features/leaderboard/LeaderboardTable";

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
        <p className="text-sm text-slate-400">
          Active models per V4 multi-task target, sorted by primary metric. Live hit rate from prediction_log
          (30-day window).
        </p>
      </header>

      <nav className="flex gap-2 border-b border-slate-800">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setActive(t.id)}
            className={`px-3 py-2 text-sm border-b-2 transition-colors ${
              active === t.id
                ? "border-emerald-500 text-emerald-300"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {isLoading ? (
        <div className="p-8 text-sm text-slate-500 text-center">Loading...</div>
      ) : (
        <LeaderboardTable models={models} labelType={active} />
      )}
    </div>
  );
}
