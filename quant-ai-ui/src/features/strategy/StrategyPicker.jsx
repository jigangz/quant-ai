import { useStrategies } from "../../api/queries";
import { cn } from "../../lib/utils";

export default function StrategyPicker({ selected, onSelect }) {
  const { data, isLoading } = useStrategies();
  if (isLoading) return <div className="text-sm text-muted p-3">Loading...</div>;
  const items = data?.strategies || data || [];

  return (
    <div className="space-y-1">
      {items.map((s) => (
        <button
          key={s.name}
          onClick={() => onSelect(s.name)}
          className={cn(
            "w-full text-left px-3 py-2 rounded-lg text-sm transition-colors",
            selected === s.name ? "bg-accent text-accent-foreground" : "hover:bg-surface-hover text-foreground"
          )}
        >
          <div className="font-medium">{s.name}</div>
          {s.description && <div className="text-xs text-muted truncate mt-0.5">{s.description}</div>}
        </button>
      ))}
    </div>
  );
}
