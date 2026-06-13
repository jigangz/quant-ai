import { useEffect, useState } from "react";

const STORAGE_KEY = "quant-ai:watchlist";
const MAX_SELECTED = 10;

export default function TickerPicker({ selected = [], onChange }) {
  const [available, setAvailable] = useState([]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setAvailable(raw ? JSON.parse(raw) : []);
    } catch {
      setAvailable([]);
    }
  }, []);

  const toggle = (t) => {
    if (selected.includes(t)) {
      onChange(selected.filter((x) => x !== t));
    } else {
      if (selected.length >= MAX_SELECTED) return;
      onChange([...selected, t]);
    }
  };

  return (
    <div className="flex flex-wrap gap-2 p-3 bg-surface-muted rounded-lg">
      <div className="text-xs text-muted mr-2 self-center">Watchlist:</div>
      {available.map((t) => (
        <button
          key={t}
          type="button"
          onClick={() => toggle(t)}
          className={`px-2 py-1 text-xs rounded border ${
            selected.includes(t)
              ? "bg-emerald-600/20 border-emerald-600/50 text-emerald-300"
              : "bg-surface-muted border-surface-border text-muted hover:text-foreground"
          }`}
        >
          {t}
        </button>
      ))}
      <div className="text-xs text-muted ml-auto self-center">
        {selected.length}/{MAX_SELECTED} selected
      </div>
    </div>
  );
}
