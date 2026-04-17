import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Search } from "lucide-react";
import { Input } from "./ui/input";
import { Button } from "./ui/button";

const SUGGESTIONS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "JPM"];

export default function TickerSearch({ defaultTicker = "" }) {
  const [value, setValue] = useState(defaultTicker);
  const navigate = useNavigate();

  const submit = (t) => {
    const ticker = (t || value).trim().toUpperCase();
    if (ticker) navigate(`/dashboard?ticker=${ticker}`);
  };

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      className="flex items-center gap-2"
    >
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted" />
        <Input
          value={value}
          onChange={(e) => setValue(e.target.value.toUpperCase())}
          placeholder="Search ticker..."
          className="pl-9 w-48"
        />
      </div>
      <Button type="submit" size="sm">
        Go
      </Button>
      <div className="flex gap-1">
        {SUGGESTIONS.slice(0, 4).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => submit(t)}
            className="text-xs px-2 py-1 rounded bg-surface-muted hover:bg-surface-hover text-muted hover:text-foreground transition-colors"
          >
            {t}
          </button>
        ))}
      </div>
    </form>
  );
}
