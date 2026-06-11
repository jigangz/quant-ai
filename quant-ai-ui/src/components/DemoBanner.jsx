import { useState } from "react";
import { X, Rocket } from "lucide-react";

const KEY = "quant-ai:demo-banner-dismissed";

/**
 * Cold-visitor banner: the live demo runs on a free-tier backend that sleeps,
 * so the first API call can take ~30s. Saying so beats looking broken.
 */
export default function DemoBanner() {
  const [open, setOpen] = useState(() => {
    try {
      return localStorage.getItem(KEY) !== "1";
    } catch {
      return true;
    }
  });

  if (!open) return null;

  const dismiss = () => {
    setOpen(false);
    try {
      localStorage.setItem(KEY, "1");
    } catch {
      /* private mode — banner just won't persist dismissal */
    }
  };

  return (
    <div
      role="note"
      aria-label="demo notice"
      className="flex items-center gap-2 bg-accent/10 border-b border-accent/20 px-4 py-2 text-xs text-foreground"
    >
      <Rocket className="h-3.5 w-3.5 text-accent shrink-0" />
      <span>
        Live demo on a free-tier backend — the first request can take ~30s while the
        server wakes up. Try <strong>AAPL</strong>, <strong>MSFT</strong> or{" "}
        <strong>NVDA</strong>.
      </span>
      <button
        type="button"
        aria-label="Dismiss demo notice"
        onClick={dismiss}
        className="ml-auto text-muted hover:text-foreground"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
