import { useState } from "react";
import { Link } from "react-router-dom";
import { X, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";

const KEY = "quant-ai:tour-done";

const STEPS = [
  {
    title: "Screener",
    to: "/screener",
    body: "Scan tickers and spot AI signals at a glance — this is the entry point.",
  },
  {
    title: "Dashboard",
    to: "/dashboard",
    body: "One-screen deep dive: model prediction, SHAP explanation and an AI-written summary.",
  },
  {
    title: "Portfolio",
    to: "/portfolio",
    body: "Your whole watchlist scored at once — bullish/bearish split plus per-ticker probabilities.",
  },
  {
    title: "Leaderboard & Ablation",
    to: "/leaderboard",
    body: "Honest scoreboard: live hit-rates from the prediction log and feature-ablation deltas.",
  },
];

/** First-visit 4-step tour. localStorage-gated; never shows again after done/skip. */
export default function Tour() {
  const [step, setStep] = useState(0);
  const [open, setOpen] = useState(() => {
    try {
      return localStorage.getItem(KEY) !== "1";
    } catch {
      return true;
    }
  });

  if (!open) return null;

  const finish = () => {
    setOpen(false);
    try {
      localStorage.setItem(KEY, "1");
    } catch {
      /* ignore */
    }
  };

  const s = STEPS[step];
  const last = step === STEPS.length - 1;

  return (
    <div
      role="dialog"
      aria-label="product tour"
      className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 w-[22rem] max-w-[calc(100vw-2rem)] rounded-xl border border-surface-border bg-surface-card shadow-xl p-4 space-y-2"
    >
      <div className="flex items-center justify-between">
        <span className="text-[11px] uppercase tracking-wide text-muted">
          Tour · {step + 1}/{STEPS.length}
        </span>
        <button
          type="button"
          aria-label="Close tour"
          onClick={finish}
          className="text-muted hover:text-foreground"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <h3 className="text-sm font-semibold text-foreground">{s.title}</h3>
      <p className="text-xs text-muted leading-relaxed">{s.body}</p>
      <Link
        to={s.to}
        className="inline-flex items-center gap-1 text-xs text-accent hover:underline"
      >
        Open {s.title} <ArrowRight className="h-3 w-3" />
      </Link>

      <div className="flex items-center justify-between pt-1">
        <button
          type="button"
          onClick={finish}
          className="text-xs text-muted hover:text-foreground"
        >
          Skip
        </button>
        {last ? (
          <Button size="sm" onClick={finish}>
            Get started
          </Button>
        ) : (
          <Button size="sm" onClick={() => setStep((n) => n + 1)}>
            Next
          </Button>
        )}
      </div>
    </div>
  );
}
