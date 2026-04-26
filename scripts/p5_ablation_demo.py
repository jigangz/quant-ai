"""
P5 Ablation Demo · V4 Phase 5

Runs `ablation_service.run_ablation` (direct service call, no HTTP) on AAPL+MSFT+GOOGL
across 3 targets × 2 feature sets, writes a markdown report.

Run:
    python -m scripts.p5_ablation_demo
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from app.services.ablation_service import run_ablation


TICKERS = ("AAPL", "MSFT", "GOOGL")
TARGETS = ["direction", "volatility", "meta_label"]
FEATURE_SETS = [
    {"name": "ta_basic", "groups": ["ta_basic"]},
    {"name": "ta_basic + sentiment", "groups": ["ta_basic", "sentiment"]},
]


def main():
    rows = []
    for ticker in TICKERS:
        print(f"[{ticker}] running ablation...")
        t0 = time.time()
        try:
            result = run_ablation(
                ticker=ticker, targets=TARGETS,
                feature_sets=FEATURE_SETS,
                horizon_days=5, model_type="xgboost",
            )
            rows.append({"ticker": ticker, "result": result,
                          "elapsed": time.time() - t0})
        except Exception as e:
            rows.append({"ticker": ticker, "error": str(e),
                          "elapsed": time.time() - t0})
            print(f"  FAILED: {e}")

    out = Path("docs/benchmarks/p5_ablation_demo.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render(rows), encoding="utf-8")
    print(f"\nReport: {out}")


def _render(rows) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# V4 Pivot · Phase 5 · Ablation Demo",
        "",
        f"**Run date**: {now}",
        "**Targets**: direction · volatility · meta_label",
        "**Feature sets**: ta_basic vs ta_basic + sentiment",
        "**Model**: xgboost (default params, no Optuna — fair comparison)",
        "**Horizon**: 5 days",
        "",
        "## Per-ticker results",
        "",
    ]
    for row in rows:
        ticker = row["ticker"]
        lines.append(f"### {ticker}")
        if "error" in row:
            lines.append(f"FAILED: {row['error']}")
            lines.append("")
            continue
        result = row["result"]
        lines.append("")
        lines.append("| Target | ta_basic | ta_basic + sentiment | Δ |")
        lines.append("|---|---|---|---|")
        for target in TARGETS:
            cell0 = result["matrix"][target].get("ta_basic", {})
            cell1 = result["matrix"][target].get("ta_basic + sentiment", {})
            primary_key = {"direction": "auc",
                           "volatility": "qlike",
                           "meta_label": "auc_mean"}[target]
            if "error" in cell0:
                v0_str = f"ERR: {cell0['error'][:40]}"
            else:
                v0 = cell0.get(primary_key)
                v0_str = f"{v0:.3f}" if v0 is not None else "—"
            if "error" in cell1:
                v1_str = f"ERR: {cell1['error'][:40]}"
                delta_str = "—"
            else:
                v1 = cell1.get(primary_key)
                v1_str = f"{v1:.3f}" if v1 is not None else "—"
                delta = cell1.get(f"delta_{primary_key}")
                delta_str = f"{delta:+.3f}" if delta is not None else "—"
            lines.append(f"| {target} | {v0_str} | {v1_str} | {delta_str} |")
        lines.append("")
        lines.append(f"**Summary**: {result['summary'].get('interpretation', '—')}")
        lines.append(f"**Elapsed**: {result['elapsed_seconds']:.1f}s")
        lines.append("")
    lines.append("## Honest framing")
    lines.append("")
    lines.append(
        "Default-params XGBoost across 3 tickers. Numbers are not optimized "
        "(no Optuna) — that's intentional: ablation shows feature contribution "
        "in isolation. Optuna in cells would obscure whether sentiment helps "
        "or hyperparameters help. If sentiment Δ is positive on a target, that "
        "target benefits from sentiment beyond what defaults can squeeze out "
        "of ta_basic alone."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
