"""
P3 Meta-Labeling Benchmark — V4 Phase 3.

Trains a meta-model on AAPL + MSFT + GOOGL using rsi_strategy as primary
and reports CV metrics. Run:

    python -m scripts.p3_meta_label_benchmark

Writes markdown to docs/benchmarks/p3_meta_label_benchmark.md.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main(tickers=("AAPL", "MSFT", "GOOGL")) -> None:
    rows = []
    for tkr in tickers:
        for primary_name in ("rsi_strategy",):  # start with 1; can extend
            print(f"[{tkr}] training meta-model w/ {primary_name} primary...")
            t0 = time.time()
            try:
                from app.services.meta_label_service import (
                    MetaLabelTrainRequest, train_meta_label_model,
                )
                from app.services.primary_signal_service import PrimarySignalSpec

                req = MetaLabelTrainRequest(
                    ticker=tkr,
                    primary=PrimarySignalSpec(
                        source="strategy", strategy_name=primary_name
                    ),
                    tp_k=2.0, sl_k=1.0, timeout_days=5,
                    vol_source="realized_sigma",
                    cv_n_splits=5, cv_embargo_pct=0.01,
                    model_type="xgboost", search_mode="default",
                    lookback_days=730, feature_group="ta_basic",
                )
                result = train_meta_label_model(req)
                elapsed = time.time() - t0
                rows.append({
                    "ticker": tkr, "primary": primary_name,
                    "event_count": result["event_count"],
                    "class_balance": result["class_balance"],
                    "cv_auc_mean": result["cv_metrics"]["auc_mean"],
                    "cv_auc_std": result["cv_metrics"]["auc_std"],
                    "precision_at_50": result["cv_metrics"]["precision_at_50"],
                    "expected_R_when_trade": result["cv_metrics"]["expected_R_when_trade"],
                    "hit_rate_when_trade": result["cv_metrics"]["hit_rate_when_trade"],
                    "folds_used": result["cv_metrics"]["folds_used"],
                    "train_time_s": round(elapsed, 2),
                    "warnings": result.get("warnings", []),
                })
            except Exception as e:
                rows.append({
                    "ticker": tkr, "primary": primary_name, "error": str(e),
                    "train_time_s": round(time.time() - t0, 2),
                })
                print(f"  FAILED: {e}")

    # Write markdown report
    out_path = Path("docs/benchmarks/p3_meta_label_benchmark.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = _render_markdown(rows)
    out_path.write_text(md, encoding="utf-8")
    print(f"\nReport: {out_path}")
    print(json.dumps(rows, indent=2, default=str))


def _render_markdown(rows: list[dict]) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# V4 Pivot · Phase 3 · Meta-Labeling Backend Benchmark",
        "",
        f"**Run date**: {now}",
        "**Primary**: rsi_strategy (rule-based, default params)",
        "**Barrier**: TP = 2 × σ, SL = 1 × σ, timeout = 5 days, vol_source = realized_sigma (20d rolling)",
        "**CV**: Purged K-Fold, n_splits=5, embargo=1% (López de Prado Ch.7)",
        "**Data window**: 730 days yfinance daily bars",
        "**Meta-model**: XGBoost classifier, default params",
        "",
        "## Per-ticker results",
        "",
        "| ticker | events | balance (✓/✗) | CV AUC (μ±σ) | precision@50% | E[R\\|trade] | hit-rate | folds | time |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(
                f"| {r['ticker']} | — | — | — | — | — | — | — | {r['train_time_s']}s ({r['error'][:40]}) |"
            )
            continue
        lines.append(
            f"| {r['ticker']} | {r['event_count']} | "
            f"{r['class_balance']['correct']}/{r['class_balance']['wrong']} | "
            f"{r['cv_auc_mean']:.3f} ± {r['cv_auc_std']:.3f} | "
            f"{r['precision_at_50']:.3f} | {r['expected_R_when_trade']:+.3f} | "
            f"{r['hit_rate_when_trade']:.3f} | {r['folds_used']} | {r['train_time_s']}s |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- **CV AUC ~0.55-0.65** is typical for meta-labeling on rule triggers — "
        "the meta-model has a narrow slice (one signal = one row), so sample size "
        "caps how sharp the classifier can be."
    )
    lines.append(
        "- **Precision-at-50% > 0.55** means the meta-model filters noise: when it "
        "says \"trade\", the primary was right >55% of the time. That's the whole "
        "point of López de Prado meta-labeling."
    )
    lines.append(
        "- **E[R | trade] > 0** (even +0.1) shows the meta-model's \"trade\" recommendations "
        "carry positive expected R. Combined with half-Kelly sizing in Paper Trading, "
        "this is a live signal quality system."
    )
    lines.append(
        "- **Small event counts** are a real constraint — rsi_strategy triggers maybe "
        "100-200 times on a 2-year window for a liquid stock. Longer windows + more "
        "strategies (multi-primary meta-ensemble) are natural v2 extensions."
    )
    lines.append("")
    lines.append(
        "## Honest framing for interview / portfolio\n\n"
        "This is Prado-rigorous backend infra: triple-barrier with dynamic vol-scaled "
        "barriers, Purged K-Fold with embargo, dual primary source (rules + ML direction). "
        "The numbers above will move as we tune primary strategies and add features. "
        "What matters is the methodology isn't fake: this is how Renaissance-style "
        "signal filtering actually works (Ch.3 in *Advances in Financial Machine Learning*)."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
