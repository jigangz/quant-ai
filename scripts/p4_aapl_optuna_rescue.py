"""
P4 AAPL Optuna Rescue — V4 Phase 4.

P3 benchmark showed AAPL × rsi_strategy meta-model AUC = 0.420 with default
XGBoost params. This script runs Optuna (n=30 trials) to see if hyperparameter
search can lift AUC above 0.5.

Run:
    python -m scripts.p4_aapl_optuna_rescue

Writes:
    - docs/benchmarks/p4_aapl_optuna.md (if AUC >= 0.5)
    - or a Quant/03_Rejected note (caller decides based on stdout)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from app.services.meta_label_service import (
    MetaLabelTrainRequest, train_meta_label_model,
)
from app.services.primary_signal_service import PrimarySignalSpec


def main():
    print("P4 AAPL × rsi_strategy Optuna rescue — 30 trials")
    t0 = time.time()
    req = MetaLabelTrainRequest(
        ticker="AAPL",
        primary=PrimarySignalSpec(source="strategy", strategy_name="rsi_strategy"),
        tp_k=2.0, sl_k=1.0, timeout_days=5,
        vol_source="realized_sigma",
        cv_n_splits=5, cv_embargo_pct=0.01,
        model_type="xgboost",
        search_mode="optuna",
        lookback_days=730, feature_group="ta_basic",
    )
    try:
        result = train_meta_label_model(req)
    except Exception as e:
        print(f"TRAINING FAILED: {e}")
        return
    elapsed = time.time() - t0
    auc = result["cv_metrics"]["auc_mean"]
    print(json.dumps(result, indent=2, default=str))
    print(f"\nElapsed: {elapsed:.1f}s · Optuna best AUC: {auc:.3f}")

    from pathlib import Path
    out = Path("docs/benchmarks/p4_aapl_optuna.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    status = "✅ rescued" if auc >= 0.5 else "❌ honest failure"
    md = f"""# P4 · AAPL × rsi_strategy Optuna Rescue

**Run date**: {now}
**Baseline (P3 default)**: AUC = 0.420
**Optuna (30 trials)**: AUC = {auc:.3f}
**Status**: {status}

## CV Metrics

- AUC mean ± std: {result['cv_metrics']['auc_mean']:.3f} ± {result['cv_metrics']['auc_std']:.3f}
- Precision @ 50%: {result['cv_metrics']['precision_at_50']:.3f}
- E[R | trade]: {result['cv_metrics']['expected_R_when_trade']:+.3f}
- Hit rate: {result['cv_metrics']['hit_rate_when_trade']:.3f}
- Folds used: {result['cv_metrics']['folds_used']}
- Event count: {result['event_count']}

## Interpretation

{_interpret(auc)}

## Raw response

```json
{json.dumps(result, indent=2, default=str)}
```
"""
    out.write_text(md, encoding="utf-8")
    print(f"Report: {out}")


def _interpret(auc: float) -> str:
    if auc >= 0.55:
        return (
            "Optuna successfully lifted AAPL's meta-label AUC above the useful-signal threshold. "
            "This suggests AAPL × rsi_strategy IS meta-labelable but requires tuned hyperparameters. "
            "Recommend updating P3 benchmark addendum with Optuna params."
        )
    if auc >= 0.5:
        return (
            "Optuna barely lifted AUC above 0.5 — marginal rescue. "
            "AAPL × rsi_strategy × ta_basic features are weakly labelable at best. "
            "Consider feature expansion (sentiment, regime) for real lift."
        )
    return (
        "Optuna could not lift AUC above 0.5 even with 30 trials. "
        "This is an **honest failure case**: AAPL × rsi_strategy × default feature set "
        "(ta_basic, 2y window) is not meta-labelable. "
        "Next investigations: (1) longer window (5y+), (2) sentiment feature group, "
        "(3) different primary strategy (momentum or bollinger_breakout), "
        "(4) different barrier config (asymmetric TP/SL). "
        "This is exactly the kind of methodological signal Prado Ch.3 promises: "
        "when the model says 'I can't learn this', you trust it rather than forcing it."
    )


if __name__ == "__main__":
    main()
