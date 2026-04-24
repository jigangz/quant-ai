# P4 · AAPL × rsi_strategy Optuna Rescue

**Run date**: 2026-04-24T17:57:00+00:00
**Baseline (P3 default)**: AUC = 0.420
**Optuna (30 trials)**: AUC = 0.417
**Status**: ❌ honest failure

## CV Metrics

- AUC mean ± std: 0.417 ± 0.078
- Precision @ 50%: 0.420
- E[R | trade]: -0.047
- Hit rate: 0.420
- Folds used: 5
- Event count: 492

## Interpretation

Optuna could not lift AUC above 0.5 even with 30 trials. This is an **honest failure case**: AAPL × rsi_strategy × default feature set (ta_basic, 2y window) is not meta-labelable. Next investigations: (1) longer window (5y+), (2) sentiment feature group, (3) different primary strategy (momentum or bollinger_breakout), (4) different barrier config (asymmetric TP/SL). This is exactly the kind of methodological signal Prado Ch.3 promises: when the model says 'I can't learn this', you trust it rather than forcing it.

## Raw response

```json
{
  "success": true,
  "model_id": "meta_aapl_89e0e737",
  "event_count": 492,
  "class_balance": {
    "correct": 250,
    "wrong": 242
  },
  "cv_metrics": {
    "auc_mean": 0.4174792054379931,
    "auc_std": 0.07819596925359827,
    "precision_at_50": 0.4204008089722375,
    "expected_R_when_trade": -0.047090761327472065,
    "hit_rate_when_trade": 0.4204008089722375,
    "folds_used": 5
  },
  "barrier_config_used": {
    "tp_k": 2.0,
    "sl_k": 1.0,
    "timeout_days": 5,
    "vol_source": "realized_sigma"
  },
  "warnings": []
}
```
