# V5 Phase C — Cross-Sectional `xs_strong` wired into the real training pipeline

> Date: 2026-06-12 · Vault plan: `01-projects/quant-ai/v5-xs-task-plan.md` (Phase C, C1–C5)
> Supersedes the standalone PoC `scripts/xs_eval.py` (precision@30% 0.372, rank IC 0.048 on full S&P 500).
>
> **STATUS: ✅ COMPLETE (2026-06-12).** All C1–C5 steps shipped + 3 review-hardening fixes.
> 20 new tests in `tests/test_v5_phase_c_xs.py`; full suite 500 passed. Adversarial 3-lens
> review verdict: "correct, leak-safe, ship." Dry-run (80-ticker DB slice) rank IC 0.021.
> Run `python -m scripts.train_and_publish_xs` (drop `--dry-run`) to publish the full-universe
> model to the registry. Next: Phase D (serving — see "Out of scope" below).

## Goal (one line)

Make `xs_strong` (per-date top-30% forward-return = "strong group") a first-class
`label_type` that trains through `TrainingService.train`, is evaluated with
**cross-sectional metrics (Rank IC + precision@top_pct, no AUC)**, trains on the
**Phase-B-selected factor set**, and is **persisted to the registry+blob** so prod
can load it by id. No HTTP surface, no UI, no promotion changes — that is Phase D.

## Exit criterion

`scripts/train_and_publish_xs.py` trains an `xs_strong` XGBoost model on the
selected factors over a ticker subset, reports `test_rank_ic` ≈ PoC ballpark,
inserts a `model_registry` row with `label_type='xs_strong'`, saves the blob, and
`load_blob` + `from_zip_bytes` round-trips to a working `predict_proba`. Full test
suite green.

---

## Architecture decisions (grounded in source)

| # | Decision | Why |
|---|----------|-----|
| D1 | `xs_strong` is **NOT** a peer in `LABEL_GENERATORS`. The per-ticker step only computes `future_price`/`future_return` + a provisional `label`; the real 0/1 label is assigned **post-`concat`** in `builder.build()`. | `_add_labels` runs inside `_process_ticker` per single ticker (builder.py:181-217); a per-date quantile needs the full panel (`xs_eval.py:74` groups by date on the concatenated panel). |
| D2 | Per-date normalization hooks in `build()` right after `pd.concat` (builder.py:122) and before `_time_series_split` (builder.py:131), gated on `label_type=='xs_strong'`. Single-ticker path is byte-for-byte unchanged. | Only point where a real multi-ticker cross-section exists. `cross_section.py:47-49` zeroes single-name groups → must never run on the per-ticker path. |
| D3 | No "drop thin dates" in training — match the validated PoC exactly (it normalized the full panel pre-split, `xs_eval.py:95`). The `<5-name` skip lives only in the **eval metric**. | Reproduces the honest 0.048 IC; dropping dates would change the number. |
| D4 | Reuse `XGBoostModel(task='classification')` as-is (predict_proba). Create model with `task='classification'`; route **eval** with `task='xs_strong'`. No ranking loss / LambdaMART in Phase C. | `xgboost_model.py:73` classification branch has `predict_proba`; PoC used a plain `XGBClassifier`. LambdaRank is a Week-3 comparison experiment, not MVP. |
| D5 | Explicit feature selection: add `feature_names: list[str] \| None` to `DatasetConfig`/`TrainRequest`. When set, `_get_feature_columns` selects those columns; else group resolution (unchanged). | The Phase-B selected-12 includes `mom_12_1`/`dollar_volume`/`downside_vol_20`/`dist_52w_low` which `add_technical_features` computes (technical.py:92-107) but **no `feature_registry` group exposes**. Task-plan C4 requires training on B's selected set. |
| D6 | Registry: `label_type='xs_strong'`, `tickers=`full trained universe, `extras={'xs_strong': {top_pct, normalization:'z_score_per_date', min_cross_section}}`. Select by `model_id` / `list_models(label_type='xs_strong')`. **Do not touch global `is_promoted`.** | `ModelRecord.label_type` + free-form `extras` are built for this (model_registry.py:41,57). `list_models(label_type=...)` already filters (model_registry.py:343). Promotion is a single global boolean → promoting xs would un-promote the per-ticker model; defer label-scoped promotion to Phase D. |
| D7 | Metrics are flat aggregate scalars: `{split}_rank_ic`, `{split}_precision_at_top`, `{split}_rank_ic_ir`, `{split}_lift`, `{split}_base_rate`, `{split}_n_days`. No per-date keys, no AUC. | `ModelRecord.metrics` is `dict[str,float]` (model_registry.py:51); the convention is `{split}_{metric}` (training_service.py:367). |

---

## TDD steps

Each step: write the test first, run red, implement, run green. Tests live in
`tests/test_v5_phase_c_xs.py` (new) unless noted. Run with the repo `.venv`.

### C1a · LabelConfig schema (`app/ml/dataset/schemas.py`)
- Add `"xs_strong"` to the `label_type` `Literal` (schemas.py:31).
- Add `top_pct: float = Field(default=0.30, gt=0, lt=1, description="xs_strong per-date strong-group fraction")`.
- Add `feature_names: list[str] | None = None` to `DatasetConfig` (explicit feature override; `None` → group behavior).
- **Test**: `LabelConfig(label_type='xs_strong', top_pct=0.3)` validates; `top_pct=0`/`1.5` rejected; `extra='forbid'` still rejects junk; `DatasetConfig(..., feature_names=['rsi_14'])` validates.

### C1b · xs_strong label builder (`app/ml/labels/xs_strong.py`, new)
- `add_xs_forward_return(df, cfg)` — **single ticker**: sort by date; `future_price = close.shift(-horizon)`; `future_return = (future_price-close)/close`; provisional `label = 0.0` where `future_return` is non-NaN, `NaN` in the unlabelable tail (so the builder's `dropna(subset=['label'])` trims the tail; ticker-stats show a clean `{"0": N}` placeholder, refined post-concat).
- `add_xs_strong_label(panel, cfg)` — **multi-ticker** panel with `future_return` present: per date `cutoff = groupby('date')['future_return'].transform(quantile(1-top_pct))`; `label = (future_return >= cutoff).astype(int)`. Port of `xs_eval.py:65-77`.
- **Test** (3-ticker × 10-date synthetic panel): per-ticker builder adds `future_return` + provisional label with NaN tail of length `horizon`; cross-sectional builder labels ≈ `top_pct` fraction `1` per date, `future_return` preserved, ranking respected (highest-return name labeled 1).

### C1c · Task routing (`app/services/training_service.py`)
- `LABEL_TYPE_TO_TASK['xs_strong'] = 'xs_strong'` (training_service.py:46).
- In `train()` model creation: `model_task = 'classification' if task == 'xs_strong' else task`; create with `model_task`.
- **Test**: `_task_for_label_type('xs_strong') == 'xs_strong'`.

### C2 · Builder gate (`app/ml/dataset/builder.py`)
- `_add_labels`: if `label_type == 'xs_strong'` → `add_xs_forward_return(df, cfg)`; else `add_labels(...)` (unchanged).
- `_get_feature_columns`: if `config.feature_names` is set → `[c for c in feature_names if c in df.columns]`; else group resolution (unchanged).
- `build()` post-concat (after builder.py:123), gated on `xs_strong`: `combined_df = add_xs_strong_label(combined_df, cfg)`; `feature_cols = self._get_feature_columns(combined_df)`; `combined_df = cross_section_normalize(combined_df, feature_cols)`; `combined_df = combined_df.dropna(subset=feature_cols)`. For xs_strong, gate `ticker_stats.label_distribution` to `{}` (provisional per-ticker label is meaningless).
- **Test**: build a 5-ticker `xs_strong` dataset from a synthetic in-memory provider (monkeypatch `market_provider.fetch`); assert (a) direction-path output is unchanged (regression guard), (b) xs path: factors are per-date z-scored (mean≈0 per date), labels are 0/1, `future_return` survives in split frames.

### C3 · Carry date + future_return for eval (`schemas.py` + `builder.py`)
- Extend `DatasetOutput.__init__` with optional `groups_train/groups_val/groups_test = None` (DataFrames `[date, future_return]`). Plain class → non-breaking.
- In `build()`, for `xs_strong` populate them from the split frames; else leave `None`.
- **Test**: xs_strong output exposes aligned `groups_test` with `date`+`future_return` matching `X_test` row count; direction output leaves them `None`.

### C4a · Cross-sectional metrics (`app/backtest/metrics.py`)
- `calculate_xs_metrics(scores, future_return, dates, top_pct=0.30, min_names=5)` → per date (skip `<min_names`): `k=max(1,int(n*top_pct))`; `true_label = future_return >= per-date quantile(1-top_pct)`; `precision = true_label[top-k by score].mean()`; `ic = spearman(score, future_return)`. Return `{precision_at_top, rank_ic, rank_ic_ir, lift, base_rate, n_days}`. Port of `xs_eval.py:112-129`.
- **Test**: hand-built 2-date case with known ranking pins `precision_at_top` and `rank_ic`; a perfectly-ordered score gives `rank_ic≈1.0`; all-thin dates → `n_days=0` and `None`/`0` metrics (no crash).

### C4b · `_evaluate` branch (`app/services/training_service.py`)
- Add `groups: dict | None = None` and `top_pct: float = 0.30` params to `_evaluate`. New first branch: `if task == 'xs_strong'` → per split use `model.predict_proba(X)[:,1]` + `groups[split]` → `calculate_xs_metrics` → flat `{split}_{key}`; `continue`. Existing classification/regression untouched; `else: raise ValueError` stays.
- `train()` passes `groups={'train':..., 'val':..., 'test':...}` (from DatasetOutput) and `top_pct` for xs_strong.
- **Test**: a tiny trained xs model yields a metrics dict containing `test_rank_ic` + `test_precision_at_top` and **no** `test_auc`.

### C5a · End-to-end train (integration)
- `TrainingService.train(TrainRequest(label_type='xs_strong', model_type='xgboost', tickers=[subset], feature_names=selected, horizon_days=5, top_pct=0.30, search_mode='none'))` on a synthetic/small-DB subset.
- **Test** (marked slow / DB-optional): `result.success`, `result.metrics` has `test_rank_ic`, `model.save` wrote an artifact dir.

### C5b · Persist to registry + blob (`scripts/train_and_publish_xs.py`, new)
- Mirror `seed_demo_models.py`: train via `TrainingService` (reads `config/xs_selected_factors.json` → `feature_names`, universe via `app/data/universe.backfill_universe` or an explicit list) → insert `model_registry` row (`label_type='xs_strong'`, `tickers=universe`, `extras={'xs_strong': {...}}`, `metrics`) over `DATABASE_URL` → `save_blob(model_id, zip_model_dir(model_path))`.
- **Test**: round-trip — `BaseModel.from_zip_bytes(zip_model_dir(model_path))` loads and `predict_proba` works on a fresh normalized frame (no live DB needed; use the local artifact).

---

## Risks / guards
- **Leakage**: per-date normalization is leak-free across the time split *because* each date normalizes against itself (`cross_section.py:40`). Hook must run on the full concatenated frame, never per split.
- **No-pollution regression**: a direction-path build must be byte-identical pre/post Phase C — assert this explicitly in C2.
- **Phase D contract**: record `extras.xs_strong.normalization='z_score_per_date'` so Phase D's serving path re-normalizes against today's cross-section with the same function (no stored scaler).
- **Feature parity train↔serve**: model stores `feature_names` in metadata (`set_metadata`); Phase D must reindex to it (existing predict path already does, predict_service.py:131).

## Out of scope (Phase D)
`/predict/ranking` endpoint, `RankingService` + `get_prices_batch`, label-type-scoped promotion, frontend ranking page. README risk-disclaimers + screenshots = Phase E.
