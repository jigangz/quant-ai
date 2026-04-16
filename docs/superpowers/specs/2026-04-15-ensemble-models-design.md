# Ensemble Models Design Spec

**Date**: 2026-04-15  
**Author**: Harry + Claude  
**Status**: Approved (pending implementation)  
**Phase**: 3 Sub-project 2 (after Optuna optimization)

## 1. Goal

Add model ensembling to Quant AI so users can combine predictions from multiple base models (logistic, random_forest, xgboost, lightgbm, catboost) to improve accuracy. Support both **voting** (fast, simple) and **stacking** (usually higher accuracy, slower) paradigms.

## 2. Design Principles

- **Zero new endpoints** — ensemble acts as a new `model_type` in the existing system
- **BaseModel conformance** — EnsembleModel inherits from BaseModel, works identically to other models in TrainingService, predict API, and ModelRegistry
- **User-selectable components** — frontend lets user pick ensemble mode, base models, meta-learner
- **YAGNI** — no auto-selection, no stacking with multiple meta-learner layers (keep it 2-layer max)

## 3. Architecture

```
┌──────────────────────────────────────────────────────────┐
│ Training.jsx                                              │
│   model_type dropdown: [..., "ensemble"]                  │
│   if ensemble: <EnsembleConfigForm/>                      │
│                                                           │
└──────────────┬───────────────────────────────────────────┘
               │ POST /api/train
               │ {model_type: "ensemble", ensemble_config: {...}}
               ▼
┌──────────────────────────────────────────────────────────┐
│ TrainingService.train()                                   │
│   ModelFactory.create("ensemble", ensemble_config=...)   │
│         │                                                 │
│         ▼                                                 │
│   EnsembleModel.fit(X, y)                                │
│     ├─ mode="voting_soft"  → fit_voting()                │
│     ├─ mode="voting_hard"  → fit_voting()                │
│     ├─ mode="stacking_*"   → fit_stacking()              │
│         (K-fold OOF predictions → meta_model)            │
│                                                           │
│   model.save(path) → pickle {base_models, meta_model,    │
│                              ensemble_config}            │
└──────────────────────────────────────────────────────────┘

Inference path (unchanged):
  POST /api/predict {model_id, ticker}
    → ModelRegistry.load(model_id) → EnsembleModel (from pickle)
    → model.predict_proba(X) dispatches by mode
```

## 4. EnsembleModel Class

**File**: `app/ml/models/ensemble_model.py`

### Pydantic config

```python
class EnsembleConfig(BaseModel):
    mode: Literal["voting_soft", "voting_hard",
                  "stacking_logistic", "stacking_xgboost"]
    base_models: list[str] = Field(min_length=2)  # ["logistic", "random_forest", ...]
    base_model_params: dict[str, dict] = Field(default_factory=dict)  # per-model hyperparams
    cv_folds: int = Field(default=5, ge=2, le=10)  # only used for stacking
    model_config = ConfigDict(extra="forbid")
```

### Class interface

```python
class EnsembleModel(BaseModel):
    model_type = "ensemble"

    def __init__(self, ensemble_config: EnsembleConfig | dict, **kwargs):
        # Accept either pydantic instance or dict (from ModelFactory.create kwargs)
        if isinstance(ensemble_config, dict):
            ensemble_config = EnsembleConfig(**ensemble_config)
        self.config = ensemble_config
        self.base_models: list[BaseModel] = []
        self.meta_model: BaseModel | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleModel": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...  # shape [N, 2]
    def save(self, path: Path) -> None: ...  # pickle whole instance
    @classmethod
    def load(cls, path: Path) -> "EnsembleModel": ...
```

### ModelFactory registration

Add to `app/ml/models/__init__.py` or `factory.py`:

```python
from app.ml.models.ensemble_model import EnsembleModel
ModelFactory.register("ensemble", EnsembleModel)
```

## 5. Training Flow

### 5.1 Voting (mode="voting_soft" or "voting_hard")

```python
def _fit_voting(self, X, y):
    for model_type in self.config.base_models:
        params = self.config.base_model_params.get(model_type, {})
        model = ModelFactory.create(model_type, **params)
        model.fit(X, y)
        self.base_models.append(model)
```

Trained sequentially (parallelism deferred as future optimization). No meta-learner.

### 5.2 Stacking (mode="stacking_logistic" or "stacking_xgboost")

Uses **K-fold out-of-fold (OOF)** predictions to prevent meta-learner from overfitting to base model predictions on training data:

```python
def _fit_stacking(self, X, y):
    n = len(X)
    n_base = len(self.config.base_models)
    oof_preds = np.zeros((n, n_base))   # OOF predictions for meta-learner training

    kf = KFold(n_splits=self.config.cv_folds, shuffle=False)  # no shuffle for time-series
    for train_idx, val_idx in kf.split(X):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]

        for i, model_type in enumerate(self.config.base_models):
            params = self.config.base_model_params.get(model_type, {})
            model = ModelFactory.create(model_type, **params)
            model.fit(X_tr, y_tr)
            oof_preds[val_idx, i] = model.predict_proba(X_val)[:, 1]

    # Train meta-learner on OOF predictions
    meta_type = "logistic" if self.config.mode == "stacking_logistic" else "xgboost"
    self.meta_model = ModelFactory.create(meta_type)
    self.meta_model.fit(pd.DataFrame(oof_preds), y)

    # Retrain base models on full data (used during inference)
    self.base_models = []
    for model_type in self.config.base_models:
        params = self.config.base_model_params.get(model_type, {})
        model = ModelFactory.create(model_type, **params)
        model.fit(X, y)
        self.base_models.append(model)
```

**Time-series note**: `KFold(shuffle=False)` preserves temporal ordering. Users that care about strict no-leak should use TimeSeriesSplit — deferred to a future improvement (YAGNI).

## 6. Inference Flow

```python
def predict_proba(self, X):
    base_probs = np.array([m.predict_proba(X)[:, 1] for m in self.base_models]).T  # [N, n_base]

    if self.config.mode == "voting_soft":
        pos = base_probs.mean(axis=1)
        return np.column_stack([1 - pos, pos])

    if self.config.mode == "voting_hard":
        base_preds = np.array([m.predict(X) for m in self.base_models]).T  # [N, n_base]
        pos = (base_preds.sum(axis=1) > len(self.base_models) / 2).astype(float)
        return np.column_stack([1 - pos, pos])

    # stacking
    return self.meta_model.predict_proba(pd.DataFrame(base_probs))

def predict(self, X):
    return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
```

## 7. Persistence

**Approach**: pickle the whole EnsembleModel instance (which contains fitted `base_models` and `meta_model`). This is simpler than saving each model separately and handles version compat via BaseModel's existing mechanism.

**Metadata** (stored alongside pickle via BaseModel.set_metadata):
- `ensemble_mode`: e.g. "stacking_logistic"
- `base_model_types`: ["logistic", "random_forest"]
- `meta_model_type`: "logistic" (null for voting)
- `cv_folds`: 5 (null for voting)

ModelRegistry requires no changes — EnsembleModel.save/load looks identical to other models.

## 8. API Changes

### TrainRequest (app/services/training_service.py)

```python
class TrainRequest(BaseModel):
    # existing fields...
    model_type: str = "logistic"  # no pattern constraint; ModelFactory validates
    model_params: dict[str, Any] = Field(default_factory=dict)
    # NEW:
    ensemble_config: Optional[dict] = None  # dict form of EnsembleConfig
```

When `model_type="ensemble"`, pass `ensemble_config` as kwargs to `ModelFactory.create("ensemble", ensemble_config=...)`. A Pydantic `model_validator(mode="after")` enforces:
- `ensemble_config` is required when `model_type == "ensemble"`
- `ensemble_config` must be `None` for all other model_types

### Predict API

**No changes** — existing `/api/predict` works because EnsembleModel conforms to BaseModel.

## 9. Frontend Changes

### Training.jsx

1. Add "ensemble" to model_type dropdown options
2. When selected, render `<EnsembleConfigForm/>`:
   - **Ensemble mode**: radio group (voting_soft, voting_hard, stacking_logistic, stacking_xgboost)
   - **Base models**: 5 checkboxes (logistic, random_forest, xgboost, lightgbm, catboost); default = [logistic, random_forest]; validate: at least 2 checked
   - **CV folds** (stacking only): number input, default 5, range [2, 10]

3. On submit, POST body becomes:
   ```json
   {
     "tickers": [...],
     "model_type": "ensemble",
     "ensemble_config": {
       "mode": "stacking_logistic",
       "base_models": ["logistic", "random_forest"],
       "cv_folds": 5
     }
   }
   ```

Styling follows existing Tailwind `bg-surface-card`, `text-accent` dark theme.

## 10. Testing

- `tests/test_ensemble_model.py` (8+ tests)
  - EnsembleConfig validation (pydantic)
  - voting_soft: fit + predict shape
  - voting_hard: fit + predict shape
  - stacking_logistic: fit + predict shape, base_models refit on full data
  - stacking_xgboost: same
  - save/load roundtrip
  - predict_proba output shape == [N, 2]
  - At least 2 base_models required (validation)
- `tests/test_ensemble_training.py` (3+ tests)
  - TrainingService end-to-end with ensemble_config → TrainResult.success
  - Metadata includes ensemble_mode, base_model_types
  - TrainRequest validation: ensemble_config required when model_type="ensemble"
- `tests/test_api_ensemble.py` (contract test, 2+ tests)
  - POST /api/train with ensemble_config → 200
  - POST /api/predict on ensemble model → returns prediction

## 11. Constraints

- **Base models list**: must be ≥2. (Single-model "ensemble" is just that model.)
- **XGBoost/LightGBM/CatBoost**: optional deps. If user selects them but they're not installed, ModelFactory already throws clear error — surface it to UI.
- **Training time**: stacking with K=5 and 3 base models means ~16 model fits. Can be slow for large datasets. Add timeout handling (reuse TrainingService's existing timeout if any).
- **Memory**: pickled ensemble size grows linearly with base model count. Typical 3-model stacking: ~20-50 MB on disk.

## 12. Out of Scope (Future Sub-projects)

- **Optuna optimization of ensemble** — use existing Optuna on each base model separately first, then ensemble with best params (user workflow, not automated)
- **TimeSeriesSplit** for stacking CV (use plain KFold for now)
- **Parallel base model training** (sequential is fine for first version)
- **Weighted voting** (voting_soft with learned weights is basically stacking — already covered)
- **Multi-layer stacking** (meta-learner of meta-learners — overkill)
- **Auto base-model selection** (AutoML style)

## 13. Success Criteria

- `pytest tests/` — all unit + contract tests pass, 15+ new tests for ensemble
- `npm run build` — clean
- `ruff check app/` — clean
- CI green on GitHub
- Manual: user can go to Training page → select Ensemble → train a stacking_logistic model → predict with it → see sharpe/accuracy in TrainResult
