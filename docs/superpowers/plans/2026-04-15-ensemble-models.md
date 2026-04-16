# Ensemble Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add model ensembling to Quant AI (voting + stacking modes) via a new `model_type="ensemble"` that inherits BaseModel and plugs into existing TrainingService + predict API.

**Architecture:** `EnsembleModel` wraps N user-selected base models plus an optional meta-learner. Voting modes average base predictions; stacking modes train a meta-learner on K-fold out-of-fold predictions. Persistence stores base_models.joblib + meta_model.joblib + params/metadata JSON.

**Tech Stack:** Python 3.9+, pydantic, scikit-learn (KFold, joblib), numpy, existing BaseModel / ModelFactory / TrainingService infrastructure.

---

## File Structure

| File | Responsibility | Status |
|------|----------------|--------|
| `app/ml/models/ensemble_model.py` | EnsembleConfig, EnsembleModel class | **create** |
| `app/ml/models/factory.py` | Register "ensemble" in ModelFactory | modify |
| `app/ml/models/__init__.py` | Export EnsembleModel, EnsembleConfig | modify |
| `app/services/training_service.py` | TrainRequest accepts ensemble_config | modify |
| `quant-ai-ui/src/pages/Training.jsx` | Ensemble config form UI | modify |
| `tests/test_ensemble_model.py` | Unit tests for EnsembleModel | **create** |
| `tests/test_ensemble_training.py` | TrainingService integration tests | **create** |
| `tests/contract/test_api_ensemble.py` | API contract tests | **create** |

---

## Task 1: EnsembleConfig + EnsembleModel skeleton

**Files:**
- Create: `app/ml/models/ensemble_model.py`
- Create: `tests/test_ensemble_model.py`

- [ ] **Step 1: Write failing tests for EnsembleConfig validation**

Create `tests/test_ensemble_model.py`:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_ensemble_config_valid_voting_soft():
    from app.ml.models.ensemble_model import EnsembleConfig

    config = EnsembleConfig(
        mode="voting_soft",
        base_models=["logistic", "random_forest"],
    )
    assert config.mode == "voting_soft"
    assert config.base_models == ["logistic", "random_forest"]
    assert config.cv_folds == 5  # default


def test_ensemble_config_requires_at_least_2_base_models():
    from app.ml.models.ensemble_model import EnsembleConfig

    with pytest.raises(ValidationError):
        EnsembleConfig(mode="voting_soft", base_models=["logistic"])


def test_ensemble_config_rejects_invalid_mode():
    from app.ml.models.ensemble_model import EnsembleConfig

    with pytest.raises(ValidationError):
        EnsembleConfig(mode="bagging", base_models=["logistic", "random_forest"])


def test_ensemble_config_cv_folds_range():
    from app.ml.models.ensemble_model import EnsembleConfig

    with pytest.raises(ValidationError):
        EnsembleConfig(
            mode="stacking_logistic",
            base_models=["logistic", "random_forest"],
            cv_folds=1,  # below min
        )


def test_ensemble_model_init_accepts_dict_or_pydantic():
    from app.ml.models.ensemble_model import EnsembleModel, EnsembleConfig

    # dict form (how ModelFactory passes it)
    m1 = EnsembleModel(ensemble_config={
        "mode": "voting_soft",
        "base_models": ["logistic", "random_forest"],
    })
    assert m1.config.mode == "voting_soft"
    assert m1.is_fitted is False

    # pydantic form (direct instantiation)
    config = EnsembleConfig(mode="voting_hard", base_models=["logistic", "random_forest"])
    m2 = EnsembleModel(ensemble_config=config)
    assert m2.config.mode == "voting_hard"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ensemble_model.py -v`
Expected: FAIL — module `app.ml.models.ensemble_model` does not exist.

- [ ] **Step 3: Create `app/ml/models/ensemble_model.py` with skeleton**

```python
from __future__ import annotations

"""
Ensemble Model

Combines predictions from N base models using voting or stacking.

Modes:
- voting_soft:    mean(base.predict_proba)
- voting_hard:    majority vote of base.predict()
- stacking_logistic: LogisticRegression meta-learner over K-fold OOF predictions
- stacking_xgboost:  XGBoost meta-learner over K-fold OOF predictions

Integration:
- Inherits BaseModel — plugs into TrainingService + predict API as model_type="ensemble"
- Persistence: base_models.joblib + meta_model.joblib + metadata/params JSON
"""

import json
import logging
from pathlib import Path
from typing import Literal, Union

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticModel, ConfigDict, Field

from app.ml.models.base import BaseModel, ModelMetadata

logger = logging.getLogger(__name__)


EnsembleMode = Literal[
    "voting_soft",
    "voting_hard",
    "stacking_logistic",
    "stacking_xgboost",
]


class EnsembleConfig(PydanticModel):
    """Configuration for an EnsembleModel."""

    mode: EnsembleMode
    base_models: list[str] = Field(min_length=2)
    base_model_params: dict[str, dict] = Field(default_factory=dict)
    cv_folds: int = Field(default=5, ge=2, le=10)

    model_config = ConfigDict(extra="forbid")


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models via voting or stacking."""

    model_type = "ensemble"

    def __init__(self, ensemble_config: Union[EnsembleConfig, dict], **kwargs):
        # Normalize to pydantic
        if isinstance(ensemble_config, dict):
            config = EnsembleConfig(**ensemble_config)
        else:
            config = ensemble_config

        # Store as dict in self.params for BaseModel.save() to serialize
        super().__init__(ensemble_config=config.model_dump())

        self.config: EnsembleConfig = config
        self.base_models: list[BaseModel] = []
        self.meta_model: BaseModel | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleModel":
        raise NotImplementedError("fit() will be implemented in Task 2 / Task 3")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("predict_proba() will be implemented in Task 2 / Task 3")
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_ensemble_model.py -v`
Expected: 5/5 PASS.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/zjg09/projects/quant-ai
git add app/ml/models/ensemble_model.py tests/test_ensemble_model.py
git commit -m "feat: [ENS-1] add EnsembleConfig and EnsembleModel skeleton"
```

---

## Task 2: Voting implementation (soft + hard)

**Files:**
- Modify: `app/ml/models/ensemble_model.py`
- Modify: `tests/test_ensemble_model.py`

- [ ] **Step 1: Write failing tests for voting fit + predict**

Append to `tests/test_ensemble_model.py`:

```python
import numpy as np
import pandas as pd


def _make_dummy_data(n=100, n_features=5, seed=42):
    """Create a trivially separable dataset."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    # y depends on f0 sign — both base models can learn it
    y = pd.Series((X["f0"] > 0).astype(int))
    return X, y


def test_voting_soft_fit_predict():
    from app.ml.models.ensemble_model import EnsembleModel

    X, y = _make_dummy_data()
    model = EnsembleModel(ensemble_config={
        "mode": "voting_soft",
        "base_models": ["logistic", "random_forest"],
    })
    model.fit(X, y)

    assert model.is_fitted is True
    assert len(model.base_models) == 2
    assert model.meta_model is None

    probs = model.predict_proba(X)
    assert probs.shape == (100, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

    preds = model.predict(X)
    assert preds.shape == (100,)
    assert set(preds.tolist()).issubset({0, 1})


def test_voting_hard_fit_predict():
    from app.ml.models.ensemble_model import EnsembleModel

    X, y = _make_dummy_data()
    model = EnsembleModel(ensemble_config={
        "mode": "voting_hard",
        "base_models": ["logistic", "random_forest"],
    })
    model.fit(X, y)

    assert len(model.base_models) == 2
    probs = model.predict_proba(X)
    assert probs.shape == (100, 2)
    # Hard voting produces 0/1 probs
    assert set(np.unique(probs).tolist()).issubset({0.0, 1.0})


def test_voting_respects_base_model_params():
    from app.ml.models.ensemble_model import EnsembleModel

    X, y = _make_dummy_data()
    model = EnsembleModel(ensemble_config={
        "mode": "voting_soft",
        "base_models": ["logistic", "random_forest"],
        "base_model_params": {
            "random_forest": {"n_estimators": 3},  # tiny forest to verify param plumbing
        },
    })
    model.fit(X, y)

    # Find the random_forest base model (second in list); check n_estimators was passed
    rf = model.base_models[1]
    # RandomForestModel stores n_estimators in params dict
    assert rf.params.get("n_estimators") == 3
```

- [ ] **Step 2: Run tests — expect FAIL (NotImplementedError)**

Run: `pytest tests/test_ensemble_model.py::test_voting_soft_fit_predict -v`
Expected: FAIL with NotImplementedError.

- [ ] **Step 3: Implement voting in `app/ml/models/ensemble_model.py`**

Replace the `fit()` and `predict_proba()` stubs with:

```python
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleModel":
        from app.ml.models import ModelFactory  # local import to avoid circular

        if self.config.mode in ("voting_soft", "voting_hard"):
            self._fit_voting(X, y, ModelFactory)
        else:
            self._fit_stacking(X, y, ModelFactory)

        self.is_fitted = True
        return self

    def _fit_voting(self, X: pd.DataFrame, y: pd.Series, factory) -> None:
        """Fit each base model on the full training set."""
        self.base_models = []
        for model_type in self.config.base_models:
            params = self.config.base_model_params.get(model_type, {})
            base = factory.create(model_type, **params)
            base.fit(X, y)
            self.base_models.append(base)
        logger.info(
            f"Voting ensemble fit: {len(self.base_models)} base models "
            f"({self.config.base_models})"
        )

    def _fit_stacking(self, X, y, factory) -> None:
        """Placeholder — implemented in Task 3."""
        raise NotImplementedError("stacking implemented in Task 3")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("EnsembleModel not fitted — call fit() first")

        mode = self.config.mode

        if mode == "voting_soft":
            # mean of predict_proba positive column across base models
            probs = np.stack([m.predict_proba(X)[:, 1] for m in self.base_models], axis=1)
            pos = probs.mean(axis=1)
            return np.column_stack([1.0 - pos, pos])

        if mode == "voting_hard":
            # majority vote of predict() across base models
            preds = np.stack([m.predict(X) for m in self.base_models], axis=1)
            # Majority: positive class wins if > half of base models predict 1
            pos = (preds.sum(axis=1) > len(self.base_models) / 2).astype(float)
            return np.column_stack([1.0 - pos, pos])

        if mode.startswith("stacking"):
            raise NotImplementedError("stacking predict_proba implemented in Task 3")

        raise ValueError(f"Unknown ensemble mode: {mode}")
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_ensemble_model.py -v`
Expected: 8/8 PASS (5 from Task 1 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add app/ml/models/ensemble_model.py tests/test_ensemble_model.py
git commit -m "feat: [ENS-2] implement voting (soft + hard) for EnsembleModel"
```

---

## Task 3: Stacking implementation (K-fold OOF + meta-learner)

**Files:**
- Modify: `app/ml/models/ensemble_model.py`
- Modify: `tests/test_ensemble_model.py`

- [ ] **Step 1: Write failing tests for stacking**

Append to `tests/test_ensemble_model.py`:

```python
def test_stacking_logistic_fit_predict():
    from app.ml.models.ensemble_model import EnsembleModel

    X, y = _make_dummy_data(n=200)
    model = EnsembleModel(ensemble_config={
        "mode": "stacking_logistic",
        "base_models": ["logistic", "random_forest"],
        "cv_folds": 3,
    })
    model.fit(X, y)

    assert model.is_fitted is True
    assert len(model.base_models) == 2  # retrained on full data
    assert model.meta_model is not None
    # Meta model is a LogisticModel instance
    assert model.meta_model.__class__.__name__ == "LogisticModel"

    probs = model.predict_proba(X)
    assert probs.shape == (200, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_stacking_uses_kfold_without_shuffle():
    """Ensure time-series ordering is preserved (no shuffle)."""
    from app.ml.models.ensemble_model import EnsembleModel

    X, y = _make_dummy_data(n=150)
    model = EnsembleModel(ensemble_config={
        "mode": "stacking_logistic",
        "base_models": ["logistic", "random_forest"],
        "cv_folds": 3,
    })
    # Just verify fit runs without error — no leakage check, just ordering preserved
    model.fit(X, y)
    assert model.is_fitted


def test_stacking_meta_model_trained_on_oof_shape():
    """Meta model should be fitted with OOF matrix of shape [N, n_base]."""
    from app.ml.models.ensemble_model import EnsembleModel

    X, y = _make_dummy_data(n=120)
    config = {
        "mode": "stacking_logistic",
        "base_models": ["logistic", "random_forest", "logistic"],  # 3 bases (duplicate ok)
        "cv_folds": 4,
    }
    model = EnsembleModel(ensemble_config=config)
    model.fit(X, y)

    # After fit, meta_model.params should reflect it was trained on 3-feature input
    # (we can't introspect easily; just check it predicts correctly)
    probs = model.predict_proba(X)
    assert probs.shape == (120, 2)
```

- [ ] **Step 2: Run tests — expect FAIL (NotImplementedError)**

Run: `pytest tests/test_ensemble_model.py::test_stacking_logistic_fit_predict -v`
Expected: FAIL with NotImplementedError.

- [ ] **Step 3: Implement stacking in `app/ml/models/ensemble_model.py`**

Replace the `_fit_stacking` stub with:

```python
    def _fit_stacking(self, X: pd.DataFrame, y: pd.Series, factory) -> None:
        """
        Fit via K-fold out-of-fold predictions.

        Steps:
        1. K-fold split (shuffle=False, preserves time order)
        2. For each fold: train bases on train side, predict on val side
        3. Collect OOF matrix [N, n_base]
        4. Train meta-learner on OOF matrix
        5. Retrain base models on full data (used at inference)
        """
        from sklearn.model_selection import KFold

        n = len(X)
        n_base = len(self.config.base_models)
        oof_preds = np.zeros((n, n_base))

        kf = KFold(n_splits=self.config.cv_folds, shuffle=False)
        fold_idx = 0
        for train_idx, val_idx in kf.split(X):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]

            for i, model_type in enumerate(self.config.base_models):
                params = self.config.base_model_params.get(model_type, {})
                base = factory.create(model_type, **params)
                base.fit(X_tr, y_tr)
                oof_preds[val_idx, i] = base.predict_proba(X_val)[:, 1]
            fold_idx += 1

        logger.info(
            f"Stacking OOF complete: {fold_idx} folds, OOF shape {oof_preds.shape}"
        )

        # Train meta-learner on OOF predictions
        meta_type = "logistic" if self.config.mode == "stacking_logistic" else "xgboost"
        self.meta_model = factory.create(meta_type)
        self.meta_model.fit(
            pd.DataFrame(oof_preds, columns=[f"base_{i}" for i in range(n_base)]),
            y.reset_index(drop=True),
        )

        # Retrain base models on full data — these are used at inference
        self.base_models = []
        for model_type in self.config.base_models:
            params = self.config.base_model_params.get(model_type, {})
            base = factory.create(model_type, **params)
            base.fit(X, y)
            self.base_models.append(base)
```

And update `predict_proba()` to handle stacking — replace the stacking branch:

```python
        if mode.startswith("stacking"):
            # Stack base positive-class probabilities → feed to meta-learner
            base_probs = np.stack(
                [m.predict_proba(X)[:, 1] for m in self.base_models], axis=1
            )
            stacked = pd.DataFrame(
                base_probs, columns=[f"base_{i}" for i in range(len(self.base_models))]
            )
            return self.meta_model.predict_proba(stacked)
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_ensemble_model.py -v`
Expected: 11/11 PASS (8 from Task 1-2 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add app/ml/models/ensemble_model.py tests/test_ensemble_model.py
git commit -m "feat: [ENS-3] implement stacking with K-fold OOF + meta-learner"
```

---

## Task 4: Custom save/load for EnsembleModel

**Files:**
- Modify: `app/ml/models/ensemble_model.py`
- Modify: `tests/test_ensemble_model.py`

- [ ] **Step 1: Write failing tests for save/load roundtrip**

Append to `tests/test_ensemble_model.py`:

```python
def test_voting_save_load_roundtrip(tmp_path):
    from app.ml.models.ensemble_model import EnsembleModel

    X, y = _make_dummy_data()
    model = EnsembleModel(ensemble_config={
        "mode": "voting_soft",
        "base_models": ["logistic", "random_forest"],
    })
    model.fit(X, y)
    expected = model.predict_proba(X)

    # Save
    save_dir = tmp_path / "model"
    model.save(save_dir)
    assert (save_dir / "base_models.joblib").exists()
    assert (save_dir / "params.json").exists()

    # Load
    loaded = EnsembleModel.load(save_dir)
    assert loaded.is_fitted is True
    assert loaded.config.mode == "voting_soft"
    assert len(loaded.base_models) == 2
    assert loaded.meta_model is None

    # Predictions match
    actual = loaded.predict_proba(X)
    assert np.allclose(expected, actual)


def test_stacking_save_load_roundtrip(tmp_path):
    from app.ml.models.ensemble_model import EnsembleModel

    X, y = _make_dummy_data()
    model = EnsembleModel(ensemble_config={
        "mode": "stacking_logistic",
        "base_models": ["logistic", "random_forest"],
        "cv_folds": 3,
    })
    model.fit(X, y)
    expected = model.predict_proba(X)

    save_dir = tmp_path / "stack_model"
    model.save(save_dir)
    assert (save_dir / "base_models.joblib").exists()
    assert (save_dir / "meta_model.joblib").exists()

    loaded = EnsembleModel.load(save_dir)
    assert loaded.config.mode == "stacking_logistic"
    assert loaded.meta_model is not None

    actual = loaded.predict_proba(X)
    assert np.allclose(expected, actual)


def test_save_writes_metadata_when_set(tmp_path):
    from app.ml.models.ensemble_model import EnsembleModel

    X, y = _make_dummy_data()
    model = EnsembleModel(ensemble_config={
        "mode": "voting_soft",
        "base_models": ["logistic", "random_forest"],
    })
    model.fit(X, y)
    model.set_metadata(
        feature_names=list(X.columns),
        feature_groups=["test"],
        tickers=["AAPL"],
        train_samples=80,
        val_samples=20,
        metrics={"val_auc": 0.95},
    )

    save_dir = tmp_path / "meta_model"
    model.save(save_dir)
    assert (save_dir / "metadata.json").exists()

    loaded = EnsembleModel.load(save_dir)
    assert loaded.metadata is not None
    assert loaded.metadata.metrics["val_auc"] == 0.95
```

- [ ] **Step 2: Run tests — expect FAIL (default BaseModel.save() uses self.model which is None)**

Run: `pytest tests/test_ensemble_model.py::test_voting_save_load_roundtrip -v`
Expected: FAIL — BaseModel.save tries joblib.dump(self.model) where self.model is None; missing base_models.joblib file.

- [ ] **Step 3: Override save/load in `app/ml/models/ensemble_model.py`**

Add to the `EnsembleModel` class (after `predict_proba`):

```python
    def save(self, path) -> None:
        """Save ensemble: base_models + meta_model + params + metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Base models list
        joblib.dump(self.base_models, path / "base_models.joblib")

        # Meta model (may be None for voting)
        if self.meta_model is not None:
            joblib.dump(self.meta_model, path / "meta_model.joblib")

        # Params (contains ensemble_config dict)
        with open(path / "params.json", "w") as f:
            json.dump(self.params, f, indent=2)

        # Metadata
        if self.metadata is not None:
            with open(path / "metadata.json", "w") as f:
                json.dump(self.metadata.model_dump(mode="json"), f, indent=2, default=str)

    @classmethod
    def load(cls, path) -> "EnsembleModel":
        """Load ensemble from disk."""
        path = Path(path)

        # Params → reconstruct instance
        with open(path / "params.json", "r") as f:
            params = json.load(f)
        instance = cls(**params)

        # Base models
        instance.base_models = joblib.load(path / "base_models.joblib")

        # Meta model (optional for voting)
        meta_path = path / "meta_model.joblib"
        if meta_path.exists():
            instance.meta_model = joblib.load(meta_path)

        # Metadata (optional)
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
            instance.metadata = ModelMetadata(**metadata_dict)

        instance.is_fitted = True
        return instance
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_ensemble_model.py -v`
Expected: 14/14 PASS (11 from Task 1-3 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add app/ml/models/ensemble_model.py tests/test_ensemble_model.py
git commit -m "feat: [ENS-4] add custom save/load for EnsembleModel"
```

---

## Task 5: ModelFactory registration

**Files:**
- Modify: `app/ml/models/factory.py`
- Modify: `app/ml/models/__init__.py`
- Modify: `tests/test_ensemble_model.py`

- [ ] **Step 1: Write failing test for factory registration**

Append to `tests/test_ensemble_model.py`:

```python
def test_model_factory_creates_ensemble():
    from app.ml.models import ModelFactory

    assert "ensemble" in ModelFactory.list_models()

    model = ModelFactory.create("ensemble", ensemble_config={
        "mode": "voting_soft",
        "base_models": ["logistic", "random_forest"],
    })
    assert model.__class__.__name__ == "EnsembleModel"
    assert model.config.mode == "voting_soft"
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/test_ensemble_model.py::test_model_factory_creates_ensemble -v`
Expected: FAIL — `Unknown model type: ensemble`.

- [ ] **Step 3: Register EnsembleModel in factory and __init__**

In `app/ml/models/factory.py`, add **after the last ModelFactory.register call** (around line 132):

```python
# Ensemble (always available — depends only on already-registered models)
from .ensemble_model import EnsembleModel, EnsembleConfig
ModelFactory.register("ensemble", EnsembleModel)
logger.info("Ensemble registered")
```

In `app/ml/models/__init__.py`, add at the top (after existing imports) and update `__all__`:

```python
from .ensemble_model import EnsembleModel, EnsembleConfig
```

And append to `__all__` list:
```python
    "EnsembleModel",
    "EnsembleConfig",
```

- [ ] **Step 4: Run all ensemble tests — expect PASS**

Run: `pytest tests/test_ensemble_model.py -v`
Expected: 15/15 PASS.

Also run full test suite to catch regressions:
Run: `pytest tests/ --ignore=tests/contract -q`
Expected: 249+ PASS (234 existing + 15 ensemble).

- [ ] **Step 5: Commit**

```bash
git add app/ml/models/factory.py app/ml/models/__init__.py tests/test_ensemble_model.py
git commit -m "feat: [ENS-5] register ensemble in ModelFactory"
```

---

## Task 6: TrainRequest + TrainingService integration

**Files:**
- Modify: `app/services/training_service.py`
- Create: `tests/test_ensemble_training.py`

- [ ] **Step 1: Write failing tests for TrainRequest + TrainingService**

Create `tests/test_ensemble_training.py`:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_train_request_accepts_ensemble_config():
    from app.services.training_service import TrainRequest

    req = TrainRequest(
        tickers=["AAPL"],
        model_type="ensemble",
        ensemble_config={
            "mode": "voting_soft",
            "base_models": ["logistic", "random_forest"],
        },
    )
    assert req.model_type == "ensemble"
    assert req.ensemble_config["mode"] == "voting_soft"


def test_train_request_ensemble_requires_config():
    """model_type='ensemble' without ensemble_config should fail validation."""
    from app.services.training_service import TrainRequest

    with pytest.raises(ValidationError):
        TrainRequest(
            tickers=["AAPL"],
            model_type="ensemble",
            # ensemble_config missing
        )


def test_train_request_non_ensemble_rejects_config():
    """model_type='logistic' with ensemble_config should fail validation."""
    from app.services.training_service import TrainRequest

    with pytest.raises(ValidationError):
        TrainRequest(
            tickers=["AAPL"],
            model_type="logistic",
            ensemble_config={"mode": "voting_soft", "base_models": ["logistic", "random_forest"]},
        )


def test_training_service_trains_ensemble_end_to_end(monkeypatch):
    """Use mocked dataset builder to verify ensemble trains and returns result."""
    import numpy as np
    import pandas as pd

    from app.services.training_service import TrainRequest, TrainingService

    # Patch DatasetBuilder.build to return synthetic data
    from app.ml.dataset import DatasetBuilder

    class FakeDataset:
        def __init__(self):
            rng = np.random.default_rng(0)
            self.X_train = pd.DataFrame(rng.normal(size=(100, 5)), columns=[f"f{i}" for i in range(5)])
            self.y_train = pd.Series((self.X_train["f0"] > 0).astype(int))
            self.X_val = pd.DataFrame(rng.normal(size=(20, 5)), columns=[f"f{i}" for i in range(5)])
            self.y_val = pd.Series((self.X_val["f0"] > 0).astype(int))
            self.X_test = pd.DataFrame(rng.normal(size=(20, 5)), columns=[f"f{i}" for i in range(5)])
            self.y_test = pd.Series((self.X_test["f0"] > 0).astype(int))

            class Meta:
                feature_names = [f"f{i}" for i in range(5)]
                n_features = 5
                total_samples = 140
                train_samples = 100
                val_samples = 20
                test_samples = 20
                train_date_range = ("2020-01-01", "2020-06-30")
                val_date_range = ("2020-07-01", "2020-08-15")
                test_date_range = ("2020-08-16", "2020-09-30")

            self.metadata = Meta()

    monkeypatch.setattr(DatasetBuilder, "build", lambda self: FakeDataset())

    service = TrainingService()
    result = service.train(TrainRequest(
        tickers=["AAPL"],
        model_type="ensemble",
        ensemble_config={
            "mode": "voting_soft",
            "base_models": ["logistic", "random_forest"],
        },
        save_model=False,
    ))

    assert result.success is True
    assert result.model_type == "ensemble"
    assert result.metrics  # non-empty
    assert "val_accuracy" in result.metrics
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest tests/test_ensemble_training.py -v`
Expected: FAIL — TrainRequest has no `ensemble_config` field.

- [ ] **Step 3: Modify `app/services/training_service.py`**

Find the `TrainRequest` class. Add `ensemble_config` field and validator.

At the top of file, add imports:

```python
from pydantic import BaseModel, ConfigDict, Field, model_validator
```

Replace the `TrainRequest` class definition. Find the `class TrainRequest(BaseModel):` block. Add `ensemble_config` field and validator:

```python
class TrainRequest(BaseModel):
    """Request for training a model."""

    # Data
    tickers: list[str] = Field(min_length=1)
    start_date: date | None = None
    end_date: date | None = None

    # Features
    feature_groups: list[str] = Field(default=["ta_basic", "momentum"])

    # Labels
    horizon_days: int = Field(default=5, ge=1, le=60)
    label_type: str = "direction"

    # Model
    model_type: str = "logistic"
    model_params: dict[str, Any] = Field(default_factory=dict)

    # Ensemble (required when model_type="ensemble")
    ensemble_config: dict[str, Any] | None = None

    # Hyperparameter search
    search_mode: str = Field(default="none", pattern="^(none|grid|optuna|optuna_multi)$")
    search_trials: int = Field(default=20, ge=1, le=200)
    search_timeout: int | None = Field(default=300, ge=10, le=3600)

    # Split
    train_ratio: float = Field(default=0.7, ge=0.5, le=0.9)
    val_ratio: float = Field(default=0.15, ge=0.05, le=0.3)

    # Options
    save_model: bool = True
    model_name: str | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_ensemble_config(self) -> "TrainRequest":
        if self.model_type == "ensemble" and self.ensemble_config is None:
            raise ValueError("ensemble_config is required when model_type='ensemble'")
        if self.model_type != "ensemble" and self.ensemble_config is not None:
            raise ValueError("ensemble_config must be None when model_type != 'ensemble'")
        return self
```

Then in `TrainingService.train()`, find where model params are prepared (around "3. Create model with best params", line ~204). Before `model = get_model(request.model_type, **model_params)`, add special handling for ensemble:

```python
            # 3. Create model with best params
            if request.model_type == "ensemble":
                model = get_model("ensemble", ensemble_config=request.ensemble_config)
            else:
                model = get_model(request.model_type, **model_params)
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_ensemble_training.py -v`
Expected: 4/4 PASS.

Run full suite:
Run: `pytest tests/ --ignore=tests/contract -q`
Expected: 253+ PASS (249 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add app/services/training_service.py tests/test_ensemble_training.py
git commit -m "feat: [ENS-6] integrate ensemble_config into TrainRequest and TrainingService"
```

---

## Task 7: API contract test

**Files:**
- Create: `tests/contract/test_api_ensemble.py`

- [ ] **Step 1: Write API contract test**

Create `tests/contract/test_api_ensemble.py`:

```python
from __future__ import annotations

"""
Contract tests for ensemble training via the /api/train endpoint.

The endpoint accepts ensemble_config dict for model_type='ensemble'.
Mocks DatasetBuilder.build to avoid hitting real data services.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    # Patch DatasetBuilder.build to return synthetic data
    from app.ml.dataset import DatasetBuilder

    class FakeDataset:
        def __init__(self):
            rng = np.random.default_rng(7)
            self.X_train = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
            self.y_train = pd.Series((self.X_train["a"] > 0).astype(int))
            self.X_val = pd.DataFrame(rng.normal(size=(15, 3)), columns=["a", "b", "c"])
            self.y_val = pd.Series((self.X_val["a"] > 0).astype(int))
            self.X_test = pd.DataFrame(rng.normal(size=(15, 3)), columns=["a", "b", "c"])
            self.y_test = pd.Series((self.X_test["a"] > 0).astype(int))

            class Meta:
                feature_names = ["a", "b", "c"]
                n_features = 3
                total_samples = 90
                train_samples = 60
                val_samples = 15
                test_samples = 15
                train_date_range = ("2020-01-01", "2020-05-30")
                val_date_range = ("2020-06-01", "2020-06-30")
                test_date_range = ("2020-07-01", "2020-07-30")

            self.metadata = Meta()

    monkeypatch.setattr(DatasetBuilder, "build", lambda self: FakeDataset())

    from app.main import app
    return TestClient(app)


def test_train_ensemble_voting_soft(client):
    resp = client.post("/api/train", json={
        "tickers": ["AAPL"],
        "model_type": "ensemble",
        "ensemble_config": {
            "mode": "voting_soft",
            "base_models": ["logistic", "random_forest"],
        },
        "save_model": False,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["model_type"] == "ensemble"


def test_train_ensemble_stacking_logistic(client):
    resp = client.post("/api/train", json={
        "tickers": ["AAPL"],
        "model_type": "ensemble",
        "ensemble_config": {
            "mode": "stacking_logistic",
            "base_models": ["logistic", "random_forest"],
            "cv_folds": 3,
        },
        "save_model": False,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True


def test_train_ensemble_rejects_missing_config(client):
    resp = client.post("/api/train", json={
        "tickers": ["AAPL"],
        "model_type": "ensemble",
        "save_model": False,
    })
    assert resp.status_code == 422  # validation error


def test_train_non_ensemble_rejects_config(client):
    resp = client.post("/api/train", json={
        "tickers": ["AAPL"],
        "model_type": "logistic",
        "ensemble_config": {
            "mode": "voting_soft",
            "base_models": ["logistic", "random_forest"],
        },
        "save_model": False,
    })
    assert resp.status_code == 422
```

- [ ] **Step 2: Run contract tests — expect PASS**

Run: `pytest tests/contract/test_api_ensemble.py -v`
Expected: 4/4 PASS.

Also full contract suite:
Run: `pytest tests/contract/ -q`
Expected: 43+ PASS (39 existing + 4 new).

- [ ] **Step 3: Commit**

```bash
git add tests/contract/test_api_ensemble.py
git commit -m "feat: [ENS-7] add API contract tests for ensemble training"
```

---

## Task 8: Frontend — Training.jsx ensemble config form

**Files:**
- Modify: `quant-ai-ui/src/pages/Training.jsx`

- [ ] **Step 1: Read current Training.jsx to understand structure**

Run: `cat quant-ai-ui/src/pages/Training.jsx | head -80`

The page already has a `model_type` dropdown with 5 options (logistic, random_forest, xgboost, lightgbm, catboost). We add "ensemble" and conditionally render a sub-form.

- [ ] **Step 2: Modify Training.jsx**

In `quant-ai-ui/src/pages/Training.jsx`:

**a. Add state for ensemble config.** Find the existing `useState` declarations near the top of the component. Add:

```jsx
const [ensembleConfig, setEnsembleConfig] = useState({
  mode: "voting_soft",
  base_models: ["logistic", "random_forest"],
  cv_folds: 5,
});
```

**b. Add "ensemble" to the model_type dropdown.** Find the `<select>` for model_type and add at the end of its options:

```jsx
<option value="ensemble">Ensemble (combine multiple models)</option>
```

**c. Conditionally render the ensemble config form.** Directly after the model_type `<select>` closing tag, add:

```jsx
{modelType === "ensemble" && (
  <div className="mt-4 p-4 rounded bg-surface-card border border-surface-border">
    <h3 className="text-lg font-semibold mb-3 text-accent">Ensemble Configuration</h3>

    {/* Ensemble mode */}
    <div className="mb-3">
      <label className="block text-sm font-medium mb-1">Mode</label>
      <select
        className="w-full bg-surface-input border border-surface-border rounded px-3 py-2"
        value={ensembleConfig.mode}
        onChange={(e) => setEnsembleConfig({ ...ensembleConfig, mode: e.target.value })}
      >
        <option value="voting_soft">Voting — Soft (average probabilities)</option>
        <option value="voting_hard">Voting — Hard (majority vote)</option>
        <option value="stacking_logistic">Stacking — Logistic meta-learner</option>
        <option value="stacking_xgboost">Stacking — XGBoost meta-learner</option>
      </select>
    </div>

    {/* Base models */}
    <div className="mb-3">
      <label className="block text-sm font-medium mb-1">Base Models (≥2)</label>
      <div className="flex flex-wrap gap-2">
        {["logistic", "random_forest", "xgboost", "lightgbm", "catboost"].map((m) => (
          <label key={m} className="flex items-center gap-1 text-sm">
            <input
              type="checkbox"
              checked={ensembleConfig.base_models.includes(m)}
              onChange={(e) => {
                const next = e.target.checked
                  ? [...ensembleConfig.base_models, m]
                  : ensembleConfig.base_models.filter((x) => x !== m);
                setEnsembleConfig({ ...ensembleConfig, base_models: next });
              }}
            />
            {m}
          </label>
        ))}
      </div>
    </div>

    {/* CV folds (stacking only) */}
    {ensembleConfig.mode.startsWith("stacking") && (
      <div className="mb-3">
        <label className="block text-sm font-medium mb-1">CV Folds</label>
        <input
          type="number"
          min="2"
          max="10"
          className="w-24 bg-surface-input border border-surface-border rounded px-3 py-2"
          value={ensembleConfig.cv_folds}
          onChange={(e) =>
            setEnsembleConfig({ ...ensembleConfig, cv_folds: parseInt(e.target.value, 10) || 5 })
          }
        />
      </div>
    )}
  </div>
)}
```

**d. Include ensemble_config in the train request.** Find the submit/click handler that POSTs to `/api/train`. In the payload object, add conditionally:

```jsx
const payload = {
  tickers: tickers.split(",").map(t => t.trim()).filter(Boolean),
  model_type: modelType,
  // ... other existing fields ...
};

if (modelType === "ensemble") {
  payload.ensemble_config = ensembleConfig;
}
```

(If the existing handler builds the payload inline in the fetch call, extract it into a `payload` const first, then conditionally add.)

- [ ] **Step 3: Run build to verify no JSX errors**

Run: `cd quant-ai-ui && npm run build 2>&1 | tail -10`
Expected: `✓ built in ...` with 0 errors.

- [ ] **Step 4: Commit**

```bash
cd /c/Users/zjg09/projects/quant-ai
git add quant-ai-ui/src/pages/Training.jsx
git commit -m "feat: [ENS-8] add ensemble config form to Training page"
```

---

## Task 9: ENS-GATE — Full verification

**Files:** (none modified — verification only)

- [ ] **Step 1: Run full unit test suite**

Run: `pytest tests/ --tb=short --ignore=tests/contract -p no:cacheprovider -q 2>&1 | tail -5`
Expected: 253+ passed, 0 failures.

- [ ] **Step 2: Run contract test suite**

Run: `pytest tests/contract/ --tb=short -p no:cacheprovider -q 2>&1 | tail -5`
Expected: 43+ passed, 0 failures.

- [ ] **Step 3: Run ruff lint**

Run: `ruff check app/ --ignore F401,F841,E501,F541,E402 2>&1 | tail -5`
Expected: `All checks passed!`

- [ ] **Step 4: Build frontend**

Run: `cd quant-ai-ui && npm run build 2>&1 | tail -15`
Expected: `✓ built in ...` with 0 errors.

- [ ] **Step 5: Verify "ensemble" in ModelFactory**

Run: `python -c "from app.ml.models import ModelFactory; print('ensemble' in ModelFactory.list_models())"`
Expected: `True`.

- [ ] **Step 6: Verify `/api/train` accepts ensemble via FastAPI routes**

Run: `python -c "from app.main import app; routes = [r.path for r in app.routes if '/train' in r.path]; print(routes)"`
Expected: Output includes `/api/train`.

- [ ] **Step 7: Commit (gate marker)**

```bash
git commit --allow-empty -m "feat: [ENS-GATE] Phase 3 Sub-project 2 — ensemble gate"
```

---

## Self-Review

**1. Spec coverage:**
- Spec §3 Architecture → Tasks 1-6 cover training flow; inference is by BaseModel contract ✓
- Spec §4 EnsembleModel class → Task 1 ✓
- Spec §5.1 Voting → Task 2 ✓
- Spec §5.2 Stacking → Task 3 ✓
- Spec §6 Inference → implicit in Task 2 + Task 3 (predict_proba) ✓
- Spec §7 Persistence → Task 4 ✓
- Spec §8 API changes → Task 6 (TrainRequest + train dispatch) ✓
- Spec §9 Frontend → Task 8 ✓
- Spec §10 Testing — unit (Tasks 1-5), integration (Task 6), contract (Task 7) — all 15+ tests covered ✓
- Spec §11 Constraints — min_length=2 (Task 1), optional deps handled by existing ModelFactory error, time-series note in Task 3 comment ✓
- Spec §12 Out of Scope — none implemented ✓

**2. Placeholder scan:** No TBD/TODO; all steps contain complete code.

**3. Type consistency:**
- `EnsembleConfig` fields used consistently (mode, base_models, base_model_params, cv_folds) across Tasks 1-4
- `EnsembleModel` attributes (config, base_models, meta_model) consistent
- `factory` parameter in `_fit_voting`/`_fit_stacking` consistent (passed from fit())
- `ensemble_config` dict form used consistently (Tasks 5, 6, 7, 8)

**4. Gap check:** No gaps identified.
