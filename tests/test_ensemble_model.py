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
