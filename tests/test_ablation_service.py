"""Tests for AblationService (V4 P5)."""
from __future__ import annotations

import pytest


@pytest.fixture
def fake_train(monkeypatch):
    """Patch training_service.train and meta_label_service.train_meta_label_model
    to return deterministic fake metrics so we can verify orchestration."""
    from app.services import training_service, meta_label_service

    def _fake_train(req):
        # Direction or volatility — return synthetic metrics
        has_sentiment = "sentiment" in req.feature_groups
        if req.label_type == "direction":
            return type("R", (), {
                "model_id": f"dir_{'sent' if has_sentiment else 'base'}",
                "metrics": {"test_auc": 0.6 if has_sentiment else 0.52,
                            "test_f1": 0.42 if has_sentiment else 0.34},
            })()
        elif req.label_type == "volatility":
            return type("R", (), {
                "model_id": f"vol_{'sent' if has_sentiment else 'base'}",
                "metrics": {"test_qlike": 0.142 if has_sentiment else 0.171,
                            "test_r2": 0.064 if has_sentiment else 0.019,
                            "test_mae": 0.072 if has_sentiment else 0.085},
            })()
        raise ValueError("unexpected target")

    def _fake_meta_train(req):
        groups = req.feature_group if isinstance(req.feature_group, list) else [req.feature_group]
        has_sentiment = "sentiment" in groups
        return {
            "success": True,
            "model_id": f"meta_{'sent' if has_sentiment else 'base'}",
            "cv_metrics": {
                "auc_mean": 0.641 if has_sentiment else 0.619,
                "precision_at_50": 0.61 if has_sentiment else 0.55,
            },
        }

    monkeypatch.setattr(training_service, "train", lambda self, req: _fake_train(req), raising=False)
    monkeypatch.setattr("app.services.ablation_service._train_target",
                         lambda req: _fake_train(req))
    monkeypatch.setattr(meta_label_service, "train_meta_label_model", _fake_meta_train)


def test_matrix_shape_3x2(fake_train):
    from app.services.ablation_service import run_ablation
    result = run_ablation(
        ticker="MSFT",
        targets=["direction", "volatility", "meta_label"],
        feature_sets=[
            {"name": "ta_basic", "groups": ["ta_basic"]},
            {"name": "ta_basic + sentiment", "groups": ["ta_basic", "sentiment"]},
        ],
        horizon_days=5, model_type="xgboost",
    )
    assert set(result["matrix"].keys()) == {"direction", "volatility", "meta_label"}
    for target in result["matrix"]:
        assert set(result["matrix"][target].keys()) == {"ta_basic", "ta_basic + sentiment"}


def test_sentiment_lift_detected(fake_train):
    from app.services.ablation_service import run_ablation
    result = run_ablation(
        ticker="MSFT",
        targets=["direction"],
        feature_sets=[
            {"name": "ta_basic", "groups": ["ta_basic"]},
            {"name": "ta_basic + sentiment", "groups": ["ta_basic", "sentiment"]},
        ],
        horizon_days=5, model_type="xgboost",
    )
    cell = result["matrix"]["direction"]["ta_basic + sentiment"]
    assert cell["delta_auc"] == pytest.approx(0.08, abs=0.01)


def test_unknown_feature_set_raises(fake_train, monkeypatch):
    from app.services.ablation_service import run_ablation
    monkeypatch.setattr("app.services.ablation_service._validate_feature_groups",
                         lambda groups: (_ for _ in ()).throw(ValueError("unknown_feature_set:mystery")))
    with pytest.raises(ValueError, match="unknown_feature_set"):
        run_ablation(
            ticker="MSFT",
            targets=["direction"],
            feature_sets=[{"name": "mystery", "groups": ["mystery"]}],
            horizon_days=5, model_type="xgboost",
        )


def test_meta_label_uses_extended_feature_group(fake_train):
    from app.services.ablation_service import run_ablation
    # The fake_meta_train asserts isinstance handling. If it sees a list it
    # detects sentiment. Verify we pass a list through:
    result = run_ablation(
        ticker="MSFT",
        targets=["meta_label"],
        feature_sets=[
            {"name": "ta_basic", "groups": ["ta_basic"]},
            {"name": "with_sentiment", "groups": ["ta_basic", "sentiment"]},
        ],
        horizon_days=5, model_type="xgboost",
    )
    assert result["matrix"]["meta_label"]["with_sentiment"]["auc_mean"] > \
           result["matrix"]["meta_label"]["ta_basic"]["auc_mean"]
