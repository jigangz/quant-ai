"""Tests for ablation metric extraction — honest absent-vs-zero handling (P6-1)."""

from app.services.ablation_service import _extract_metrics


class _Result:
    """Minimal stand-in for a TrainResult (has a .metrics attribute)."""

    def __init__(self, metrics):
        self.metrics = metrics


def test_direction_missing_auc_returns_none_not_zero():
    # No test_auc key → the metric is ABSENT, which must surface as None,
    # not a misleading 0.0 that looks like a real (terrible) score.
    out = _extract_metrics("direction", _Result({}))
    assert out["auc"] is None
    assert out["f1"] is None


def test_direction_real_auc_passthrough():
    out = _extract_metrics("direction", _Result({"test_auc": 0.62, "test_f1": 0.55}))
    assert out["auc"] == 0.62
    assert out["f1"] == 0.55


def test_direction_genuine_zero_is_kept():
    # A real 0.0 (model truly scored 0) must NOT be coerced to None.
    out = _extract_metrics("direction", _Result({"test_auc": 0.0}))
    assert out["auc"] == 0.0


def test_volatility_missing_metrics_all_none():
    out = _extract_metrics("volatility", _Result({}))
    assert out["qlike"] is None
    assert out["r2"] is None
    assert out["mae"] is None


def test_volatility_passthrough():
    out = _extract_metrics(
        "volatility", _Result({"test_qlike": 0.29, "test_r2": 0.11, "test_mae": 0.02})
    )
    assert out["qlike"] == 0.29
    assert out["r2"] == 0.11
    assert out["mae"] == 0.02


def test_meta_label_missing_cv_returns_none():
    out = _extract_metrics("meta_label", {})  # dict path, no cv_metrics
    assert out["auc_mean"] is None
    assert out["precision_at_50"] is None


def test_meta_label_passthrough():
    out = _extract_metrics(
        "meta_label", {"cv_metrics": {"auc_mean": 0.585, "precision_at_50": 0.6}}
    )
    assert out["auc_mean"] == 0.585
    assert out["precision_at_50"] == 0.6
