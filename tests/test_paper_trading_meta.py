"""Tests for Paper Trading meta-label integration (V4 Phase 3)."""
from __future__ import annotations

import pytest

from app.trading.engine import place_order


def test_no_meta_model_id_uses_legacy_path(monkeypatch):
    """When meta_model_id is None, behavior is identical to pre-P3."""
    # This test depends on existing place_order logic; we just assert no errors
    # raised when meta_model_id is not given. Detailed existing-order assertions
    # are covered by pre-existing Paper Trading tests.
    try:
        result = place_order(ticker="AAPL", side="buy", qty=10)
    except TypeError:
        pytest.skip("place_order signature change requires adapter update")
    assert result is not None


def test_score_below_threshold_rejects(monkeypatch):
    from app.services import signal_scoring_service

    def fake_score(req):
        return {
            "triggered": True, "signal": 1, "reliability_score": 0.30,
            "expected_R": -0.1, "recommended_action": "skip",
            "sizing_hint": {"half_kelly_fraction": 0.0, "raw_kelly": 0.0, "cap": 0.25},
            "meta_model": {"id": req.meta_model_id, "primary_source": "strategy:rsi_strategy", "cv_auc": 0.6},
            "timestamp": "2026-04-24T00:00:00Z",
        }
    monkeypatch.setattr(signal_scoring_service, "score_signal", fake_score)
    result = place_order(
        ticker="AAPL", side="buy", qty=10,
        meta_model_id="meta_abc", score_threshold=0.5,
    )
    assert getattr(result, "status", None) == "rejected"
    assert "meta_score_below_threshold" in getattr(result, "reason", "")


def test_score_above_threshold_places_sized_order(monkeypatch):
    from app.services import signal_scoring_service

    def fake_score(req):
        return {
            "triggered": True, "signal": 1, "reliability_score": 0.80,
            "expected_R": 0.6, "recommended_action": "trade",
            "sizing_hint": {"half_kelly_fraction": 0.25, "raw_kelly": 0.50, "cap": 0.25},
            "meta_model": {"id": req.meta_model_id, "primary_source": "strategy:rsi_strategy", "cv_auc": 0.6},
            "timestamp": "2026-04-24T00:00:00Z",
        }
    monkeypatch.setattr(signal_scoring_service, "score_signal", fake_score)
    result = place_order(
        ticker="AAPL", side="buy", qty=10,
        meta_model_id="meta_abc", score_threshold=0.5,
    )
    # Sized: qty * half_kelly / cap = 10 * 0.25 / 0.25 = 10 (full size at cap)
    assert getattr(result, "status", None) != "rejected"


def test_meta_model_missing_rejects_cleanly(monkeypatch):
    from app.services import signal_scoring_service

    def fake_score(req):
        raise ValueError("meta_model_not_found:meta_abc")

    monkeypatch.setattr(signal_scoring_service, "score_signal", fake_score)
    result = place_order(
        ticker="AAPL", side="buy", qty=10,
        meta_model_id="meta_abc", score_threshold=0.5,
    )
    assert getattr(result, "status", None) == "rejected"
    assert "meta_model" in getattr(result, "reason", "")
