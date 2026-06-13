"""V5 Phase D — serving the xs_strong ranking (/predict/ranking + RankingService)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")


# --- synthetic OHLCV (self-contained; mirrors the Phase C fixture) ---

class _FakeProvider:
    def __init__(self, n: int = 320):
        self.n = n

    def fetch(self, ticker, start_date=None, end_date=None, **kwargs):
        seed = abs(hash(ticker)) % (2**32)
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2024-01-01", periods=self.n)
        drift = 0.0003 * (seed % 7 - 3)
        rets = rng.normal(drift, 0.015, self.n)
        close = 100.0 * np.exp(np.cumsum(rets))
        return pd.DataFrame(
            {
                "ticker": ticker,
                "date": dates,
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": rng.integers(1_000_000, 5_000_000, self.n).astype(float),
            }
        )


_XS_FEATURES = ["rsi_14", "returns_5d", "bb_position", "volume_ratio"]
_TICKERS = ["AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH"]


def _train_tiny_xs(tmp_path, monkeypatch, feature_names=None):
    monkeypatch.setattr(
        "app.ml.dataset.builder.get_market_provider",
        lambda *a, **k: _FakeProvider(),
    )
    from app.services.training_service import TrainingService, TrainRequest

    svc = TrainingService(artifacts_path=str(tmp_path))
    res = svc.train(
        TrainRequest(
            tickers=_TICKERS,
            feature_names=feature_names or _XS_FEATURES,
            feature_groups=["momentum", "volatility"],
            label_type="xs_strong",
            model_type="xgboost",
            horizon_days=5,
            top_pct=0.30,
            search_mode="none",
            save_model=True,
        )
    )
    assert res.success, res.error
    from app.ml.models.xgboost_model import XGBoostModel

    model = XGBoostModel.load(res.model_path)
    return model


def _prices_dict():
    prov = _FakeProvider()
    return {t: prov.fetch(t) for t in _TICKERS}


# ===================================================================
# D1 — batch price fetch
# ===================================================================

class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        return self

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _FakeEngine:
    def __init__(self, rows):
        self._rows = rows

    def begin(self):
        return _FakeConn(self._rows)


def test_get_prices_batch_groups_and_sorts(monkeypatch):
    from app.db import prices_repo

    rows = [
        {"ticker": "AA", "date": "2024-01-03", "open": 1, "high": 1, "low": 1, "close": 3, "volume": 9},
        {"ticker": "AA", "date": "2024-01-01", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 7},
        {"ticker": "BB", "date": "2024-01-02", "open": 1, "high": 1, "low": 1, "close": 2, "volume": 8},
    ]
    monkeypatch.setattr(prices_repo, "engine", _FakeEngine(rows))
    out = prices_repo.get_prices_batch(["AA", "BB"])
    assert set(out.keys()) == {"AA", "BB"}
    assert len(out["AA"]) == 2
    # sorted ascending by date within a ticker
    assert list(out["AA"]["close"]) == [1, 3]


def test_get_prices_batch_empty_inputs():
    from app.db.prices_repo import get_prices_batch

    assert get_prices_batch([]) == {}


# ===================================================================
# D2 — RankingService
# ===================================================================

def test_default_xs_model_id_picks_latest(monkeypatch):
    from app.services import ranking_service

    class _Rec:
        id = "xs_model_42"

    monkeypatch.setattr(
        ranking_service, "_list_models", lambda **k: [_Rec()] if k.get("label_type") == "xs_strong" else []
    )
    assert ranking_service.default_xs_model_id() == "xs_model_42"


def test_rank_sorts_scores_and_returns_top_n(tmp_path, monkeypatch):
    model = _train_tiny_xs(tmp_path, monkeypatch)
    from app.services.ranking_service import RankingService

    out = RankingService().rank(
        top_n=5, universe=_TICKERS, prices=_prices_dict(), model=model
    )
    assert out["success"] is True
    rk = out["rankings"]
    assert len(rk) == 5
    scores = [r["score"] for r in rk]
    assert scores == sorted(scores, reverse=True)  # descending
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert [r["rank"] for r in rk] == [1, 2, 3, 4, 5]
    assert out["scored"] == len(_TICKERS)
    assert "score_semantics" in out


def test_rank_no_model_returns_graceful_failure(monkeypatch):
    from app.services.ranking_service import RankingService

    # no model passed, no default available
    monkeypatch.setattr(
        "app.services.ranking_service.default_xs_model_id", lambda: None
    )
    out = RankingService().rank(top_n=5, universe=_TICKERS, prices=_prices_dict())
    assert out["success"] is False
    assert "model" in out["error"].lower()


def test_rank_skips_thin_history(tmp_path, monkeypatch):
    model = _train_tiny_xs(tmp_path, monkeypatch)
    from app.services.ranking_service import RankingService

    prices = _prices_dict()
    prices["AA"] = prices["AA"].head(10)  # too short to build features
    out = RankingService().rank(
        top_n=10, universe=_TICKERS, prices=prices, model=model
    )
    tickers = {r["ticker"] for r in out["rankings"]}
    assert "AA" not in tickers
    assert out["scored"] == len(_TICKERS) - 1


def test_rank_drops_ticker_with_nan_long_window_factor(tmp_path, monkeypatch):
    """A name that clears MIN_HISTORY but is NaN in a long-window factor
    (mom_12_1 needs 252 rows) must be excluded, not median-imputed."""
    feats = _XS_FEATURES + ["mom_12_1"]
    model = _train_tiny_xs(tmp_path, monkeypatch, feature_names=feats)
    from app.services.ranking_service import RankingService

    prices = _prices_dict()
    prices["AA"] = prices["AA"].head(120)  # >60 rows but <252 → mom_12_1 NaN
    out = RankingService().rank(top_n=10, universe=_TICKERS, prices=prices, model=model)
    assert "AA" not in {r["ticker"] for r in out["rankings"]}
    assert out["scored"] == len(_TICKERS) - 1


# ===================================================================
# D3 — endpoint wiring
# ===================================================================

def test_ranking_endpoint_shape(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    canned = {
        "success": True,
        "model_id": "m1",
        "as_of": "2026-06-13",
        "universe_size": 3,
        "scored": 3,
        "top_n": 2,
        "rankings": [
            {"rank": 1, "ticker": "NVDA", "score": 0.82, "percentile": 99.0},
            {"rank": 2, "ticker": "AVGO", "score": 0.70, "percentile": 66.0},
        ],
        "score_semantics": "strength score (0-1), not a price target",
    }
    monkeypatch.setattr(
        "app.api.predict.RankingService",
        lambda: type("S", (), {"rank": lambda self, **k: canned})(),
    )
    r = TestClient(app).get("/predict/ranking?top_n=2")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert len(body["rankings"]) == 2
    assert body["rankings"][0]["ticker"] == "NVDA"
