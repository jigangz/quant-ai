"""V5 Phase C — xs_strong cross-sectional label wired into the training pipeline.

Covers C1 (schema + label builder + task routing), C2/C3 (builder gate + group
carry), C4 (cross-sectional metrics + _evaluate branch). C5 (end-to-end train +
registry round-trip) lives in the marked `slow` tests at the bottom.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ===================================================================
# C1a — LabelConfig / DatasetConfig schema
# ===================================================================

def test_label_config_accepts_xs_strong_and_top_pct():
    from app.ml.dataset.schemas import LabelConfig

    cfg = LabelConfig(label_type="xs_strong", top_pct=0.3, horizon_days=5)
    assert cfg.label_type == "xs_strong"
    assert cfg.top_pct == 0.3


def test_label_config_rejects_bad_top_pct_and_junk():
    from pydantic import ValidationError
    from app.ml.dataset.schemas import LabelConfig

    with pytest.raises(ValidationError):
        LabelConfig(label_type="xs_strong", top_pct=0.0)
    with pytest.raises(ValidationError):
        LabelConfig(label_type="xs_strong", top_pct=1.5)
    with pytest.raises(ValidationError):
        LabelConfig(label_type="xs_strong", nonsense=1)  # extra='forbid'


def test_dataset_config_accepts_feature_names_override():
    from app.ml.dataset.schemas import DatasetConfig

    cfg = DatasetConfig(tickers=["AAPL"], feature_names=["rsi_14", "mom_12_1"])
    assert cfg.feature_names == ["rsi_14", "mom_12_1"]
    # default stays None (group behavior)
    assert DatasetConfig(tickers=["AAPL"]).feature_names is None


# ===================================================================
# C1b — xs_strong label builder
# ===================================================================

def _single_ticker(close, start="2025-01-01"):
    dates = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame({"date": dates, "close": close})


def test_forward_return_has_horizon_nan_tail():
    from app.ml.dataset.schemas import LabelConfig
    from app.ml.labels.xs_strong import add_xs_forward_return

    df = _single_ticker([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    out = add_xs_forward_return(df, LabelConfig(label_type="xs_strong", horizon_days=2))

    # future_return present; last `horizon` rows are NaN (no future price)
    assert "future_return" in out.columns
    assert out["future_return"].iloc[:-2].notna().all()
    assert out["future_return"].iloc[-2:].isna().all()
    # provisional label: 0.0 on valid rows, NaN on the tail (so builder dropna trims)
    assert (out["label"].iloc[:-2] == 0.0).all()
    assert out["label"].iloc[-2:].isna().all()
    # value matches definition: close[t+h]/close[t]-1
    assert out["future_return"].iloc[0] == pytest.approx(12.0 / 10.0 - 1.0)


def _multi_panel():
    """3 tickers x 6 dates, future_return precomputed, distinct per-date order."""
    rows = []
    dates = pd.bdate_range("2025-01-01", periods=6)
    # per date, ticker C strongest, B middle, A weakest
    fwd = {"A": 0.01, "B": 0.02, "C": 0.05}
    for d in dates:
        for t, f in fwd.items():
            rows.append({"date": d, "ticker": t, "close": 100.0, "future_return": f})
    return pd.DataFrame(rows)


def test_xs_strong_label_picks_per_date_top_fraction():
    from app.ml.dataset.schemas import LabelConfig
    from app.ml.labels.xs_strong import add_xs_strong_label

    panel = _multi_panel()
    out = add_xs_strong_label(panel, LabelConfig(label_type="xs_strong", top_pct=0.34))

    # top ~1/3 of 3 names = the single strongest (C) each date
    for _, g in out.groupby("date"):
        strong = set(g[g["label"] == 1]["ticker"])
        assert strong == {"C"}, strong
    assert out["future_return"].notna().all()  # preserved for eval
    assert set(out["label"].unique()) <= {0, 1}


# ===================================================================
# C1c — task routing
# ===================================================================

def test_task_for_xs_strong():
    from app.services.training_service import _task_for_label_type

    assert _task_for_label_type("xs_strong") == "xs_strong"
    assert _task_for_label_type("direction") == "classification"


# ===================================================================
# C2 / C3 — builder gate (post-concat label + per-date normalize) + group carry
# ===================================================================

class _FakeProvider:
    """Deterministic synthetic OHLCV per ticker (no network/DB)."""

    def __init__(self, n: int = 220):
        self.n = n

    def fetch(self, ticker, start_date=None, end_date=None, **kwargs):
        seed = abs(hash(ticker)) % (2**32)
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2024-01-01", periods=self.n)
        drift = 0.0003 * (seed % 7 - 3)  # different trend per ticker
        rets = rng.normal(drift, 0.015, self.n)
        close = 100.0 * np.exp(np.cumsum(rets))
        high = close * (1 + np.abs(rng.normal(0, 0.005, self.n)))
        low = close * (1 - np.abs(rng.normal(0, 0.005, self.n)))
        vol = rng.integers(1_000_000, 5_000_000, self.n).astype(float)
        return pd.DataFrame(
            {
                "ticker": ticker,
                "date": dates,
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
            }
        )


_XS_FEATURES = ["rsi_14", "returns_5d", "bb_position", "volume_ratio"]


def _build(label_type, feature_names=None, feature_groups=None, tickers=None):
    from app.ml.dataset.builder import DatasetBuilder
    from app.ml.dataset.schemas import DatasetConfig, LabelConfig

    cfg = DatasetConfig(
        tickers=tickers or ["AA", "BB", "CC", "DD", "EE"],
        feature_groups=feature_groups or ["momentum", "volatility"],
        feature_names=feature_names,
        label_config=LabelConfig(label_type=label_type, horizon_days=5, top_pct=0.30),
        min_samples_per_ticker=50,
    )
    builder = DatasetBuilder(cfg)
    builder.market_provider = _FakeProvider()
    return builder.build()


def test_xs_build_normalizes_per_date_and_labels_binary():
    out = _build("xs_strong", feature_names=_XS_FEATURES)

    # labels are 0/1
    assert set(pd.unique(out.y_test)) <= {0, 1}
    # each date's cross-section is z-scored: per-date mean ~ 0 for every factor
    test = out.X_test.copy()
    test["date"] = out.groups_test["date"].values
    for col in _XS_FEATURES:
        per_date_mean = test.groupby("date")[col].mean().abs()
        assert (per_date_mean < 1e-6).all(), f"{col} not per-date centered"
    # normalization produced real spread (not all-zero)
    assert float(out.X_test[_XS_FEATURES].to_numpy().std()) > 0.1


def test_xs_build_carries_date_and_future_return():
    out = _build("xs_strong", feature_names=_XS_FEATURES)
    assert out.groups_test is not None
    assert list(out.groups_test.columns) == ["date", "future_return"]
    assert len(out.groups_test) == len(out.X_test)
    assert out.groups_test["future_return"].notna().all()


def test_direction_build_unchanged_no_group_carry():
    """Regression guard: the single-ticker path must be untouched."""
    out = _build("direction", feature_groups=["momentum", "volatility"])
    # group resolution still drives columns (no feature_names override)
    from app.ml.features.registry import feature_registry

    expected = feature_registry.get_feature_names(["momentum", "volatility"])
    assert list(out.X_train.columns) == [c for c in expected if c in out.X_train.columns]
    # no cross-sectional carry for non-xs labels
    assert out.groups_train is None
    assert out.groups_test is None


# ===================================================================
# C4a — cross-sectional metrics
# ===================================================================

def test_xs_metrics_perfect_ranking():
    from app.backtest.metrics import calculate_xs_metrics

    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    fwd = np.array([0.05, 0.04, 0.03, 0.02, 0.01])  # same order as scores
    dates = np.array(["2025-01-02"] * 5)
    m = calculate_xs_metrics(scores, fwd, dates, top_pct=0.4)

    assert m["n_days"] == 1
    assert m["rank_ic"] == pytest.approx(1.0, abs=1e-6)
    assert m["precision_at_top"] == pytest.approx(1.0, abs=1e-6)
    assert m["base_rate"] == pytest.approx(0.4, abs=1e-6)
    assert m["lift"] == pytest.approx(2.5, abs=1e-6)


def test_xs_metrics_skips_thin_dates():
    from app.backtest.metrics import calculate_xs_metrics

    # only 3 names on the single date -> below min_names=5 -> skipped
    scores = np.array([0.9, 0.5, 0.1])
    fwd = np.array([0.05, 0.02, -0.01])
    dates = np.array(["2025-01-02"] * 3)
    m = calculate_xs_metrics(scores, fwd, dates, top_pct=0.3)
    assert m["n_days"] == 0
    assert m["rank_ic"] is None  # nothing to average


# ===================================================================
# C4b — _evaluate dispatches to xs metrics (no AUC)
# ===================================================================

class _FakeProbaModel:
    def __init__(self, scores):
        self._scores = np.asarray(scores)

    def predict_proba(self, X):
        s = self._scores[: len(X)]
        return np.column_stack([1.0 - s, s])


def test_evaluate_xs_strong_emits_rank_ic_not_auc():
    from app.services.training_service import TrainingService

    svc = TrainingService()
    n = 5
    X = pd.DataFrame({"f": range(n)})
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    fwd = np.array([0.05, 0.04, 0.03, 0.02, 0.01])
    g = pd.DataFrame({"date": ["2025-01-02"] * n, "future_return": fwd})
    empty = pd.DataFrame()
    es = pd.Series(dtype=float)

    metrics = svc._evaluate(
        _FakeProbaModel(scores),
        empty, es, empty, es, X, pd.Series([0, 0, 1, 1, 1]),
        task="xs_strong",
        groups={"train": None, "val": None, "test": g},
        top_pct=0.4,
    )
    assert "test_rank_ic" in metrics
    assert "test_precision_at_top" in metrics
    assert "test_auc" not in metrics
    assert metrics["test_rank_ic"] == pytest.approx(1.0, abs=1e-6)


# ===================================================================
# C5 — end-to-end train through TrainingService + blob round-trip
# ===================================================================

pytest.importorskip("xgboost")

_XS_TICKERS = ["AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH"]


def _train_xs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.ml.dataset.builder.get_market_provider",
        lambda *a, **k: _FakeProvider(n=300),
    )
    from app.services.training_service import TrainingService, TrainRequest

    svc = TrainingService(artifacts_path=str(tmp_path))
    return svc.train(
        TrainRequest(
            tickers=_XS_TICKERS,
            feature_names=_XS_FEATURES,
            feature_groups=["momentum", "volatility"],
            label_type="xs_strong",
            model_type="xgboost",
            horizon_days=5,
            top_pct=0.30,
            search_mode="none",
            save_model=True,
        )
    )


def test_train_xs_strong_end_to_end(tmp_path, monkeypatch):
    res = _train_xs(tmp_path, monkeypatch)
    assert res.success, res.error
    # cross-sectional metrics present, AUC absent
    assert "test_rank_ic" in res.metrics
    assert "test_precision_at_top" in res.metrics
    assert "test_auc" not in res.metrics
    # artifact written to disk (the thing we then zip into a blob)
    assert res.model_path and Path(res.model_path).exists()


def test_xs_model_blob_roundtrip(tmp_path, monkeypatch):
    """Exit criterion: prod can load the persisted model and score a fresh frame."""
    res = _train_xs(tmp_path, monkeypatch)
    assert res.success, res.error

    from app.ml.models.base import zip_model_dir
    from app.ml.models.xgboost_model import XGBoostModel

    blob = zip_model_dir(res.model_path)  # exactly what save_blob stores
    loaded = XGBoostModel.from_zip_bytes(blob)
    feats = loaded.metadata.feature_names
    assert feats == _XS_FEATURES
    X = pd.DataFrame(
        np.random.default_rng(0).normal(size=(10, len(feats))), columns=feats
    )
    proba = loaded.predict_proba(X)
    assert proba.shape == (10, 2)


# ===================================================================
# C5 (DB) — training reads the prices DB table, not yahoo
# ===================================================================

def test_dataset_config_market_provider_knob():
    from app.ml.dataset.schemas import DatasetConfig

    assert DatasetConfig(tickers=["AAPL"], market_provider="db").market_provider == "db"
    assert DatasetConfig(tickers=["AAPL"]).market_provider is None  # default = settings


def test_prices_db_provider_reads_repo_and_filters_dates(monkeypatch):
    from app.providers.market.prices_db import PricesDBProvider

    rows = pd.DataFrame(
        {
            "ticker": ["X"] * 5,
            "date": pd.bdate_range("2024-01-01", periods=5),
            "open": [1.0, 2, 3, 4, 5],
            "high": [1.0, 2, 3, 4, 5],
            "low": [1.0, 2, 3, 4, 5],
            "close": [1.0, 2, 3, 4, 5],
            "volume": [10, 20, 30, 40, 50],
        }
    )
    monkeypatch.setattr(
        "app.db.prices_repo.get_prices_df", lambda ticker, limit=0: rows.copy()
    )
    prov = PricesDBProvider()
    out = prov.fetch("X")
    assert "close" in out.columns and "ticker" in out.columns
    assert len(out) == 5
    # date filtering honored
    win = prov.fetch("X", start_date=pd.Timestamp("2024-01-03").date())
    assert len(win) == 3
    # empty repo -> empty frame (not a crash)
    monkeypatch.setattr("app.db.prices_repo.get_prices_df", lambda ticker, limit=0: None)
    assert PricesDBProvider().fetch("X").empty


def test_get_market_provider_db_registered():
    from app.providers.factory import get_market_provider
    from app.providers.market.prices_db import PricesDBProvider

    assert isinstance(get_market_provider("db"), PricesDBProvider)


# ===================================================================
# Review hardening — degenerate cross-sections + provider contract
# ===================================================================

def test_xs_label_constant_date_is_all_zero():
    """A date with no return spread has no 'strong' group → all 0 (not all 1)."""
    from app.ml.dataset.schemas import LabelConfig
    from app.ml.labels.xs_strong import add_xs_strong_label

    panel = pd.DataFrame(
        {
            "date": ["2025-01-02"] * 5,
            "ticker": list("ABCDE"),
            "future_return": [0.01] * 5,
        }
    )
    out = add_xs_strong_label(panel, LabelConfig(label_type="xs_strong", top_pct=0.3))
    assert (out["label"] == 0).all()


class _StaggeredProvider(_FakeProvider):
    """Some tickers start later → early dates have a thin cross-section."""

    def __init__(self, late, offset=120, n=240):
        super().__init__(n)
        self._late = set(late)
        self._offset = offset

    def fetch(self, ticker, start_date=None, end_date=None, **kwargs):
        df = super().fetch(ticker, start_date, end_date, **kwargs)
        if ticker in self._late:
            df = df.iloc[self._offset:].reset_index(drop=True)
        return df


def test_xs_build_drops_thin_dates():
    from app.ml.dataset.builder import DatasetBuilder, MIN_CROSS_SECTION
    from app.ml.dataset.schemas import DatasetConfig, LabelConfig

    tickers = ["AA", "BB", "CC", "DD", "EE", "FF"]
    cfg = DatasetConfig(
        tickers=tickers,
        feature_names=_XS_FEATURES,
        label_config=LabelConfig(label_type="xs_strong", horizon_days=5, top_pct=0.30),
        min_samples_per_ticker=50,
    )
    builder = DatasetBuilder(cfg)
    builder.market_provider = _StaggeredProvider(late=["EE", "FF"])
    out = builder.build()

    allg = pd.concat([out.groups_train, out.groups_val, out.groups_test])
    counts = allg.groupby("date").size()
    # every surviving date has at least the floor; thin (4-name) early dates gone
    assert counts.min() >= MIN_CROSS_SECTION


def test_prices_db_provider_requires_ohlcv_columns(monkeypatch):
    from app.providers.market.prices_db import PricesDBProvider

    bad = pd.DataFrame(
        {"ticker": ["X"] * 3, "date": pd.bdate_range("2024-01-01", periods=3), "close": [1, 2, 3]}
    )
    monkeypatch.setattr("app.db.prices_repo.get_prices_df", lambda ticker, limit=0: bad.copy())
    with pytest.raises(ValueError):
        PricesDBProvider().fetch("X")
