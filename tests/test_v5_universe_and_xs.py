"""V5 survivorship + cross-section-normalization fixes (pure-logic tests)."""

import numpy as np
import pandas as pd
import pytest

from app.data.universe import reconstruct_members
from app.ml.features.cross_section import cross_section_normalize


# --- point-in-time membership (survivorship fix) ---

def _changes():
    # NEWCO added 2025-06-01; OLDCO removed 2025-06-01.
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-06-01")],
            "added": ["NEWCO"],
            "removed": ["OLDCO"],
        }
    )


def test_membership_before_change_has_old_not_new():
    current = {"AAPL", "MSFT", "NEWCO"}  # NEWCO is in the current list
    members = reconstruct_members(current, _changes(), pd.Timestamp("2025-01-01"))
    # Before 2025-06-01: NEWCO wasn't a member yet, OLDCO still was.
    assert "NEWCO" not in members
    assert "OLDCO" in members
    assert {"AAPL", "MSFT"} <= members


def test_membership_after_change_matches_current():
    current = {"AAPL", "MSFT", "NEWCO"}
    members = reconstruct_members(current, _changes(), pd.Timestamp("2025-12-01"))
    # After the change: current list is correct as-is.
    assert members == current


def test_no_changes_returns_current():
    current = {"AAPL", "MSFT"}
    empty = pd.DataFrame({"date": [], "added": [], "removed": []})
    assert reconstruct_members(current, empty, pd.Timestamp("2024-01-01")) == current


# --- per-date cross-section normalization (standardization fix) ---

def _panel():
    # 2 dates x 3 tickers, one factor "mom" with different scales per date.
    rows = []
    for d, vals in {
        "2025-01-01": [10.0, 20.0, 30.0],
        "2025-01-02": [100.0, 200.0, 300.0],
    }.items():
        for t, v in zip(["A", "B", "C"], vals):
            rows.append({"date": d, "ticker": t, "mom": v})
    return pd.DataFrame(rows)


def test_each_date_is_zero_mean_unit_std():
    out = cross_section_normalize(_panel(), ["mom"], winsor=0.0)
    for _, g in out.groupby("date"):
        assert g["mom"].mean() == pytest.approx(0.0, abs=1e-9)
        assert g["mom"].std() == pytest.approx(1.0, abs=1e-9)


def test_normalization_is_per_date_not_global():
    # Same shape per date -> identical normalized values across dates,
    # proving each date used its own (not a global) mean/std.
    out = cross_section_normalize(_panel(), ["mom"], winsor=0.0)
    d1 = out[out.date == "2025-01-01"]["mom"].to_numpy()
    d2 = out[out.date == "2025-01-02"]["mom"].to_numpy()
    assert np.allclose(d1, d2)


def test_zero_variance_group_becomes_zero():
    df = pd.DataFrame(
        {"date": ["2025-01-01"] * 3, "ticker": list("ABC"), "mom": [5.0, 5.0, 5.0]}
    )
    out = cross_section_normalize(df, ["mom"])
    assert (out["mom"] == 0.0).all()
