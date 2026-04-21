"""
Tests for ModelRecord.label_type + horizon_days (V4 Pivot P1 Day 9).

Ensures multi-task target metadata round-trips through the registry.
"""

from __future__ import annotations

from app.db.model_registry import ModelRecord


class TestModelRecordLabelType:
    def test_default_label_type_is_direction(self):
        """Backward compat: default ModelRecord has label_type='direction'."""
        rec = ModelRecord(
            name="test", model_type="logistic",
            tickers=["AAPL"], feature_groups=["ta_basic"],
        )
        assert rec.label_type == "direction"
        assert rec.horizon_days == 5

    def test_volatility_label_type_round_trip(self):
        """label_type='volatility' persists in serialization."""
        rec = ModelRecord(
            name="vol_aapl",
            model_type="xgboost",
            tickers=["AAPL"],
            feature_groups=["ta_basic"],
            label_type="volatility",
            horizon_days=10,
        )
        assert rec.label_type == "volatility"
        assert rec.horizon_days == 10

        # Serialize + deserialize
        data = rec.model_dump()
        assert data["label_type"] == "volatility"
        assert data["horizon_days"] == 10

        rec2 = ModelRecord(**data)
        assert rec2.label_type == "volatility"
        assert rec2.horizon_days == 10

    def test_legacy_record_without_label_type_defaults(self):
        """Old persisted records (pre-V4) should default to 'direction' when loaded."""
        legacy_json = {
            "name": "old_model",
            "model_type": "logistic",
            "tickers": ["AAPL"],
            "feature_groups": ["ta_basic"],
            # no label_type or horizon_days fields
        }
        rec = ModelRecord(**legacy_json)
        assert rec.label_type == "direction"
        assert rec.horizon_days == 5

    def test_meta_label_accepted(self):
        """Future V4 P3 target type accepted by schema."""
        rec = ModelRecord(
            name="meta_aapl",
            model_type="xgboost",
            tickers=["AAPL"],
            feature_groups=["ta_basic"],
            label_type="meta_label",
        )
        assert rec.label_type == "meta_label"
