"""Tests for explain_service lazy loading."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

# Pre-import so patch() can resolve the dotted path
import app.services.explain_service as explain_mod


class TestExplainServiceLazyLoad:
    def setup_method(self):
        # Reset global explainer state before each test
        explain_mod._explainer = None

    def test_import_does_not_create_explainer_eagerly(self):
        """Module-level _explainer must be None right after (re)load."""
        with patch(
            "app.explain.shap_explainer.ShapExplainer.__init__",
            side_effect=RuntimeError("should not be called on import"),
        ):
            import importlib
            try:
                importlib.reload(explain_mod)
            except RuntimeError:
                pytest.fail("ShapExplainer instantiated at import time")
        # Reset after reload
        explain_mod._explainer = None

    def test_explain_creates_explainer_lazily(self):
        """explain() must create a ShapExplainer on first call."""
        mock_instance = MagicMock()
        mock_instance.explain.return_value = {"success": True}
        with patch.object(explain_mod, "ShapExplainer", return_value=mock_instance) as mock_cls:
            result = explain_mod.explain(ticker="AAPL")
            mock_cls.assert_called_once()
            assert result["status"] == "ok"

    def test_explain_reuses_explainer_on_second_call(self):
        """ShapExplainer must be created only once across multiple calls."""
        mock_instance = MagicMock()
        mock_instance.explain.return_value = {"success": True}
        with patch.object(explain_mod, "ShapExplainer", return_value=mock_instance) as mock_cls:
            explain_mod.explain(ticker="AAPL")
            explain_mod.explain(ticker="MSFT")
            mock_cls.assert_called_once()
