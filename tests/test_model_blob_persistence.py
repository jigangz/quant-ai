"""Model zip-blob round-trip — the persistence path that lets a stateless
prod (Render free tier wipes local disk) reload a model after restart."""

import numpy as np
import pandas as pd

from app.ml.models.base import zip_model_dir
from app.ml.models.factory import ModelFactory


def _tiny_xy(n=80):
    X = pd.DataFrame(np.random.RandomState(0).rand(n, 4), columns=list("abcd"))
    y = pd.Series((np.random.RandomState(1).rand(n) > 0.5).astype(int))
    return X, y


def test_to_from_zip_bytes_preserves_predictions():
    X, y = _tiny_xy()
    m = ModelFactory.create("xgboost")
    m.fit(X, y)

    blob = m.to_zip_bytes()
    assert isinstance(blob, bytes) and len(blob) > 0

    from app.ml.models.xgboost_model import XGBoostModel
    restored = XGBoostModel.from_zip_bytes(blob)

    assert (m.predict(X) == restored.predict(X)).all()


def test_zip_model_dir_matches_to_zip_bytes_roundtrip(tmp_path):
    X, y = _tiny_xy()
    m = ModelFactory.create("lightgbm")
    m.fit(X, y)

    # save to a dir, then zip the dir (the path train.py / seed use)
    m.save(tmp_path)
    blob = zip_model_dir(tmp_path)

    from app.ml.models.lightgbm_model import LightGBMModel
    restored = LightGBMModel.from_zip_bytes(blob)

    assert (m.predict(X) == restored.predict(X)).all()
