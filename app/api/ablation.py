"""Ablation API — V4 P5 FE-ENH-4: POST /api/ablation/run."""

from __future__ import annotations

from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import ablation_service

router = APIRouter()


class _FeatureSet(BaseModel):
    name: str
    groups: list[str] = Field(min_length=1)


class AblationRunRequest(BaseModel):
    ticker: str
    targets: list[Literal["direction", "volatility", "meta_label"]] = Field(min_length=1)
    feature_sets: list[_FeatureSet] = Field(min_length=2, max_length=4)
    horizon_days: int = Field(default=5, ge=1, le=60)
    model_type: Literal["xgboost", "lightgbm", "ensemble"] = "xgboost"


@router.post("/api/ablation/run")
def ablation_run(req: AblationRunRequest):
    try:
        return ablation_service.run_ablation(
            ticker=req.ticker,
            targets=list(req.targets),
            feature_sets=[fs.model_dump() for fs in req.feature_sets],
            horizon_days=req.horizon_days,
            model_type=req.model_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ablation_failed:{e}")
