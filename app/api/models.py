"""
Models API - Model Registry endpoints

GET /models/types - List available model types
GET /models - List all models
GET /models/{id} - Get model metadata
PATCH /models/{id} - Update model (e.g., archive)
DELETE /models/{id} - Soft delete model
"""

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.db.model_registry import ModelRecord, get_model_registry

logger = logging.getLogger(__name__)
router = APIRouter()


# ===================================
# Model Type Info
# ===================================
class ModelTypeInfo(BaseModel):
    """Info about an available model type."""
    
    type: str
    class_name: str
    available: bool = True


class ModelTypesResponse(BaseModel):
    """Response for GET /models/types."""
    
    types: list[ModelTypeInfo]
    total: int


@router.get("/models/types", response_model=ModelTypesResponse)
def list_model_types():
    """
    List all available model types.
    
    Returns which model types can be used for training:
    - logistic: Logistic Regression (always available)
    - random_forest: Random Forest (always available)
    - xgboost: XGBoost (optional)
    - lightgbm: LightGBM (optional)
    - catboost: CatBoost (optional)
    """
    from app.ml.models import ModelFactory
    
    model_info = ModelFactory.get_model_info()
    
    types = [
        ModelTypeInfo(
            type=info["type"],
            class_name=info["class"],
            available=info.get("available", True),
        )
        for info in model_info
    ]
    
    return ModelTypesResponse(types=types, total=len(types))


# ===================================
# Response Schemas
# ===================================
class ModelListResponse(BaseModel):
    """Response for GET /models."""

    models: list[ModelRecord]
    total: int


class ModelUpdateRequest(BaseModel):
    """Request for PATCH /models/{id}."""

    name: str | None = None
    status: Literal["active", "archived"] | None = None


# ===================================
# GET /models
# ===================================
@router.get("/models", response_model=ModelListResponse)
def list_models(
    status: str | None = Query("active", description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
):
    """
    List all models in the registry.

    - status: "active" (default), "archived", or None for all
    - limit: Maximum number of results (default 50)
    """
    registry = get_model_registry()
    models = registry.list_models(status=status, limit=limit)

    return ModelListResponse(
        models=models,
        total=len(models),
    )


# ===================================
# GET /models/{model_id}
# ===================================
@router.get("/models/{model_id}", response_model=ModelRecord)
def get_model(model_id: str):
    """
    Get metadata for a specific model.

    Returns 404 if model not found.
    """
    registry = get_model_registry()
    model = registry.get_model(model_id)

    if not model:
        raise HTTPException(
            status_code=404,
            detail={"error": "Model not found", "model_id": model_id},
        )

    return model


# ===================================
# PATCH /models/{model_id}
# ===================================
@router.patch("/models/{model_id}", response_model=ModelRecord)
def update_model(model_id: str, request: ModelUpdateRequest):
    """
    Update model metadata.

    Currently supports:
    - name: Rename the model
    - status: Change status (active/archived)
    """
    registry = get_model_registry()

    # Check model exists
    existing = registry.get_model(model_id)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail={"error": "Model not found", "model_id": model_id},
        )

    # Build updates
    updates = {}
    if request.name is not None:
        updates["name"] = request.name
    if request.status is not None:
        updates["status"] = request.status

    if not updates:
        return existing

    # Apply updates
    updated = registry.update_model(model_id, updates)
    if not updated:
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to update model"},
        )

    logger.info(f"Model updated: {model_id} -> {updates}")
    return updated


# ===================================
# DELETE /models/{model_id}
# ===================================
@router.delete("/models/{model_id}")
def delete_model(model_id: str):
    """
    Soft delete a model (sets status to 'deleted').

    The model artifacts are not removed.
    """
    registry = get_model_registry()

    # Check model exists
    existing = registry.get_model(model_id)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail={"error": "Model not found", "model_id": model_id},
        )

    # Soft delete
    registry.update_model(model_id, {"status": "deleted"})
    logger.info(f"Model soft deleted: {model_id}")

    return {"status": "deleted", "model_id": model_id}
