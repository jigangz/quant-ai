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


# ===================================
# Model Cache & Promotion
# ===================================
class CacheStatsResponse(BaseModel):
    """Response for GET /models/cache."""
    
    size: int
    max_size: int
    cached_models: list[str]
    promoted_id: str | None


class PromotionResponse(BaseModel):
    """Response for promotion endpoints."""
    
    promoted_id: str | None
    model_type: str | None = None
    tickers: list[str] = []
    metrics: dict = {}


@router.get("/models/cache", response_model=CacheStatsResponse)
def get_cache_stats():
    """
    Get model cache statistics.
    
    Returns:
    - Current cache size and max size
    - List of cached model IDs
    - Currently promoted model ID
    """
    from app.services.model_cache import get_model_cache
    
    cache = get_model_cache()
    stats = cache.stats()
    
    return CacheStatsResponse(**stats)


@router.delete("/models/cache")
def clear_cache():
    """Clear the model cache."""
    from app.services.model_cache import get_model_cache
    
    cache = get_model_cache()
    cache.clear()
    
    return {"status": "cleared"}


@router.delete("/models/cache/{model_id}")
def invalidate_cached_model(model_id: str):
    """Remove a specific model from cache."""
    from app.services.model_cache import get_model_cache
    
    cache = get_model_cache()
    cache.invalidate(model_id)
    
    return {"status": "invalidated", "model_id": model_id}


@router.get("/models/promoted", response_model=PromotionResponse)
def get_promoted_model():
    """
    Get the currently promoted (production) model.
    
    Returns model info if one is promoted, or null fields if none.
    """
    from app.services.model_cache import get_model_cache
    
    cache = get_model_cache()
    model_id, model = cache.get_promoted()
    
    if not model_id:
        return PromotionResponse(promoted_id=None)
    
    # Get model record for metadata
    registry = get_model_registry()
    record = registry.get_model(model_id)
    
    return PromotionResponse(
        promoted_id=model_id,
        model_type=record.model_type if record else None,
        tickers=record.tickers if record else [],
        metrics=record.metrics if record else {},
    )


@router.post("/models/{model_id}/promote", response_model=PromotionResponse)
def promote_model(model_id: str):
    """
    Promote a model to production.
    
    The promoted model:
    - Is never evicted from cache
    - Is used as the default for /predict
    - Can be retrieved via GET /models/promoted
    """
    from app.services.model_cache import get_model_cache
    
    # Check model exists
    registry = get_model_registry()
    record = registry.get_model(model_id)
    
    if not record:
        raise HTTPException(
            status_code=404,
            detail={"error": "Model not found", "model_id": model_id},
        )
    
    # Promote
    cache = get_model_cache()
    success = cache.promote(model_id)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to promote model"},
        )
    
    logger.info(f"Model promoted to production: {model_id}")
    
    return PromotionResponse(
        promoted_id=model_id,
        model_type=record.model_type,
        tickers=record.tickers,
        metrics=record.metrics,
    )


@router.delete("/models/promoted")
def demote_model():
    """Remove the current production model promotion."""
    from app.services.model_cache import get_model_cache
    
    cache = get_model_cache()
    old_id = cache.get_promoted_id()
    cache.demote()
    
    logger.info(f"Model demoted: {old_id}")
    
    return {"status": "demoted", "previous_id": old_id}
