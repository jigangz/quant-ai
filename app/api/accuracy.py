"""Accuracy API — V4 P5 G1: GET /models/{id}/accuracy."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services import accuracy_service

router = APIRouter()


def _get_model_record(model_id: str) -> dict | None:
    """Return minimal record dict ({label_type:...}) or None. Wrapped for tests."""
    try:
        from app.db.model_registry import get_model_registry
        registry = get_model_registry()
        record = registry.get_model(model_id)
        if record is None:
            return None
        return {"label_type": record.label_type, "model_type": record.model_type}
    except Exception:
        return None


@router.get("/models/{model_id}/accuracy")
def get_model_accuracy(
    model_id: str,
    window_days: int = Query(default=30, ge=1, le=365),
    resolve: bool = Query(default=True),
):
    record = _get_model_record(model_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"model_not_found:{model_id}")

    resolve_run = (
        accuracy_service.resolve_pending(model_id, limit=100)
        if resolve else
        {"checked": 0, "newly_resolved": 0, "errors": 0}
    )
    stats = accuracy_service.aggregate(model_id, window_days=window_days)
    bt = accuracy_service.by_ticker(model_id, window_days=window_days)
    last = accuracy_service.last_predictions(model_id, limit=20)

    return {
        "model_id": model_id,
        "label_type": record["label_type"],
        "window_days": window_days,
        "resolve_run": resolve_run,
        "stats": stats,
        "by_ticker": bt,
        "last_predictions": last,
    }
