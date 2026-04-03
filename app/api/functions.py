"""
Functions API — invoke serverless functions via REST.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.functions import FUNCTIONS
from app.infra.functions import get_function_runner

router = APIRouter(prefix="/api/v1/functions")


class InvokeRequest(BaseModel):
    """Request body for function invocation."""
    payload: dict[str, Any] = {}
    async_invoke: bool = False


class InvokeResponse(BaseModel):
    """Response from function invocation."""
    function: str
    result: dict[str, Any] | None = None
    async_invoke: bool = False


@router.get("")
async def list_functions():
    """List all available functions."""
    return {
        "functions": [
            {"name": name, "handler": handler.__module__ + "." + handler.__name__}
            for name, handler in FUNCTIONS.items()
        ],
    }


@router.post("/{name}/invoke")
async def invoke_function(name: str, body: InvokeRequest):
    """Invoke a function by name."""
    if name not in FUNCTIONS:
        raise HTTPException(status_code=404, detail=f"Function not found: {name}")

    runner = get_function_runner()

    try:
        result = await runner.invoke(
            name,
            body.payload,
            async_invoke=body.async_invoke,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return InvokeResponse(
        function=name,
        result=result,
        async_invoke=body.async_invoke,
    )
