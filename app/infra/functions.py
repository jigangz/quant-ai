"""
Serverless Function Abstraction (Lambda / Local)

Supports: AWS Lambda, Local (direct function call).
Swap backend via FUNCTIONS_BACKEND env var.

Usage:
    runner = get_function_runner()
    result = await runner.invoke("sentiment-analyzer", {
        "text": "AAPL beats earnings expectations",
        "model": "haiku"
    })
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable, Optional

from app.core.settings import settings

logger = logging.getLogger(__name__)


FunctionCallable = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


# ===================================
# Abstract Interface
# ===================================

class BaseFunctionRunner(ABC):
    """Abstract serverless function runner."""

    @abstractmethod
    async def invoke(
        self,
        function_name: str,
        payload: dict[str, Any],
        async_invoke: bool = False,
    ) -> Optional[dict[str, Any]]:
        """
        Invoke a function.
        
        Args:
            function_name: Function identifier
            payload: Input data
            async_invoke: If True, fire-and-forget (no response)
            
        Returns:
            Function result, or None if async_invoke=True
        """
        ...

    @abstractmethod
    def register(self, function_name: str, handler: FunctionCallable) -> None:
        """Register a local function handler (used by local backend)."""
        ...


# ===================================
# Lambda Implementation
# ===================================

class LambdaRunner(BaseFunctionRunner):
    """
    AWS Lambda function invocation.
    
    Works with: real AWS Lambda, LocalStack, SAM CLI (sam local).
    Set LAMBDA_ENDPOINT_URL=http://localhost:3001 for SAM local.
    """

    def __init__(self):
        self._client = None
        self._local_handlers: dict[str, FunctionCallable] = {}

    async def _get_client(self):
        if self._client is None:
            import aioboto3
            session = aioboto3.Session()
            endpoint = getattr(settings, "LAMBDA_ENDPOINT_URL", None)
            kwargs = {"service_name": "lambda", "region_name": getattr(settings, "AWS_REGION", "us-east-1")}
            if endpoint:
                kwargs["endpoint_url"] = endpoint
            self._client = await session.client(**kwargs).__aenter__()
        return self._client

    async def invoke(
        self,
        function_name: str,
        payload: dict[str, Any],
        async_invoke: bool = False,
    ) -> Optional[dict[str, Any]]:
        client = await self._get_client()
        
        invocation_type = "Event" if async_invoke else "RequestResponse"
        resp = await client.invoke(
            FunctionName=f"quant-ai-{function_name}",
            InvocationType=invocation_type,
            Payload=json.dumps(payload, default=str).encode(),
        )
        
        if async_invoke:
            return None
        
        response_payload = await resp["Payload"].read()
        return json.loads(response_payload)

    def register(self, function_name: str, handler: FunctionCallable) -> None:
        # Lambda doesn't use local handlers, but store for fallback
        self._local_handlers[function_name] = handler


# ===================================
# Local Implementation
# ===================================

class LocalRunner(BaseFunctionRunner):
    """
    Local function runner — calls registered Python functions directly.
    
    Perfect for development. Same interface as Lambda, zero infra.
    Register handlers at app startup, invoke just like Lambda.
    """

    def __init__(self):
        self._handlers: dict[str, FunctionCallable] = {}

    async def invoke(
        self,
        function_name: str,
        payload: dict[str, Any],
        async_invoke: bool = False,
    ) -> Optional[dict[str, Any]]:
        handler = self._handlers.get(function_name)
        if not handler:
            raise ValueError(f"No handler registered for function: {function_name}")
        
        if async_invoke:
            asyncio.create_task(handler(payload))
            return None
        
        return await handler(payload)

    def register(self, function_name: str, handler: FunctionCallable) -> None:
        self._handlers[function_name] = handler
        logger.info(f"Registered local function: {function_name}")


# ===================================
# Factory
# ===================================

_runner_instance: Optional[BaseFunctionRunner] = None


def get_function_runner() -> BaseFunctionRunner:
    """
    Get the configured function runner.
    
    Set FUNCTIONS_BACKEND env var:
      - "lambda" → LambdaRunner (AWS Lambda / LocalStack / SAM)
      - "local"  → LocalRunner (direct function call, zero infra)
    
    Default: "local"
    """
    global _runner_instance
    if _runner_instance is None:
        backend = getattr(settings, "FUNCTIONS_BACKEND", "local")
        if backend == "lambda":
            _runner_instance = LambdaRunner()
        elif backend == "local":
            _runner_instance = LocalRunner()
        else:
            raise ValueError(f"Unknown functions backend: {backend}")
        logger.info(f"Functions backend: {backend}")
    return _runner_instance
