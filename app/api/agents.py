from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/agents/run")
def run_agent():
    """
    Placeholder for agent execution.

    v2 will support:
    - multi-step reasoning
    - tool calling (predict / explain / search)
    - portfolio-level decisions
    """
    raise HTTPException(
        status_code=501,
        detail="Agent runtime coming soon (v2)",
    )
