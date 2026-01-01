from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/rag/answer")
def rag_answer():
    """
    Placeholder for RAG-based answering.

    v2 will support:
    - retrieval over SHAP / research / errors
    - LLM-based reasoning
    """
    raise HTTPException(
        status_code=501,
        detail="RAG answering coming soon (v2)",
    )
