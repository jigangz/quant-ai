"""
RAG API - Retrieval Augmented Generation

POST /rag/answer - Answer questions using indexed documents
GET /rag/index - Get index statistics
POST /rag/index/refresh - Refresh the index
POST /rag/search - Raw semantic search
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# ===================================
# Request/Response Schemas
# ===================================
class RAGQueryRequest(BaseModel):
    """Request for /rag/answer."""
    
    query: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class EvidenceItem(BaseModel):
    """A piece of evidence."""
    
    id: str
    type: str
    text: str
    score: float
    metadata: dict[str, Any] = {}


class RAGAnswerResponse(BaseModel):
    """Response for /rag/answer."""
    
    query: str
    answer: str
    evidence: list[EvidenceItem]
    confidence: float


class IndexStatsResponse(BaseModel):
    """Response for /rag/index."""
    
    total_documents: int
    document_types: dict[str, int]
    faiss_available: bool


class SearchRequest(BaseModel):
    """Request for /rag/search."""
    
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=10, ge=1, le=50)


class SearchResult(BaseModel):
    """A search result."""
    
    id: str
    type: str
    text: str
    score: float


class SearchResponse(BaseModel):
    """Response for /rag/search."""
    
    query: str
    results: list[SearchResult]
    total: int


# ===================================
# POST /rag/answer
# ===================================
@router.post("/rag/answer", response_model=RAGAnswerResponse)
def rag_answer_endpoint(request: RAGQueryRequest):
    """
    Answer a question using RAG.
    
    Retrieves relevant documents from the index and generates
    a summary answer with evidence.
    
    Example questions:
    - "What models have been trained?"
    - "What is XGBoost good for?"
    - "What features are in ta_basic?"
    - "What was the performance of the last training run?"
    """
    from app.rag.answer import rag_answer
    
    try:
        result = rag_answer(request.query, top_k=request.top_k)
        
        return RAGAnswerResponse(
            query=result.query,
            answer=result.answer,
            evidence=[
                EvidenceItem(
                    id=e.id,
                    type=e.type,
                    text=e.text,
                    score=e.score,
                    metadata=e.metadata,
                )
                for e in result.evidence
            ],
            confidence=result.confidence,
        )
    
    except Exception as e:
        logger.error(f"RAG answer failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ===================================
# GET /rag/index
# ===================================
@router.get("/rag/index", response_model=IndexStatsResponse)
def get_index_stats():
    """
    Get RAG index statistics.
    
    Returns document counts by type and FAISS availability.
    """
    from app.rag.index import get_rag_index, FAISS_AVAILABLE
    
    index = get_rag_index()
    
    # Count by type
    type_counts: dict[str, int] = {}
    for doc in index.documents:
        doc_type = doc.get("type", "unknown")
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    
    return IndexStatsResponse(
        total_documents=index.size(),
        document_types=type_counts,
        faiss_available=FAISS_AVAILABLE,
    )


# ===================================
# POST /rag/index/refresh
# ===================================
@router.post("/rag/index/refresh")
def refresh_index():
    """
    Refresh the RAG index.
    
    Re-indexes all documents from:
    - Training runs
    - Feature documentation
    - Model documentation
    """
    from app.rag.index import (
        get_rag_index,
        index_training_runs,
        index_feature_docs,
        index_model_docs,
    )
    
    # Clear and rebuild
    index = get_rag_index()
    old_size = index.size()
    index.clear()
    
    try:
        index_training_runs()
        index_feature_docs()
        index_model_docs()
    except Exception as e:
        logger.error(f"Index refresh failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "status": "refreshed",
        "previous_size": old_size,
        "new_size": index.size(),
    }


# ===================================
# POST /rag/search
# ===================================
@router.post("/rag/search", response_model=SearchResponse)
def raw_search(request: SearchRequest):
    """
    Raw semantic search over the index.
    
    Returns documents ranked by similarity score.
    Use for debugging or custom retrieval.
    """
    from app.rag.index import get_rag_index
    
    index = get_rag_index()
    results = index.search(request.query, k=request.k)
    
    return SearchResponse(
        query=request.query,
        results=[
            SearchResult(
                id=r.get("id", ""),
                type=r.get("type", "unknown"),
                text=r.get("text", ""),
                score=r.get("score", 0),
            )
            for r in results
        ],
        total=len(results),
    )
