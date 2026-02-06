"""
Search service using FAISS-based vector index (optional).

If faiss-cpu is not installed, search returns a 'not available' response
instead of crashing the app.
"""

from __future__ import annotations

import logging
from typing import Dict

from app.vector.embedder import embed_texts

logger = logging.getLogger(__name__)

_faiss_index = None
_faiss_available = False

try:
    from app.vector.index import FaissIndex, FAISS_AVAILABLE
    _faiss_available = FAISS_AVAILABLE
except ImportError:
    _faiss_available = False
    logger.info("FAISS not available, /search endpoint will return 501")


def _get_faiss_index():
    """Lazily initialise and return a global FAISS index."""
    global _faiss_index
    if not _faiss_available:
        return None
    if _faiss_index is None:
        from app.vector.index import FaissIndex
        test_vec = embed_texts(["test"])
        dim = test_vec.shape[1]
        _faiss_index = FaissIndex(dim=dim)
    return _faiss_index


def search(query: str, top_k: int = 5) -> Dict:
    """Perform a semantic search over indexed embeddings.

    Returns a 'not_available' status if FAISS is not installed.
    """
    if not _faiss_available:
        return {
            "status": "not_available",
            "query": query,
            "message": "Search requires faiss-cpu. Install with: pip install faiss-cpu sentence-transformers",
            "results": [],
        }

    q_vec = embed_texts([query])
    index = _get_faiss_index()
    results = index.search(q_vec, top_k=top_k)

    return {
        "status": "ok",
        "query": query,
        "results": results,
    }
