from typing import List, Dict

from app.vector.embedder import embed_texts
from app.vector.store import vector_store


def search(
    query: str,
    top_k: int = 5,
) -> Dict:
    """
    Semantic search over explanations / notes.
    """
    q_vec = embed_texts([query])
    results = vector_store.search(q_vec, top_k=top_k)

    return {
        "status": "ok",
        "query": query,
        "results": results,
    }
