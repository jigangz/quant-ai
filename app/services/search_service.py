"""
Search service using FAISS-based vector index.

This module provides a semantic search capability over stored text
embeddings. The previous implementation attempted to call a
``search`` method on the in-memory ``VectorStore`` metadata store,
which does not support searching and only holds metadata.  As a
result, queries to the ``/search`` API endpoint resulted in an
``AttributeError`` complaining that ``VectorStore`` has no attribute
``search``.  To resolve this, we now initialise a FAISS index
(``FaissIndex``) on first use and delegate all similarity search
requests to it.  The index is created with an embedding dimension
determined dynamically from the model used in ``embedder.embed_texts``.

Because this is a simple demonstration, the index starts out empty
and therefore returns no results until vectors have been added via
other services.  Future versions may populate the index with
explanations, notes or other domain-specific text.
"""

from __future__ import annotations

from typing import Dict

from app.vector.embedder import embed_texts
from app.vector.index import FaissIndex

_faiss_index: FaissIndex | None = None


def _get_faiss_index() -> FaissIndex:
    """Lazily initialise and return a global FAISS index.

    The index is initialised on first access with a dimensionality
    matching the embedding model.  Subsequent calls return the
    previously created instance.

    Returns
    -------
    FaissIndex
        The global FAISS vector index.
    """
    global _faiss_index
    if _faiss_index is None:
        # Determine the embedding dimensionality by encoding a dummy
        # string.  This avoids hard‑coding model dimensions and ensures
        # consistency with the embedder configuration.
        test_vec = embed_texts(["test"])
        dim = test_vec.shape[1]
        _faiss_index = FaissIndex(dim=dim)
    return _faiss_index


def search(query: str, top_k: int = 5) -> Dict:
    """Perform a semantic search over indexed embeddings.

    Parameters
    ----------
    query : str
        The natural language query to embed and search for similar
        documents.
    top_k : int, optional
        The maximum number of results to return, by default 5.

    Returns
    -------
    dict
        A dictionary containing the query, status and a list of
        matching records with similarity scores.  If no vectors have
        been indexed yet, the results list will be empty.
    """
    # Embed the query text into a vector.  The embedder returns a
    # numpy array with shape (1, dim).
    q_vec = embed_texts([query])

    # Retrieve or initialise the FAISS index and perform the search.
    index = _get_faiss_index()
    results = index.search(q_vec, top_k=top_k)

    return {
        "status": "ok",
        "query": query,
        "results": results,
    }
