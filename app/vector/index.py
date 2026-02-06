import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    logger.info("faiss not installed, vector search disabled")

from app.vector.store import vector_store


class FaissIndex:
    def __init__(self, dim: int):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu is required for vector search. Install with: pip install faiss-cpu")
        self.index = faiss.IndexFlatIP(dim)
        self.dim = dim

    def add(self, embeddings: np.ndarray, records: List[dict]):
        start_id = self.index.ntotal
        self.index.add(embeddings)

        for i, record in enumerate(records):
            vector_store.add(start_id + i, record)

    def search(self, query_vec: np.ndarray, top_k: int = 5):
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            record = vector_store.get(int(idx))
            if record:
                results.append(
                    {
                        **record,
                        "score": float(score),
                    }
                )
        return results
