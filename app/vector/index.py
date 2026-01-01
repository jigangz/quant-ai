import faiss
import numpy as np
from app.vector.store import VectorStore


class FaissIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.store = VectorStore()
        self.next_id = 0

    def add(self, embeddings: np.ndarray, records: list[dict]):
        """
        embeddings: (N, dim)
        records: list of metadata dicts
        """
        assert len(embeddings) == len(records)

        start_id = self.next_id
        self.index.add(embeddings)

        for i, record in enumerate(records):
            self.store.add(start_id + i, record)

        self.next_id += len(records)

    def search(self, query_vec: np.ndarray, top_k: int = 5):
        """
        query_vec: (1, dim)
        """
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            record = self.store.get(idx)
            if record:
                record = record.copy()
                record["score"] = float(score)
                results.append(record)

        return results
