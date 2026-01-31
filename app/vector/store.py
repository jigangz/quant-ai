from typing import Dict, Any, List


class VectorStore:
    """
    In-memory metadata store for FAISS vectors.
    Index id -> record
    """

    def __init__(self):
        self._data: Dict[int, Dict[str, Any]] = {}

    def add(self, idx: int, record: Dict[str, Any]):
        self._data[idx] = record

    def get(self, idx: int) -> Dict[str, Any] | None:
        return self._data.get(idx)

    def batch_get(self, indices: List[int]) -> List[Dict[str, Any]]:
        return [self._data[i] for i in indices if i in self._data]

    def size(self) -> int:
        return len(self._data)


vector_store = VectorStore()
