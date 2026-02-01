"""
RAG Module - Retrieval Augmented Generation

Provides:
- FAISS-based vector index
- Document indexing (training runs, features, models)
- Question answering with evidence
"""

from app.rag.index import (
    get_rag_index,
    RAGIndex,
    index_training_runs,
    index_feature_docs,
    index_model_docs,
)
from app.rag.answer import rag_answer, RAGService, RAGAnswer

__all__ = [
    "get_rag_index",
    "RAGIndex",
    "index_training_runs",
    "index_feature_docs",
    "index_model_docs",
    "rag_answer",
    "RAGService",
    "RAGAnswer",
]
