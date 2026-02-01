"""
RAG Index - FAISS vector index for document retrieval

Indexes:
- Training run metadata and results
- Model documentation
- Feature descriptions
- Error logs
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.core.settings import settings

logger = logging.getLogger(__name__)

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. RAG will use fallback.")


class RAGIndex:
    """
    FAISS-based vector index for RAG retrieval.
    
    Usage:
        index = RAGIndex()
        index.add_documents([
            {"id": "1", "text": "Model trained on AAPL", "type": "training"},
            {"id": "2", "text": "RSI indicates overbought", "type": "feature"},
        ])
        results = index.search("What model was trained?", k=3)
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize the index.
        
        Args:
            dimension: Embedding dimension (384 for MiniLM)
        """
        self.dimension = dimension
        self.documents: list[dict[str, Any]] = []
        
        if FAISS_AVAILABLE:
            # Use FAISS IndexFlatIP (inner product for normalized vectors)
            self.index = faiss.IndexFlatIP(dimension)
        else:
            # Fallback: numpy-based similarity
            self.index = None
            self._embeddings = None
        
        logger.info(f"RAGIndex initialized (FAISS={FAISS_AVAILABLE})")
    
    def add_documents(self, docs: list[dict[str, Any]]):
        """
        Add documents to the index.
        
        Each doc should have at least:
        - text: The content to embed
        - id: Unique identifier
        - type: Document type (training, feature, error, etc.)
        """
        from app.vector.embedder import embed_texts
        
        if not docs:
            return
        
        # Extract texts
        texts = [doc.get("text", "") for doc in docs]
        
        # Get embeddings
        embeddings = embed_texts(texts)
        
        # Store documents
        start_idx = len(self.documents)
        self.documents.extend(docs)
        
        # Add to index
        if FAISS_AVAILABLE:
            self.index.add(embeddings.astype(np.float32))
        else:
            if self._embeddings is None:
                self._embeddings = embeddings
            else:
                self._embeddings = np.vstack([self._embeddings, embeddings])
        
        logger.info(f"Added {len(docs)} documents (total: {len(self.documents)})")
    
    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of documents with scores
        """
        from app.vector.embedder import embed_texts
        
        if len(self.documents) == 0:
            return []
        
        # Embed query
        query_embedding = embed_texts([query])[0]
        
        # Search
        k = min(k, len(self.documents))
        
        if FAISS_AVAILABLE:
            scores, indices = self.index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                k
            )
            scores = scores[0]
            indices = indices[0]
        else:
            # Numpy fallback
            scores = np.dot(self._embeddings, query_embedding)
            indices = np.argsort(scores)[::-1][:k]
            scores = scores[indices]
        
        # Build results
        results = []
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx].copy()
            doc["score"] = float(score)
            results.append(doc)
        
        return results
    
    def size(self) -> int:
        """Get number of indexed documents."""
        return len(self.documents)
    
    def clear(self):
        """Clear the index."""
        self.documents = []
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self._embeddings = None
        logger.info("Index cleared")


# Singleton instance
_rag_index: RAGIndex | None = None


def get_rag_index() -> RAGIndex:
    """Get the global RAG index."""
    global _rag_index
    if _rag_index is None:
        _rag_index = RAGIndex()
        # Auto-index on first access
        _auto_index()
    return _rag_index


def _auto_index():
    """Auto-index available documents on startup."""
    try:
        index_training_runs()
        index_feature_docs()
        index_model_docs()
    except Exception as e:
        logger.warning(f"Auto-indexing failed: {e}")


def index_training_runs():
    """Index training run records."""
    from app.db.model_registry import get_model_registry
    
    registry = get_model_registry()
    runs = registry.list_runs(limit=100)
    
    docs = []
    for run in runs:
        text = (
            f"Training run {run.id[:8]}: "
            f"trained {run.model_type} model on {', '.join(run.tickers)}. "
            f"Features: {', '.join(run.feature_groups)}. "
            f"Horizon: {run.horizon_days} days. "
        )
        if run.metrics:
            metrics_str = ", ".join(f"{k}={v}" for k, v in run.metrics.items())
            text += f"Results: {metrics_str}. "
        if run.error:
            text += f"Error: {run.error}. "
        if run.git_sha:
            text += f"Git: {run.git_sha}. "
        
        docs.append({
            "id": run.id,
            "type": "training_run",
            "text": text,
            "model_type": run.model_type,
            "tickers": run.tickers,
            "success": run.success,
        })
    
    if docs:
        get_rag_index().add_documents(docs)
        logger.info(f"Indexed {len(docs)} training runs")


def index_feature_docs():
    """Index feature group documentation."""
    from app.ml.features.registry import feature_registry
    
    docs = []
    for group_name in feature_registry.list_groups():
        group = feature_registry.get(group_name)
        if group:
            text = (
                f"Feature group '{group_name}': {group.description}. "
                f"Contains features: {', '.join(group.feature_names[:10])}. "
            )
            docs.append({
                "id": f"feature_{group_name}",
                "type": "feature_group",
                "text": text,
                "group_name": group_name,
                "feature_count": len(group.feature_names),
            })
    
    if docs:
        get_rag_index().add_documents(docs)
        logger.info(f"Indexed {len(docs)} feature groups")


def index_model_docs():
    """Index model type documentation."""
    from app.ml.models import ModelFactory
    
    model_descriptions = {
        "logistic": "Logistic Regression: Linear model for binary classification. Fast, interpretable, good baseline. Uses L2 regularization.",
        "random_forest": "Random Forest: Ensemble of decision trees. Handles non-linear patterns, provides feature importance. Robust to outliers.",
        "xgboost": "XGBoost: Gradient boosting with regularization. High accuracy, handles missing values. Supports GPU acceleration.",
        "lightgbm": "LightGBM: Fast gradient boosting using histogram-based algorithm. Lower memory usage, faster training than XGBoost.",
        "catboost": "CatBoost: Gradient boosting with ordered boosting. Native categorical feature support, robust to overfitting.",
    }
    
    docs = []
    for model_type, desc in model_descriptions.items():
        if ModelFactory.is_available(model_type):
            docs.append({
                "id": f"model_{model_type}",
                "type": "model_doc",
                "text": desc,
                "model_type": model_type,
            })
    
    if docs:
        get_rag_index().add_documents(docs)
        logger.info(f"Indexed {len(docs)} model docs")
