"""
Text Embedder - Sentence Transformers (optional)

Falls back to simple hashing if not installed.
"""

import logging
import hashlib
import numpy as np

logger = logging.getLogger(__name__)

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDER_AVAILABLE = True
except ImportError:
    EMBEDDER_AVAILABLE = False
    SentenceTransformer = None
    logger.info("sentence-transformers not installed, using hash fallback")

_model = None


def get_embedder():
    """Get sentence transformer model."""
    global _model
    
    if not EMBEDDER_AVAILABLE:
        return None
    
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    
    return _model


def embed_texts(texts: list[str], dimension: int = 384) -> np.ndarray:
    """
    Embed texts into vectors.
    
    Uses sentence-transformers if available, otherwise falls back
    to a simple hash-based embedding (for basic functionality).
    """
    if EMBEDDER_AVAILABLE:
        model = get_embedder()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings
    else:
        # Fallback: hash-based pseudo-embeddings
        embeddings = []
        for text in texts:
            # Create deterministic pseudo-embedding from hash
            hash_bytes = hashlib.sha256(text.encode()).digest()
            # Expand hash to fill dimension
            np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
            vec = np.random.randn(dimension).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # Normalize
            embeddings.append(vec)
        return np.array(embeddings)
