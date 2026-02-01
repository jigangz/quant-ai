"""
Model Cache Service

Provides:
- LRU cache for loaded models (fast inference)
- Promoted model concept (production version)
- Thread-safe model loading
"""

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from app.core.settings import settings
from app.ml.models import BaseModel, ModelFactory

logger = logging.getLogger(__name__)


class ModelCache:
    """
    LRU cache for loaded ML models.
    
    Features:
    - Caches up to N models in memory
    - Thread-safe access
    - Automatic eviction of least-recently-used
    - Promoted model always kept in cache
    
    Usage:
        cache = ModelCache(max_size=5)
        model = cache.get("model_123")  # Loads and caches
        model = cache.get("model_123")  # Returns cached
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, max_size: int = 10):
        if self._initialized:
            return
            
        self.max_size = max_size
        self._cache: OrderedDict[str, BaseModel] = OrderedDict()
        self._promoted_id: str | None = None
        self._model_lock = threading.Lock()
        self._initialized = True
        
        logger.info(f"ModelCache initialized with max_size={max_size}")
    
    def get(self, model_id: str, model_type: str | None = None) -> BaseModel | None:
        """
        Get a model from cache, loading if necessary.
        
        Args:
            model_id: Model ID to load
            model_type: Model type (required if not cached)
            
        Returns:
            Loaded model or None if not found
        """
        with self._model_lock:
            # Check cache
            if model_id in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(model_id)
                logger.debug(f"Cache hit: {model_id}")
                return self._cache[model_id]
        
        # Not in cache - load from disk
        model = self._load_model(model_id, model_type)
        
        if model is not None:
            with self._model_lock:
                self._add_to_cache(model_id, model)
        
        return model
    
    def _load_model(self, model_id: str, model_type: str | None) -> BaseModel | None:
        """Load model from disk."""
        from app.db.model_registry import get_model_registry
        
        # Get model record for path and type
        registry = get_model_registry()
        record = registry.get_model(model_id)
        
        if not record:
            logger.warning(f"Model not found in registry: {model_id}")
            return None
        
        model_path = record.artifact_path
        model_type = model_type or record.model_type
        
        if not model_path:
            logger.warning(f"No artifact path for model: {model_id}")
            return None
        
        path = Path(model_path)
        if not path.exists():
            logger.warning(f"Model path does not exist: {model_path}")
            return None
        
        try:
            # Get model class and load
            model_class = ModelFactory._registry.get(model_type)
            if not model_class:
                logger.error(f"Unknown model type: {model_type}")
                return None
            
            model = model_class.load(path)
            logger.info(f"Loaded model from disk: {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return None
    
    def _add_to_cache(self, model_id: str, model: BaseModel):
        """Add model to cache, evicting if necessary."""
        # If at capacity, evict LRU (but not promoted)
        while len(self._cache) >= self.max_size:
            # Find oldest that's not promoted
            for key in self._cache:
                if key != self._promoted_id:
                    evicted = self._cache.pop(key)
                    logger.info(f"Evicted model from cache: {key}")
                    break
            else:
                # All models are promoted? Just evict oldest
                oldest = next(iter(self._cache))
                self._cache.pop(oldest)
                logger.warning(f"Evicted promoted model: {oldest}")
        
        self._cache[model_id] = model
        logger.debug(f"Added to cache: {model_id} (size={len(self._cache)})")
    
    def invalidate(self, model_id: str):
        """Remove a model from cache."""
        with self._model_lock:
            if model_id in self._cache:
                del self._cache[model_id]
                logger.info(f"Invalidated cache: {model_id}")
    
    def clear(self):
        """Clear the entire cache."""
        with self._model_lock:
            self._cache.clear()
            logger.info("Cache cleared")
    
    # === Promotion System ===
    
    def promote(self, model_id: str) -> bool:
        """
        Promote a model to production.
        
        The promoted model is:
        - Never evicted from cache
        - Returned by get_promoted()
        - Used as default for predictions
        
        Args:
            model_id: Model to promote
            
        Returns:
            True if successful
        """
        # Verify model exists
        model = self.get(model_id)
        if model is None:
            logger.error(f"Cannot promote non-existent model: {model_id}")
            return False
        
        # Update promotion
        old_promoted = self._promoted_id
        self._promoted_id = model_id
        
        # Persist promotion
        self._save_promotion(model_id)
        
        logger.info(f"Promoted model: {model_id} (was: {old_promoted})")
        return True
    
    def demote(self):
        """Remove production promotion."""
        old = self._promoted_id
        self._promoted_id = None
        self._save_promotion(None)
        logger.info(f"Demoted model: {old}")
    
    def get_promoted(self) -> tuple[str | None, BaseModel | None]:
        """
        Get the promoted (production) model.
        
        Returns:
            (model_id, model) or (None, None)
        """
        if not self._promoted_id:
            # Try to load from disk
            self._promoted_id = self._load_promotion()
        
        if not self._promoted_id:
            return None, None
        
        model = self.get(self._promoted_id)
        return self._promoted_id, model
    
    def get_promoted_id(self) -> str | None:
        """Get just the promoted model ID."""
        if not self._promoted_id:
            self._promoted_id = self._load_promotion()
        return self._promoted_id
    
    def _save_promotion(self, model_id: str | None):
        """Persist promotion to disk."""
        promotion_file = Path(settings.STORAGE_LOCAL_PATH) / ".promoted_model"
        
        if model_id:
            promotion_file.write_text(model_id)
        elif promotion_file.exists():
            promotion_file.unlink()
    
    def _load_promotion(self) -> str | None:
        """Load promotion from disk."""
        promotion_file = Path(settings.STORAGE_LOCAL_PATH) / ".promoted_model"
        
        if promotion_file.exists():
            return promotion_file.read_text().strip()
        return None
    
    # === Stats ===
    
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._model_lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "cached_models": list(self._cache.keys()),
                "promoted_id": self._promoted_id,
            }


# Singleton accessor
_cache_instance: ModelCache | None = None


def get_model_cache(max_size: int = 10) -> ModelCache:
    """Get the global model cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ModelCache(max_size=max_size)
    return _cache_instance
