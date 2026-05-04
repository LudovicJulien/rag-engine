# src/embeddings/cache.py
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """File-based cache for embedding vectors using MD5 content hashing.

    Caches embedding vectors on disk to avoid re-embedding unchanged chunks.
    The cache key is derived from both the text content and the model name,
    ensuring cache misses when the model changes.

    Cache structure on disk:
        cache_dir/
        └── {model_key}/
            └── {content_hash}.json

    Example usage:
        cache = EmbeddingCache(cache_dir=".embedding_cache")
        cache.set(text="Le Plateau", model_key="abc123", vector=[0.1, 0.2])
        vector = cache.get(text="Le Plateau", model_key="abc123")
    """

    _CACHE_GLOB = "*.json"

    def __init__(self, cache_dir: str | Path = ".embedding_cache") -> None:
        """Initialize the embedding cache.

        Args:
            cache_dir: Directory where cached vectors are stored.
                       Created automatically if it does not exist.
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    @property
    def hit_count(self) -> int:
        """Number of cache hits since initialization."""
        return self._hits

    @property
    def miss_count(self) -> int:
        """Number of cache misses since initialization."""
        return self._misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a float between 0.0 and 1.0.

        Returns 0.0 if no lookups have been performed yet.
        """
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def get(
        self,
        text: str,
        model_key: str,
    ) -> list[float] | None:
        """Retrieve a cached embedding vector.

        Args:
            text: The text whose embedding to retrieve.
            model_key: Unique identifier for the embedding model.
                       Use EmbeddingModel.get_cache_key() for this.

        Returns:
            The cached embedding vector, or None if not cached.
        """
        path = self._cache_path(text, model_key)

        if not path.exists():
            self._misses += 1
            return None

        try:
            data: Any = json.loads(path.read_text(encoding="utf-8"))
            self._hits += 1
            logger.debug("Cache hit for text hash %s", self._content_hash(text))
            return list(data["vector"])
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning("Cache read failed for %s: %s", path, e)
            self._misses += 1
            return None

    def set(
        self,
        text: str,
        model_key: str,
        vector: list[float],
    ) -> None:
        """Store an embedding vector in the cache.

        Args:
            text: The text that was embedded.
            model_key: Unique identifier for the embedding model.
            vector: The embedding vector to cache.

        Raises:
            ValueError: If vector is empty.
        """
        if not vector:
            raise ValueError("Cannot cache an empty vector")

        path = self._cache_path(text, model_key)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            path.write_text(
                json.dumps({"vector": vector}, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.debug("Cached vector for text hash %s", self._content_hash(text))
        except OSError as e:
            logger.warning("Cache write failed for %s: %s", path, e)

    def get_batch(
        self,
        texts: list[str],
        model_key: str,
    ) -> list[list[float] | None]:
        """Retrieve cached embeddings for a batch of texts.

        Args:
            texts: List of texts to retrieve embeddings for.
            model_key: Unique identifier for the embedding model.

        Returns:
            List of cached vectors or None for cache misses,
            in the same order as input texts.
        """
        return [self.get(text, model_key) for text in texts]

    def set_batch(
        self,
        texts: list[str],
        model_key: str,
        vectors: list[list[float]],
    ) -> None:
        """Store a batch of embedding vectors in the cache.

        Args:
            texts: List of texts that were embedded.
            model_key: Unique identifier for the embedding model.
            vectors: List of embedding vectors to cache.

        Raises:
            ValueError: If texts and vectors have different lengths.
        """
        if len(texts) != len(vectors):
            raise ValueError(
                f"texts and vectors must have the same length, "
                f"got {len(texts)} and {len(vectors)}"
            )
        for text, vector in zip(texts, vectors):
            self.set(text, model_key, vector)

    def exists(self, text: str, model_key: str) -> bool:
        """Check if a cached vector exists without retrieving it.

        Args:
            text: The text to check.
            model_key: Unique identifier for the embedding model.

        Returns:
            True if a cached vector exists, False otherwise.
        """
        return self._cache_path(text, model_key).exists()

    def invalidate(self, text: str, model_key: str) -> bool:
        """Remove a specific cached vector.

        Args:
            text: The text whose cached vector to remove.
            model_key: Unique identifier for the embedding model.

        Returns:
            True if the cache entry was removed, False if it did not exist.
        """
        path = self._cache_path(text, model_key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self, model_key: str | None = None) -> int:
        """Clear cached vectors.

        Args:
            model_key: If provided, clear only vectors for this model.
                       If None, clear all cached vectors.

        Returns:
            Number of cache entries removed.
        """
        if model_key is not None:
            model_dir = self._cache_dir / model_key
            if not model_dir.exists():
                return 0
            files = list(model_dir.glob(self._CACHE_GLOB))
            for f in files:
                f.unlink()
            return len(files)

        files = list(self._cache_dir.rglob(self._CACHE_GLOB))
        for f in files:
            f.unlink()
        return len(files)

    def stats(self) -> dict[str, int | float]:
        """Return cache statistics.

        Returns:
            Dict with hit_count, miss_count, hit_rate, and cached_count.
        """
        cached_count = len(list(self._cache_dir.rglob(self._CACHE_GLOB)))
        return {
            "hit_count": self._hits,
            "miss_count": self._misses,
            "hit_rate": self.hit_rate,
            "cached_count": cached_count,
        }

    def reset_stats(self) -> None:
        """Reset hit-and-miss counters to zero."""
        self._hits = 0
        self._misses = 0

    def _cache_path(self, text: str, model_key: str) -> Path:
        """Build the file path for a cached vector.

        Args:
            text: The text to build the path for.
            model_key: The model key to namespace the cache.

        Returns:
            Path to the cache file for this text and model.
        """
        content_hash = self._content_hash(text)
        return self._cache_dir / model_key / f"{content_hash}.json"

    @staticmethod
    def _content_hash(text: str) -> str:
        """Compute MD5 hash of text content.

        Args:
            text: Text to hash.

        Returns:
            Hex MD5 digest of the UTF-8 encoded text.
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()
