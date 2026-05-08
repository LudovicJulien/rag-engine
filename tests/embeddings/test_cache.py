# tests/embeddings/test_cache.py
from __future__ import annotations

from pathlib import Path

import pytest

from src.embeddings.embedding_cache import EmbeddingCache


@pytest.fixture
def cache(tmp_path: Path) -> EmbeddingCache:
    """Create a fresh EmbeddingCache in a temporary directory."""
    return EmbeddingCache(cache_dir=tmp_path / "test_cache")


@pytest.fixture
def populated_cache(cache: EmbeddingCache) -> EmbeddingCache:
    """Create a cache pre-populated with sample vectors."""
    cache.set(text="Le Plateau", model_key="model-a", vector=[0.1, 0.2, 0.3])
    cache.set(text="Rosemont", model_key="model-a", vector=[0.4, 0.5, 0.6])
    cache.set(text="Mile End", model_key="model-b", vector=[0.7, 0.8, 0.9])
    return cache


class TestEmbeddingCacheInit:
    """Tests for EmbeddingCache initialization."""

    def test_cache_dir_created_on_init(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()
        EmbeddingCache(cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=str(tmp_path / "str_cache"))
        assert isinstance(cache, EmbeddingCache)

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=tmp_path / "path_cache")
        assert isinstance(cache, EmbeddingCache)

    def test_initial_hit_count_is_zero(self, cache: EmbeddingCache) -> None:
        assert cache.hit_count == 0

    def test_initial_miss_count_is_zero(self, cache: EmbeddingCache) -> None:
        assert cache.miss_count == 0

    def test_initial_hit_rate_is_zero(self, cache: EmbeddingCache) -> None:
        assert cache.hit_rate == pytest.approx(0.0)


class TestEmbeddingCacheGet:
    """Tests for EmbeddingCache.get()."""

    def test_get_returns_none_on_miss(self, cache: EmbeddingCache) -> None:
        result = cache.get(text="unknown", model_key="model-a")
        assert result is None

    def test_get_returns_vector_on_hit(self, cache: EmbeddingCache) -> None:
        cache.set(text="Le Plateau", model_key="model-a", vector=[0.1, 0.2, 0.3])
        result = cache.get(text="Le Plateau", model_key="model-a")
        assert result == pytest.approx([0.1, 0.2, 0.3])

    def test_get_returns_list_of_floats(self, cache: EmbeddingCache) -> None:
        cache.set(text="Le Plateau", model_key="model-a", vector=[0.1, 0.2, 0.3])
        result = cache.get(text="Le Plateau", model_key="model-a")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_get_miss_increments_miss_count(self, cache: EmbeddingCache) -> None:
        cache.get(text="unknown", model_key="model-a")
        assert cache.miss_count == 1

    def test_get_hit_increments_hit_count(self, cache: EmbeddingCache) -> None:
        cache.set(text="Le Plateau", model_key="model-a", vector=[0.1, 0.2, 0.3])
        cache.get(text="Le Plateau", model_key="model-a")
        assert cache.hit_count == 1

    def test_get_different_model_key_is_miss(self, cache: EmbeddingCache) -> None:
        cache.set(text="Le Plateau", model_key="model-a", vector=[0.1, 0.2, 0.3])
        result = cache.get(text="Le Plateau", model_key="model-b")
        assert result is None

    def test_get_handles_corrupted_cache_file(
        self, cache: EmbeddingCache, tmp_path: Path
    ) -> None:
        """Corrupted cache file returns None
        treated as miss rather than raised as exception."""
        cache.set(text="Le Plateau", model_key="model-a", vector=[0.1, 0.2, 0.3])
        path = cache._cache_path("Le Plateau", "model-a")
        path.write_text("corrupted json {{{", encoding="utf-8")
        result = cache.get(text="Le Plateau", model_key="model-a")
        assert result is None
        assert cache.miss_count == 1


class TestEmbeddingCacheSet:
    """Tests for EmbeddingCache.set()."""

    def test_set_creates_cache_file(self, cache: EmbeddingCache) -> None:
        cache.set(text="Le Plateau", model_key="model-a", vector=[0.1, 0.2, 0.3])
        assert cache.exists(text="Le Plateau", model_key="model-a")

    def test_set_empty_vector_raises(self, cache: EmbeddingCache) -> None:
        with pytest.raises(ValueError, match="Cannot cache an empty vector"):
            cache.set(text="Le Plateau", model_key="model-a", vector=[])

    def test_set_overwrites_existing(self, cache: EmbeddingCache) -> None:
        cache.set(text="Le Plateau", model_key="model-a", vector=[0.1, 0.2, 0.3])
        cache.set(text="Le Plateau", model_key="model-a", vector=[0.9, 0.8, 0.7])
        result = cache.get(text="Le Plateau", model_key="model-a")
        assert result == pytest.approx([0.9, 0.8, 0.7])

    def test_set_different_models_independent(self, cache: EmbeddingCache) -> None:
        cache.set(text="Le Plateau", model_key="model-a", vector=[0.1, 0.2, 0.3])
        cache.set(text="Le Plateau", model_key="model-b", vector=[0.9, 0.8, 0.7])
        result_a = cache.get(text="Le Plateau", model_key="model-a")
        result_b = cache.get(text="Le Plateau", model_key="model-b")
        assert result_a == pytest.approx([0.1, 0.2, 0.3])
        assert result_b == pytest.approx([0.9, 0.8, 0.7])

    def test_set_creates_model_subdirectory(self, cache: EmbeddingCache) -> None:
        cache.set(text="Le Plateau", model_key="model-a", vector=[0.1, 0.2])
        path = cache._cache_path("Le Plateau", "model-a")
        assert path.parent.exists()
        assert path.parent.name == "model-a"


class TestEmbeddingCacheGetBatch:
    """Tests for EmbeddingCache.get_batch()."""

    def test_get_batch_all_hits(self, populated_cache: EmbeddingCache) -> None:
        results = populated_cache.get_batch(
            texts=["Le Plateau", "Rosemont"],
            model_key="model-a",
        )
        assert len(results) == 2
        assert results[0] == pytest.approx([0.1, 0.2, 0.3])
        assert results[1] == pytest.approx([0.4, 0.5, 0.6])

    def test_get_batch_all_misses(self, cache: EmbeddingCache) -> None:
        results = cache.get_batch(
            texts=["unknown-1", "unknown-2"],
            model_key="model-a",
        )
        assert results == [None, None]

    def test_get_batch_mixed_hits_and_misses(
        self, populated_cache: EmbeddingCache
    ) -> None:
        results = populated_cache.get_batch(
            texts=["Le Plateau", "unknown"],
            model_key="model-a",
        )
        assert results[0] == pytest.approx([0.1, 0.2, 0.3])
        assert results[1] is None

    def test_get_batch_preserves_order(self, populated_cache: EmbeddingCache) -> None:
        """Order of returned vectors must match order of input texts."""
        results = populated_cache.get_batch(
            texts=["Rosemont", "Le Plateau"],
            model_key="model-a",
        )
        assert results[0] == pytest.approx([0.4, 0.5, 0.6])
        assert results[1] == pytest.approx([0.1, 0.2, 0.3])

    def test_get_batch_empty_list(self, cache: EmbeddingCache) -> None:
        results = cache.get_batch(texts=[], model_key="model-a")
        assert results == []


class TestEmbeddingCacheSetBatch:
    """Tests for EmbeddingCache.set_batch()."""

    def test_set_batch_stores_all_vectors(self, cache: EmbeddingCache) -> None:
        texts = ["Le Plateau", "Rosemont", "Mile End"]
        vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        cache.set_batch(texts=texts, model_key="model-a", vectors=vectors)
        for text, vector in zip(texts, vectors):
            result = cache.get(text=text, model_key="model-a")
            assert result == pytest.approx(vector)

    def test_set_batch_mismatched_lengths_raises(self, cache: EmbeddingCache) -> None:
        with pytest.raises(ValueError, match="same length"):
            cache.set_batch(
                texts=["a", "b"],
                model_key="model-a",
                vectors=[[0.1, 0.2]],
            )

    def test_set_batch_empty_lists(self, cache: EmbeddingCache) -> None:
        """Setting empty batch is a no-op."""
        cache.set_batch(texts=[], model_key="model-a", vectors=[])
        assert cache.stats()["cached_count"] == 0


class TestEmbeddingCacheExists:
    """Tests for EmbeddingCache.exists()."""

    def test_exists_returns_false_before_set(self, cache: EmbeddingCache) -> None:
        assert cache.exists(text="Le Plateau", model_key="model-a") is False

    def test_exists_returns_true_after_set(self, cache: EmbeddingCache) -> None:
        cache.set(text="Le Plateau", model_key="model-a", vector=[0.1, 0.2])
        assert cache.exists(text="Le Plateau", model_key="model-a") is True

    def test_exists_different_model_key_returns_false(
        self, cache: EmbeddingCache
    ) -> None:
        cache.set(text="Le Plateau", model_key="model-a", vector=[0.1, 0.2])
        assert cache.exists(text="Le Plateau", model_key="model-b") is False


class TestEmbeddingCacheInvalidate:
    """Tests for EmbeddingCache.invalidate()."""

    def test_invalidate_existing_entry_returns_true(
        self, cache: EmbeddingCache
    ) -> None:
        cache.set(text="Le Plateau", model_key="model-a", vector=[0.1, 0.2])
        result = cache.invalidate(text="Le Plateau", model_key="model-a")
        assert result is True

    def test_invalidate_nonexistent_entry_returns_false(
        self, cache: EmbeddingCache
    ) -> None:
        result = cache.invalidate(text="unknown", model_key="model-a")
        assert result is False

    def test_invalidate_removes_entry(self, cache: EmbeddingCache) -> None:
        cache.set(text="Le Plateau", model_key="model-a", vector=[0.1, 0.2])
        cache.invalidate(text="Le Plateau", model_key="model-a")
        assert cache.exists(text="Le Plateau", model_key="model-a") is False

    def test_invalidate_does_not_affect_other_entries(
        self, populated_cache: EmbeddingCache
    ) -> None:
        populated_cache.invalidate(text="Le Plateau", model_key="model-a")
        assert populated_cache.exists(text="Rosemont", model_key="model-a") is True


class TestEmbeddingCacheClear:
    """Tests for EmbeddingCache.clear()."""

    def test_clear_all_removes_all_entries(
        self, populated_cache: EmbeddingCache
    ) -> None:
        count = populated_cache.clear()
        assert count == 3
        assert populated_cache.stats()["cached_count"] == 0

    def test_clear_by_model_key_removes_only_that_model(
        self, populated_cache: EmbeddingCache
    ) -> None:
        count = populated_cache.clear(model_key="model-a")
        assert count == 2
        assert populated_cache.exists(text="Mile End", model_key="model-b") is True

    def test_clear_nonexistent_model_key_returns_zero(
        self, cache: EmbeddingCache
    ) -> None:
        count = cache.clear(model_key="nonexistent")
        assert count == 0

    def test_clear_empty_cache_returns_zero(self, cache: EmbeddingCache) -> None:
        count = cache.clear()
        assert count == 0


class TestEmbeddingCacheStats:
    """Tests for EmbeddingCache.stats() and hit_rate."""

    def test_hit_rate_zero_with_no_lookups(self, cache: EmbeddingCache) -> None:
        assert cache.hit_rate == pytest.approx(0.0)

    def test_hit_rate_one_with_all_hits(self, cache: EmbeddingCache) -> None:
        cache.set(text="Le Plateau", model_key="model-a", vector=[0.1, 0.2])
        cache.get(text="Le Plateau", model_key="model-a")
        cache.get(text="Le Plateau", model_key="model-a")
        assert cache.hit_rate == pytest.approx(1.0)

    def test_hit_rate_zero_with_all_misses(self, cache: EmbeddingCache) -> None:
        cache.get(text="unknown-1", model_key="model-a")
        cache.get(text="unknown-2", model_key="model-a")
        assert cache.hit_rate == pytest.approx(0.0)

    def test_hit_rate_partial(self, cache: EmbeddingCache) -> None:
        cache.set(text="Le Plateau", model_key="model-a", vector=[0.1, 0.2])
        cache.get(text="Le Plateau", model_key="model-a")  # hit
        cache.get(text="unknown", model_key="model-a")  # miss
        assert cache.hit_rate == pytest.approx(0.5)

    def test_stats_returns_correct_structure(self, cache: EmbeddingCache) -> None:
        stats = cache.stats()
        assert "hit_count" in stats
        assert "miss_count" in stats
        assert "hit_rate" in stats
        assert "cached_count" in stats

    def test_stats_cached_count(self, populated_cache: EmbeddingCache) -> None:
        stats = populated_cache.stats()
        assert stats["cached_count"] == 3

    def test_reset_stats_clears_counters(self, cache: EmbeddingCache) -> None:
        cache.set(text="Le Plateau", model_key="model-a", vector=[0.1, 0.2])
        cache.get(text="Le Plateau", model_key="model-a")
        cache.get(text="unknown", model_key="model-a")
        cache.reset_stats()
        assert cache.hit_count == 0
        assert cache.miss_count == 0
        assert cache.hit_rate == pytest.approx(0.0)


class TestEmbeddingCacheContentHash:
    """Tests for EmbeddingCache._content_hash()."""

    def test_hash_is_string(self) -> None:
        assert isinstance(EmbeddingCache._content_hash("hello"), str)

    def test_hash_is_deterministic(self) -> None:
        assert EmbeddingCache._content_hash("hello") == EmbeddingCache._content_hash(
            "hello"
        )

    def test_different_texts_different_hashes(self) -> None:
        assert EmbeddingCache._content_hash("hello") != EmbeddingCache._content_hash(
            "world"
        )

    def test_hash_length_is_32(self) -> None:
        """MD5 hex digest is always 32 characters."""
        assert len(EmbeddingCache._content_hash("hello")) == 32

    def test_hash_is_lowercase_hex(self) -> None:
        result = EmbeddingCache._content_hash("hello")
        assert all(c in "0123456789abcdef" for c in result)
