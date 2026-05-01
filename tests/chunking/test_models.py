# tests/chunking/test_models.py
from __future__ import annotations

import pytest
from src.chunking.models import Chunk, ChunkMetadata


class TestChunkMetadataCreation:
    """Tests for valid ChunkMetadata instantiation."""

    def test_default_metadata_is_empty_dict(self) -> None:
        """ChunkMetadata defaults to empty dict."""
        meta = ChunkMetadata()
        assert meta.metadata == {}

    def test_custom_metadata(self) -> None:
        """ChunkMetadata accepts arbitrary key-value pairs."""
        meta = ChunkMetadata(metadata={"language": "fr", "season": "hiver"})
        assert meta.metadata["language"] == "fr"
        assert meta.metadata["season"] == "hiver"

    def test_get_existing_key(self) -> None:
        """get() returns value for existing key."""
        meta = ChunkMetadata(metadata={"type": "restaurant"})
        assert meta.get("type") == "restaurant"

    def test_get_missing_key_returns_default(self) -> None:
        """get() returns None for missing key by default."""
        meta = ChunkMetadata()
        assert meta.get("missing_key") is None

    def test_get_missing_key_returns_custom_default(self) -> None:
        """get() returns custom default for missing key."""
        meta = ChunkMetadata()
        assert meta.get("missing_key", "fallback") == "fallback"

    def test_metadata_not_shared_between_instances(self) -> None:
        """Each ChunkMetadata instance has its own metadata dict."""
        meta1 = ChunkMetadata()
        meta2 = ChunkMetadata()
        meta1.metadata["key"] = "value"
        assert meta2.metadata == {}


class TestChunkCreation:
    """Tests for valid Chunk instantiation."""

    def _make_chunk(self, **kwargs) -> Chunk:  # type: ignore
        """Helper to create a valid Chunk with sensible defaults."""
        defaults = {
            "chunk_id": "chunk-001",
            "parent_doc_id": "doc-001",
            "text": "Hello world",
            "chunk_index": 0,
            "total_chunks": 3,
        }
        defaults.update(kwargs)
        return Chunk(**defaults)

    def test_minimal_valid_chunk(self) -> None:
        """Chunk creates successfully with required fields only."""
        chunk = self._make_chunk()
        assert chunk.chunk_id == "chunk-001"
        assert chunk.parent_doc_id == "doc-001"
        assert chunk.text == "Hello world"
        assert chunk.chunk_index == 0
        assert chunk.total_chunks == 3

    def test_default_embedding_is_empty_list(self) -> None:
        """Embedding defaults to empty list."""
        chunk = self._make_chunk()
        assert chunk.embedding == []

    def test_default_metadata_is_chunk_metadata(self) -> None:
        """Metadata defaults to empty ChunkMetadata instance."""
        chunk = self._make_chunk()
        assert isinstance(chunk.metadata, ChunkMetadata)
        assert chunk.metadata.metadata == {}

    def test_custom_embedding(self) -> None:
        """Chunk accepts a custom embedding vector."""
        embedding = [0.1, 0.2, 0.3]
        chunk = self._make_chunk(embedding=embedding)
        assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_custom_metadata(self) -> None:
        """Chunk accepts custom ChunkMetadata."""
        meta = ChunkMetadata(metadata={"language": "fr"})
        chunk = self._make_chunk(metadata=meta)
        assert chunk.metadata.get("language") == "fr"


class TestChunkValidation:
    """Tests for Chunk validation in __post_init__."""

    def _make_chunk(self, **kwargs) -> Chunk:  # type: ignore
        defaults = {
            "chunk_id": "chunk-001",
            "parent_doc_id": "doc-001",
            "text": "Hello world",
            "chunk_index": 0,
            "total_chunks": 3,
        }
        defaults.update(kwargs)
        return Chunk(**defaults)

    def test_empty_chunk_id_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_id cannot be empty"):
            self._make_chunk(chunk_id="")

    def test_empty_parent_doc_id_raises(self) -> None:
        with pytest.raises(ValueError, match="parent_doc_id cannot be empty"):
            self._make_chunk(parent_doc_id="")

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValueError, match="text cannot be empty"):
            self._make_chunk(text="")

    def test_negative_chunk_index_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_index must be >= 0"):
            self._make_chunk(chunk_index=-1)

    def test_zero_total_chunks_raises(self) -> None:
        with pytest.raises(ValueError, match="total_chunks must be >= 1"):
            self._make_chunk(total_chunks=0)

    def test_chunk_index_equals_total_chunks_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_index must be < total_chunks"):
            self._make_chunk(chunk_index=3, total_chunks=3)

    def test_chunk_index_greater_than_total_chunks_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_index must be < total_chunks"):
            self._make_chunk(chunk_index=5, total_chunks=3)


class TestChunkProperties:
    """Tests for Chunk computed properties."""

    def _make_chunk(self, **kwargs) -> Chunk:  # type: ignore
        defaults = {
            "chunk_id": "chunk-001",
            "parent_doc_id": "doc-001",
            "text": "Hello world",
            "chunk_index": 0,
            "total_chunks": 3,
        }
        defaults.update(kwargs)
        return Chunk(**defaults)

    def test_is_embedded_false_by_default(self) -> None:
        chunk = self._make_chunk()
        assert chunk.is_embedded is False

    def test_is_embedded_true_when_embedding_set(self) -> None:
        chunk = self._make_chunk(embedding=[0.1, 0.2, 0.3])
        assert chunk.is_embedded is True

    def test_word_count(self) -> None:
        chunk = self._make_chunk(text="Hello world foo")
        assert chunk.word_count == 3

    def test_char_count(self) -> None:
        chunk = self._make_chunk(text="Hello")
        assert chunk.char_count == 5

    def test_is_first_true_for_index_zero(self) -> None:
        chunk = self._make_chunk(chunk_index=0, total_chunks=3)
        assert chunk.is_first is True

    def test_is_first_false_for_non_zero_index(self) -> None:
        chunk = self._make_chunk(chunk_index=1, total_chunks=3)
        assert chunk.is_first is False

    def test_is_last_true_for_last_index(self) -> None:
        chunk = self._make_chunk(chunk_index=2, total_chunks=3)
        assert chunk.is_last is True

    def test_is_last_false_for_non_last_index(self) -> None:
        chunk = self._make_chunk(chunk_index=0, total_chunks=3)
        assert chunk.is_last is False

    def test_single_chunk_is_both_first_and_last(self) -> None:
        """A document with only one chunk should be both first and last."""
        chunk = self._make_chunk(chunk_index=0, total_chunks=1)
        assert chunk.is_first is True
        assert chunk.is_last is True