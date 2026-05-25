# tests/shared/test_models.py
from __future__ import annotations

from typing import Any

import pytest

from src.shared.models import Chunk, ChunkMetadata, Document, MetadataFilter


def _make_chunk(**kwargs: Any) -> Chunk:
    """Factory for Chunk with sensible defaults; override any field via kwargs."""
    defaults: dict[str, Any] = {
        "chunk_id": "chunk-001",
        "parent_doc_id": "doc-001",
        "text": "Hello world",
        "chunk_index": 0,
        "total_chunks": 3,
    }
    defaults.update(kwargs)
    return Chunk(**defaults)


class TestMetadataFilter:
    def test_field_and_value_stored(self) -> None:
        f = MetadataFilter(field="metadata.language", value="fr")
        assert f.field == "metadata.language"
        assert f.value == "fr"

    def test_empty_field_raises(self) -> None:
        with pytest.raises(ValueError, match="field cannot be empty"):
            MetadataFilter(field="", value="fr")

    def test_is_frozen(self) -> None:
        f = MetadataFilter(field="metadata.language", value="fr")
        with pytest.raises(AttributeError):
            setattr(f, "field", "other")

    def test_equality(self) -> None:
        assert MetadataFilter(field="metadata.language", value="fr") == MetadataFilter(
            field="metadata.language", value="fr"
        )

    def test_inequality_on_different_value(self) -> None:
        assert MetadataFilter(field="metadata.language", value="fr") != MetadataFilter(
            field="metadata.language", value="en"
        )


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

    def test_minimal_valid_chunk(self) -> None:
        """Chunk creates successfully with required fields only."""
        chunk = _make_chunk()
        assert chunk.chunk_id == "chunk-001"
        assert chunk.parent_doc_id == "doc-001"
        assert chunk.text == "Hello world"
        assert chunk.chunk_index == 0
        assert chunk.total_chunks == 3

    def test_default_embedding_is_empty_list(self) -> None:
        """Embedding defaults to empty list."""
        chunk = _make_chunk()
        assert chunk.embedding == []

    def test_default_metadata_is_chunk_metadata(self) -> None:
        """Metadata defaults to empty ChunkMetadata instance."""
        chunk = _make_chunk()
        assert isinstance(chunk.metadata, ChunkMetadata)
        assert chunk.metadata.metadata == {}

    def test_custom_embedding(self) -> None:
        """Chunk accepts a custom embedding vector."""
        embedding = [0.1, 0.2, 0.3]
        chunk = _make_chunk(embedding=embedding)
        assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_custom_metadata(self) -> None:
        """Chunk accepts custom ChunkMetadata."""
        meta = ChunkMetadata(metadata={"language": "fr"})
        chunk = _make_chunk(metadata=meta)
        assert chunk.metadata.get("language") == "fr"


class TestChunkValidation:
    """Tests for Chunk validation in __post_init__."""

    def test_empty_chunk_id_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_id cannot be empty"):
            _make_chunk(chunk_id="")

    def test_empty_parent_doc_id_raises(self) -> None:
        with pytest.raises(ValueError, match="parent_doc_id cannot be empty"):
            _make_chunk(parent_doc_id="")

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValueError, match="text cannot be empty"):
            _make_chunk(text="")

    def test_negative_chunk_index_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_index must be >= 0"):
            _make_chunk(chunk_index=-1)

    def test_zero_total_chunks_raises(self) -> None:
        with pytest.raises(ValueError, match="total_chunks must be >= 1"):
            _make_chunk(total_chunks=0)

    def test_chunk_index_equals_total_chunks_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_index must be < total_chunks"):
            _make_chunk(chunk_index=3, total_chunks=3)

    def test_chunk_index_greater_than_total_chunks_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_index must be < total_chunks"):
            _make_chunk(chunk_index=5, total_chunks=3)


class TestChunkProperties:
    """Tests for Chunk computed properties."""

    def test_is_embedded_false_by_default(self) -> None:
        chunk = _make_chunk()
        assert chunk.is_embedded is False

    def test_is_embedded_true_when_embedding_set(self) -> None:
        chunk = _make_chunk(embedding=[0.1, 0.2, 0.3])
        assert chunk.is_embedded is True

    def test_word_count(self) -> None:
        chunk = _make_chunk(text="Hello world foo")
        assert chunk.word_count == 3

    def test_char_count(self) -> None:
        chunk = _make_chunk(text="Hello")
        assert chunk.char_count == 5

    def test_is_first_true_for_index_zero(self) -> None:
        chunk = _make_chunk(chunk_index=0, total_chunks=3)
        assert chunk.is_first is True

    def test_is_first_false_for_non_zero_index(self) -> None:
        chunk = _make_chunk(chunk_index=1, total_chunks=3)
        assert chunk.is_first is False

    def test_is_last_true_for_last_index(self) -> None:
        chunk = _make_chunk(chunk_index=2, total_chunks=3)
        assert chunk.is_last is True

    def test_is_last_false_for_non_last_index(self) -> None:
        chunk = _make_chunk(chunk_index=0, total_chunks=3)
        assert chunk.is_last is False

    def test_single_chunk_is_both_first_and_last(self) -> None:
        """A document with only one chunk should be both first and last."""
        chunk = _make_chunk(chunk_index=0, total_chunks=1)
        assert chunk.is_first is True
        assert chunk.is_last is True


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def _make_document(**kwargs: Any) -> Document:
    """Factory for Document with sensible defaults; override any field via kwargs."""
    defaults: dict[str, Any] = {
        "doc_id": "doc-001",
        "text": "Hello world",
    }
    defaults.update(kwargs)
    return Document(**defaults)


class TestDocumentCreation:
    """Tests for valid Document instantiation."""

    def test_minimal_document(self) -> None:
        """Document creates successfully with only doc_id and text."""
        doc = _make_document()
        assert doc.doc_id == "doc-001"
        assert doc.text == "Hello world"

    def test_default_source_is_empty_string(self) -> None:
        doc = _make_document()
        assert doc.source == ""

    def test_default_metadata_is_empty_chunk_metadata(self) -> None:
        doc = _make_document()
        assert isinstance(doc.metadata, ChunkMetadata)
        assert doc.metadata.metadata == {}

    def test_custom_source(self) -> None:
        doc = _make_document(source="/data/notice.md")
        assert doc.source == "/data/notice.md"

    def test_custom_metadata(self) -> None:
        meta = ChunkMetadata(metadata={"language": "fr"})
        doc = _make_document(metadata=meta)
        assert doc.metadata.get("language") == "fr"

    def test_metadata_not_shared_between_instances(self) -> None:
        """Each Document instance has its own independent metadata dict."""
        doc1 = _make_document()
        doc2 = _make_document()
        doc1.metadata.metadata["key"] = "value"
        assert doc2.metadata.metadata == {}


class TestDocumentValidation:
    """Tests for Document validation in __post_init__."""

    def test_empty_doc_id_raises(self) -> None:
        with pytest.raises(ValueError, match="doc_id cannot be empty"):
            _make_document(doc_id="")

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValueError, match="text cannot be empty"):
            _make_document(text="")


class TestDocumentProperties:
    """Tests for Document computed properties."""

    def test_char_count(self) -> None:
        doc = _make_document(text="Hello")
        assert doc.char_count == 5

    def test_word_count(self) -> None:
        doc = _make_document(text="Hello world foo")
        assert doc.word_count == 3

    def test_char_count_empty_source_not_counted(self) -> None:
        """char_count reflects only text, not source or other fields."""
        doc = _make_document(text="Hi", source="some/very/long/path.txt")
        assert doc.char_count == 2

    def test_word_count_single_word(self) -> None:
        doc = _make_document(text="unique")
        assert doc.word_count == 1
