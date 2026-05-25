from __future__ import annotations

import pytest

from src.chunking.chunker import TextChunker
from src.shared.models import Chunk, ChunkMetadata, Document

# ---------------------------------------------------------------------------
# Minimal concrete implementation used only in this test module.
# It returns a single Chunk containing the full document text — just enough
# to satisfy the ABC contract without importing RecursiveTextChunker.
# ---------------------------------------------------------------------------


class _StubChunker(TextChunker):
    """Minimal valid implementation of TextChunker for contract testing."""

    def chunk(self, document: Document) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=f"{document.doc_id}__0",
                parent_doc_id=document.doc_id,
                text=document.text,
                chunk_index=0,
                total_chunks=1,
                metadata=document.metadata,
            )
        ]


class _EmptyChunker(TextChunker):
    """Raises ValueError as specified by the contract when no chunk is produced."""

    def chunk(self, document: Document) -> list[Chunk]:
        raise ValueError("no chunks produced")


def _make_document(**kwargs: object) -> Document:
    defaults = {"doc_id": "doc-001", "text": "Hello world"}
    defaults.update(kwargs)  # type: ignore[arg-type]
    return Document(**defaults)  # type: ignore[arg-type]


class TestTextChunkerIsAbstract:
    """TextChunker cannot be instantiated and enforces implementation of chunk()."""

    def test_direct_instantiation_raises(self) -> None:
        with pytest.raises(TypeError):
            TextChunker()  # type: ignore[abstract]

    def test_chunk_is_the_only_abstract_method(self) -> None:
        assert TextChunker.__abstractmethods__ == frozenset({"chunk"})

    def test_subclass_missing_chunk_raises_on_instantiation(self) -> None:
        class _Incomplete(TextChunker):
            pass

        with pytest.raises(TypeError, match="chunk"):
            _Incomplete()  # type: ignore[abstract]

    def test_subclass_with_chunk_implemented_can_be_instantiated(self) -> None:
        chunker = _StubChunker()
        assert isinstance(chunker, TextChunker)


class TestTextChunkerContract:
    """Concrete implementations must honour the chunk() return contract."""

    def test_chunk_returns_a_list(self) -> None:
        result = _StubChunker().chunk(_make_document())
        assert isinstance(result, list)

    def test_chunk_returns_chunk_instances(self) -> None:
        result = _StubChunker().chunk(_make_document())
        assert all(isinstance(c, Chunk) for c in result)

    def test_chunk_result_is_non_empty_for_valid_document(self) -> None:
        result = _StubChunker().chunk(_make_document())
        assert len(result) > 0

    def test_chunk_id_is_set_on_returned_chunks(self) -> None:
        result = _StubChunker().chunk(_make_document())
        assert all(c.chunk_id for c in result)

    def test_parent_doc_id_matches_document(self) -> None:
        doc = _make_document(doc_id="my-doc")
        result = _StubChunker().chunk(doc)
        assert all(c.parent_doc_id == "my-doc" for c in result)

    def test_chunk_index_starts_at_zero(self) -> None:
        result = _StubChunker().chunk(_make_document())
        assert result[0].chunk_index == 0

    def test_total_chunks_equals_list_length(self) -> None:
        result = _StubChunker().chunk(_make_document())
        assert all(c.total_chunks == len(result) for c in result)

    def test_metadata_propagated_from_document(self) -> None:
        meta = ChunkMetadata({"language": "fr"})
        doc = _make_document(metadata=meta)
        result = _StubChunker().chunk(doc)
        assert all(c.metadata.get("language") == "fr" for c in result)

    def test_raises_value_error_when_no_chunk_produced(self) -> None:
        """Contract specifies ValueError when the document yields no valid chunks."""
        with pytest.raises(ValueError):
            _EmptyChunker().chunk(_make_document())
