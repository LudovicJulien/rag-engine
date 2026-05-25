from __future__ import annotations

import re

import pytest

from src.chunking.chunker import ChunkConfig
from src.chunking.recursive import RecursiveTextChunker
from src.shared.models import Chunk, ChunkMetadata, Document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LONG_TEXT = (
    "The first section covers introduction topics and context.\n\n"
    "The second section explains the core algorithm in detail.\n\n"
    "The third section discusses performance and trade-offs.\n\n"
    "The fourth section provides practical examples and usage.\n\n"
    "The fifth section concludes with next steps and references."
)


def _doc(
    text: str = "Hello world", doc_id: str = "doc-001", **kwargs: object
) -> Document:
    return Document(doc_id=doc_id, text=text, **kwargs)  # type: ignore[arg-type]


def _chunker(
    chunk_size: int = 60,
    chunk_overlap: int = 0,
    min_chunk_size: int = 1,
) -> RecursiveTextChunker:
    return RecursiveTextChunker(
        ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )
    )


# ---------------------------------------------------------------------------
# Basic return type and count
# ---------------------------------------------------------------------------


class TestChunkBasic:
    def test_returns_list(self) -> None:
        result = _chunker().chunk(_doc())
        assert isinstance(result, list)

    def test_returns_chunk_instances(self) -> None:
        result = _chunker().chunk(_doc())
        assert all(isinstance(c, Chunk) for c in result)

    def test_short_text_produces_single_chunk(self) -> None:
        result = _chunker(chunk_size=200).chunk(_doc("Short text."))
        assert len(result) == 1

    def test_long_text_produces_multiple_chunks(self) -> None:
        result = _chunker(chunk_size=60).chunk(_doc(_LONG_TEXT))
        assert len(result) > 1

    def test_result_is_never_empty_for_valid_document(self) -> None:
        result = _chunker().chunk(_doc("Some content."))
        assert len(result) >= 1

    def test_single_chunk_text_matches_document_text(self) -> None:
        result = _chunker(chunk_size=500).chunk(_doc("Exact content here."))
        assert result[0].text == "Exact content here."


# ---------------------------------------------------------------------------
# Chunk fields — index, total, parent
# ---------------------------------------------------------------------------


class TestChunkFields:
    def test_chunk_index_starts_at_zero(self) -> None:
        result = _chunker().chunk(_doc(_LONG_TEXT))
        assert result[0].chunk_index == 0

    def test_chunk_index_is_contiguous(self) -> None:
        result = _chunker().chunk(_doc(_LONG_TEXT))
        assert [c.chunk_index for c in result] == list(range(len(result)))

    def test_total_chunks_equals_list_length(self) -> None:
        result = _chunker().chunk(_doc(_LONG_TEXT))
        assert all(c.total_chunks == len(result) for c in result)

    def test_total_chunks_consistent_across_all_chunks(self) -> None:
        result = _chunker().chunk(_doc(_LONG_TEXT))
        totals = {c.total_chunks for c in result}
        assert len(totals) == 1

    def test_parent_doc_id_matches_document(self) -> None:
        result = _chunker().chunk(_doc(_LONG_TEXT, doc_id="my-doc-42"))
        assert all(c.parent_doc_id == "my-doc-42" for c in result)

    def test_first_chunk_is_first(self) -> None:
        result = _chunker().chunk(_doc(_LONG_TEXT))
        assert result[0].is_first is True

    def test_last_chunk_is_last(self) -> None:
        result = _chunker().chunk(_doc(_LONG_TEXT))
        assert result[-1].is_last is True

    def test_intermediate_chunks_are_neither_first_nor_last(self) -> None:
        result = _chunker().chunk(_doc(_LONG_TEXT))
        for c in result[1:-1]:
            assert c.is_first is False
            assert c.is_last is False

    def test_single_chunk_is_both_first_and_last(self) -> None:
        result = _chunker(chunk_size=500).chunk(_doc("One short sentence."))
        assert result[0].is_first is True
        assert result[0].is_last is True

    def test_embedding_is_empty_by_default(self) -> None:
        result = _chunker().chunk(_doc("Some content."))
        assert all(c.is_embedded is False for c in result)


# ---------------------------------------------------------------------------
# Chunk ID — determinism and format
# ---------------------------------------------------------------------------


class TestChunkId:
    def test_chunk_id_is_non_empty(self) -> None:
        result = _chunker().chunk(_doc(_LONG_TEXT))
        assert all(c.chunk_id for c in result)

    def test_chunk_ids_are_unique_within_document(self) -> None:
        result = _chunker().chunk(_doc(_LONG_TEXT))
        ids = [c.chunk_id for c in result]
        assert len(ids) == len(set(ids))

    def test_chunk_id_contains_doc_id(self) -> None:
        result = _chunker().chunk(_doc("Some content.", doc_id="proj-007"))
        assert all("proj-007" in c.chunk_id for c in result)

    def test_chunk_id_contains_zero_padded_index(self) -> None:
        result = _chunker().chunk(_doc(_LONG_TEXT))
        for c in result:
            expected_index = f"{c.chunk_index:04d}"
            assert expected_index in c.chunk_id

    def test_chunk_id_matches_expected_format(self) -> None:
        # Format: {doc_id}__chunk_{index:04d}_{12-char hex}
        result = _chunker().chunk(_doc("Content.", doc_id="d1"))
        pattern = re.compile(r"^d1__chunk_\d{4}_[0-9a-f]{12}$")
        assert all(pattern.match(c.chunk_id) for c in result)

    def test_chunk_id_is_deterministic(self) -> None:
        doc = _doc(_LONG_TEXT, doc_id="stable")
        ids_a = [c.chunk_id for c in _chunker().chunk(doc)]
        ids_b = [c.chunk_id for c in _chunker().chunk(doc)]
        assert ids_a == ids_b

    def test_different_doc_ids_produce_different_chunk_ids(self) -> None:
        text = "Same content."
        ids_a = {
            c.chunk_id for c in _chunker(chunk_size=500).chunk(_doc(text, doc_id="A"))
        }
        ids_b = {
            c.chunk_id for c in _chunker(chunk_size=500).chunk(_doc(text, doc_id="B"))
        }
        assert ids_a.isdisjoint(ids_b)


# ---------------------------------------------------------------------------
# Metadata propagation
# ---------------------------------------------------------------------------


class TestChunkMetadataPropagation:
    def test_metadata_propagated_to_all_chunks(self) -> None:
        meta = ChunkMetadata({"language": "fr", "source": "wiki"})
        doc = _doc(_LONG_TEXT, metadata=meta)
        result = _chunker().chunk(doc)
        assert all(c.metadata.get("language") == "fr" for c in result)
        assert all(c.metadata.get("source") == "wiki" for c in result)

    def test_empty_metadata_propagated(self) -> None:
        result = _chunker().chunk(_doc("Content."))
        assert all(isinstance(c.metadata, ChunkMetadata) for c in result)

    def test_all_chunks_share_same_metadata_object(self) -> None:
        meta = ChunkMetadata({"key": "value"})
        doc = _doc(_LONG_TEXT, metadata=meta)
        result = _chunker().chunk(doc)
        assert all(c.metadata is meta for c in result)


# ---------------------------------------------------------------------------
# Strip and min_chunk_size filter
# ---------------------------------------------------------------------------


class TestChunkStripAndFilter:
    def test_chunk_text_has_no_leading_whitespace(self) -> None:
        result = _chunker().chunk(_doc(_LONG_TEXT))
        assert all(c.text == c.text.lstrip() for c in result)

    def test_chunk_text_has_no_trailing_whitespace(self) -> None:
        result = _chunker().chunk(_doc(_LONG_TEXT))
        assert all(c.text == c.text.rstrip() for c in result)

    def test_min_chunk_size_filters_short_fragments(self) -> None:
        # "a" is a 1-char paragraph that should be filtered with min=5.
        text = "a\n\n" + "A longer paragraph that passes the threshold easily."
        chunker = _chunker(chunk_size=60, min_chunk_size=5)
        result = chunker.chunk(_doc(text))
        assert all(len(c.text) >= 5 for c in result)

    def test_raises_if_all_fragments_below_min_chunk_size(self) -> None:
        chunker = RecursiveTextChunker(
            ChunkConfig(chunk_size=512, chunk_overlap=0, min_chunk_size=9999)
        )
        with pytest.raises(ValueError, match="no chunks"):
            chunker.chunk(_doc("Short."))

    def test_error_message_includes_doc_id(self) -> None:
        chunker = RecursiveTextChunker(
            ChunkConfig(chunk_size=512, chunk_overlap=0, min_chunk_size=9999)
        )
        with pytest.raises(ValueError, match="target-doc"):
            chunker.chunk(_doc("Short.", doc_id="target-doc"))

    def test_total_chunks_reflects_post_filter_count(self) -> None:
        # Force filtering: min_chunk_size removes some fragments.
        text = "tiny\n\n" + "A much longer paragraph that is kept." * 3
        chunker = _chunker(chunk_size=60, min_chunk_size=10)
        result = chunker.chunk(_doc(text))
        assert all(c.total_chunks == len(result) for c in result)


# ---------------------------------------------------------------------------
# Overlap end-to-end in chunk()
# ---------------------------------------------------------------------------


class TestChunkOverlap:
    def test_second_chunk_starts_with_tail_of_first(self) -> None:
        chunker = _chunker(chunk_size=60, chunk_overlap=10)
        result = chunker.chunk(_doc(_LONG_TEXT))
        if len(result) >= 2:
            # Strip both sides before comparing — overlap suffix is already stripped
            # because it came from a stripped chunk.
            tail = result[0].text[-10:]
            assert result[1].text.startswith(tail)

    def test_no_overlap_chunks_do_not_share_prefix(self) -> None:
        chunker = _chunker(chunk_size=60, chunk_overlap=0)
        result = chunker.chunk(_doc(_LONG_TEXT))
        if len(result) >= 2:
            tail = result[0].text[-5:]
            assert not result[1].text.startswith(tail)

    def test_overlap_does_not_shrink_chunks(self) -> None:
        no_overlap = _chunker(chunk_size=60, chunk_overlap=0).chunk(_doc(_LONG_TEXT))
        with_overlap = _chunker(chunk_size=60, chunk_overlap=10).chunk(_doc(_LONG_TEXT))
        # Every chunk with overlap is at least as long as without.
        for i, c in enumerate(no_overlap):
            if i < len(with_overlap):
                assert len(with_overlap[i].text) >= len(c.text) - 1


# ---------------------------------------------------------------------------
# _make_chunk_id static method
# ---------------------------------------------------------------------------


class TestMakeChunkId:
    def test_returns_string(self) -> None:
        cid = RecursiveTextChunker._make_chunk_id("doc", 0, "text")
        assert isinstance(cid, str)

    def test_deterministic(self) -> None:
        a = RecursiveTextChunker._make_chunk_id("doc", 3, "hello")
        b = RecursiveTextChunker._make_chunk_id("doc", 3, "hello")
        assert a == b

    def test_different_index_gives_different_id(self) -> None:
        a = RecursiveTextChunker._make_chunk_id("doc", 0, "hello")
        b = RecursiveTextChunker._make_chunk_id("doc", 1, "hello")
        assert a != b

    def test_different_text_gives_different_id(self) -> None:
        a = RecursiveTextChunker._make_chunk_id("doc", 0, "hello")
        b = RecursiveTextChunker._make_chunk_id("doc", 0, "world")
        assert a != b

    def test_different_doc_id_gives_different_id(self) -> None:
        a = RecursiveTextChunker._make_chunk_id("doc-A", 0, "text")
        b = RecursiveTextChunker._make_chunk_id("doc-B", 0, "text")
        assert a != b

    def test_index_zero_padded_to_four_digits(self) -> None:
        cid = RecursiveTextChunker._make_chunk_id("d", 7, "t")
        assert "0007" in cid

    def test_large_index_not_truncated(self) -> None:
        cid = RecursiveTextChunker._make_chunk_id("d", 12345, "t")
        assert "12345" in cid
