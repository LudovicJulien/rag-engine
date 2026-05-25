from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.chunking.chunker import TextChunker
from src.ingestion.ingestion_pipeline import DataIngestionPipeline
from src.ingestion.text_pipeline import TextIngestionPipeline
from src.shared.models import Chunk, ChunkMetadata, Document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str, text: str, parent_doc_id: str = "doc-001") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        parent_doc_id=parent_doc_id,
        text=text,
        chunk_index=0,
        total_chunks=1,
        metadata=ChunkMetadata(),
    )


class _StubChunker(TextChunker):
    """Returns a pre-configured list regardless of the Document received."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def chunk(self, document: Document) -> list[Chunk]:
        return self._chunks


class _CapturingChunker(TextChunker):
    """Captures the Document it receives; returns one chunk derived from it."""

    def __init__(self) -> None:
        self.received: list[Document] = []

    def chunk(self, document: Document) -> list[Chunk]:
        self.received.append(document)
        return [_make_chunk("captured", document.text, document.doc_id)]


# ---------------------------------------------------------------------------
# ABC compliance
# ---------------------------------------------------------------------------


class TestTextIngestionPipelineABC:
    def test_is_data_ingestion_pipeline(self) -> None:
        assert isinstance(TextIngestionPipeline(), DataIngestionPipeline)

    def test_has_ingest_method(self) -> None:
        assert callable(getattr(TextIngestionPipeline, "ingest", None))

    def test_has_validate_method(self) -> None:
        assert callable(getattr(TextIngestionPipeline, "validate", None))

    def test_has_ingest_and_validate_method(self) -> None:
        assert callable(getattr(TextIngestionPipeline, "ingest_and_validate", None))


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestTextIngestionPipelineInit:
    def test_default_chunker_is_text_chunker(self) -> None:
        pipeline = TextIngestionPipeline()
        assert isinstance(pipeline._chunker, TextChunker)

    def test_custom_chunker_stored(self) -> None:
        stub = _StubChunker([_make_chunk("c1", "hello")])
        pipeline = TextIngestionPipeline(chunker=stub)
        assert pipeline._chunker is stub

    def test_none_chunker_falls_back_to_default(self) -> None:
        pipeline = TextIngestionPipeline(chunker=None)
        assert isinstance(pipeline._chunker, TextChunker)


# ---------------------------------------------------------------------------
# ingest() — happy paths
# ---------------------------------------------------------------------------


class TestTextIngestionPipelineIngest:
    def test_ingest_txt_file_returns_chunks(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text(
            "This is a valid text file with sufficient content.", encoding="utf-8"
        )
        pipeline = TextIngestionPipeline()
        chunks = pipeline.ingest(path)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_ingest_md_file_returns_chunks(self, tmp_path: Path) -> None:
        path = tmp_path / "guide.md"
        path.write_text(
            "# Title\n\nThis section has enough text content to be valid.",
            encoding="utf-8",
        )
        pipeline = TextIngestionPipeline()
        chunks = pipeline.ingest(path)
        assert len(chunks) >= 1

    def test_ingest_accepts_string_path(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text(
            "Content long enough to satisfy the min chunk size constraint.",
            encoding="utf-8",
        )
        pipeline = TextIngestionPipeline()
        chunks = pipeline.ingest(str(path))
        assert len(chunks) >= 1

    def test_ingest_returns_list(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.txt"
        path.write_text("word " * 20, encoding="utf-8")
        pipeline = TextIngestionPipeline()
        result = pipeline.ingest(path)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# ingest() — error paths
# ---------------------------------------------------------------------------


class TestTextIngestionPipelineIngestErrors:
    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Source file not found"):
            TextIngestionPipeline().ingest("nonexistent.txt")

    def test_unsupported_extension_csv_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        path.write_text("col1,col2", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file type"):
            TextIngestionPipeline().ingest(path)

    def test_unsupported_extension_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "data.json"
        path.write_text('{"key": "val"}', encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file type"):
            TextIngestionPipeline().ingest(path)

    def test_unsupported_extension_pdf_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF-1.4")
        with pytest.raises(ValueError, match="Unsupported file type"):
            TextIngestionPipeline().ingest(path)

    def test_error_message_lists_allowed_extensions(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        path.write_text("col1", encoding="utf-8")
        with pytest.raises(ValueError, match=r"\.(md|txt)"):
            TextIngestionPipeline().ingest(path)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="File is empty"):
            TextIngestionPipeline().ingest(path)

    def test_whitespace_only_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "blank.md"
        path.write_text("   \n\t  ", encoding="utf-8")
        with pytest.raises(ValueError, match="File is empty"):
            TextIngestionPipeline().ingest(path)


# ---------------------------------------------------------------------------
# Document construction — doc_id
# ---------------------------------------------------------------------------


class TestTextIngestionPipelineDocId:
    def test_doc_id_is_md5_hex_of_filename(self, tmp_path: Path) -> None:
        path = tmp_path / "guide.md"
        path.write_text("Some content", encoding="utf-8")
        expected = hashlib.md5("guide.md".encode()).hexdigest()[:16]
        cap = _CapturingChunker()
        TextIngestionPipeline(chunker=cap).ingest(path)
        assert cap.received[0].doc_id == expected

    def test_doc_id_is_sixteen_hex_chars(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("Content", encoding="utf-8")
        cap = _CapturingChunker()
        TextIngestionPipeline(chunker=cap).ingest(path)
        assert len(cap.received[0].doc_id) == 16

    def test_doc_id_depends_on_name_not_content(self, tmp_path: Path) -> None:
        path_a = tmp_path / "file.txt"
        path_b = tmp_path / "other" / "file.txt"
        path_b.parent.mkdir()
        path_a.write_text("Alpha content", encoding="utf-8")
        path_b.write_text("Beta content — completely different", encoding="utf-8")
        cap_a, cap_b = _CapturingChunker(), _CapturingChunker()
        TextIngestionPipeline(chunker=cap_a).ingest(path_a)
        TextIngestionPipeline(chunker=cap_b).ingest(path_b)
        assert cap_a.received[0].doc_id == cap_b.received[0].doc_id

    def test_different_filenames_give_different_doc_ids(self, tmp_path: Path) -> None:
        for name in ("alpha.txt", "beta.txt"):
            (tmp_path / name).write_text("Same content", encoding="utf-8")
        cap_a, cap_b = _CapturingChunker(), _CapturingChunker()
        TextIngestionPipeline(chunker=cap_a).ingest(tmp_path / "alpha.txt")
        TextIngestionPipeline(chunker=cap_b).ingest(tmp_path / "beta.txt")
        assert cap_a.received[0].doc_id != cap_b.received[0].doc_id


# ---------------------------------------------------------------------------
# Document construction — metadata & source
# ---------------------------------------------------------------------------


class TestTextIngestionPipelineMetadata:
    def test_document_source_is_string_path(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("Content", encoding="utf-8")
        cap = _CapturingChunker()
        TextIngestionPipeline(chunker=cap).ingest(path)
        assert cap.received[0].source == str(path)

    def test_metadata_source_is_string_path(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("Content", encoding="utf-8")
        cap = _CapturingChunker()
        TextIngestionPipeline(chunker=cap).ingest(path)
        assert cap.received[0].metadata.get("source") == str(path)

    def test_metadata_filename_is_basename(self, tmp_path: Path) -> None:
        path = tmp_path / "report.md"
        path.write_text("# Report\n\nBody.", encoding="utf-8")
        cap = _CapturingChunker()
        TextIngestionPipeline(chunker=cap).ingest(path)
        assert cap.received[0].metadata.get("filename") == "report.md"

    def test_metadata_filename_matches_path_name(self, tmp_path: Path) -> None:
        path = tmp_path / "my_doc.txt"
        path.write_text("Some text", encoding="utf-8")
        cap = _CapturingChunker()
        TextIngestionPipeline(chunker=cap).ingest(path)
        assert cap.received[0].metadata.get("filename") == path.name


# ---------------------------------------------------------------------------
# Chunker delegation
# ---------------------------------------------------------------------------


class TestTextIngestionPipelineChunkDelegation:
    def test_custom_chunker_output_is_returned(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.txt"
        path.write_text("Some content here", encoding="utf-8")
        expected = [_make_chunk("c1", "Chunk A"), _make_chunk("c2", "Chunk B")]
        stub = _StubChunker(expected)
        result = TextIngestionPipeline(chunker=stub).ingest(path)
        assert result == expected

    def test_chunker_receives_stripped_text(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.txt"
        path.write_text("  hello world  \n", encoding="utf-8")
        cap = _CapturingChunker()
        TextIngestionPipeline(chunker=cap).ingest(path)
        assert cap.received[0].text == "hello world"

    def test_chunker_called_once_per_ingest(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.txt"
        path.write_text("Content", encoding="utf-8")
        cap = _CapturingChunker()
        TextIngestionPipeline(chunker=cap).ingest(path)
        assert len(cap.received) == 1


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


class TestTextIngestionPipelineValidate:
    def _chunk(self, chunk_id: str, text: str) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            parent_doc_id="doc-001",
            text=text,
            chunk_index=0,
            total_chunks=1,
            metadata=ChunkMetadata(),
        )

    def test_unique_chunks_unchanged(self) -> None:
        chunks = [
            self._chunk("c1", "Alpha"),
            self._chunk("c2", "Beta"),
            self._chunk("c3", "Gamma"),
        ]
        result = TextIngestionPipeline().validate(chunks)
        assert len(result) == 3

    def test_duplicate_text_removed(self) -> None:
        chunks = [
            self._chunk("c1", "Same text"),
            self._chunk("c2", "Same text"),
            self._chunk("c3", "Different text"),
        ]
        result = TextIngestionPipeline().validate(chunks)
        assert len(result) == 2

    def test_first_occurrence_kept_on_duplicate(self) -> None:
        chunks = [
            self._chunk("c1", "Same text"),
            self._chunk("c2", "Same text"),
        ]
        result = TextIngestionPipeline().validate(chunks)
        assert result[0].chunk_id == "c1"

    def test_order_preserved(self) -> None:
        chunks = [
            self._chunk("c1", "Alpha"),
            self._chunk("c2", "Beta"),
            self._chunk("c3", "Gamma"),
        ]
        result = TextIngestionPipeline().validate(chunks)
        assert [c.chunk_id for c in result] == ["c1", "c2", "c3"]

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="No valid chunks found"):
            TextIngestionPipeline().validate([])

    def test_all_duplicates_leaves_one(self) -> None:
        chunks = [self._chunk(f"c{i}", "Duplicate") for i in range(4)]
        result = TextIngestionPipeline().validate(chunks)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# ingest_and_validate() — inherited from DataIngestionPipeline
# ---------------------------------------------------------------------------


class TestTextIngestionPipelineIngestAndValidate:
    def test_returns_chunks(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text(
            "This is enough content to satisfy the min chunk size requirement.",
            encoding="utf-8",
        )
        chunks = TextIngestionPipeline().ingest_and_validate(path)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_deduplication_is_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / "dup.txt"
        path.write_text("Content", encoding="utf-8")
        dup = _make_chunk("c1", "Content")
        # ingest() deduplicates to 1 chunk; ingest_and_validate() re-validates (no-op)
        stub = _StubChunker([dup, dup])
        result = TextIngestionPipeline(chunker=stub).ingest_and_validate(path)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Integration — end-to-end with real RecursiveTextChunker
# ---------------------------------------------------------------------------


class TestTextIngestionPipelineIntegration:
    def test_long_text_produces_multiple_chunks(self, tmp_path: Path) -> None:
        path = tmp_path / "long.txt"
        path.write_text("word " * 200, encoding="utf-8")
        chunks = TextIngestionPipeline().ingest(path)
        assert len(chunks) > 1

    def test_short_text_produces_one_chunk(self, tmp_path: Path) -> None:
        content = "This is a short but valid piece of text for testing."
        path = tmp_path / "short.txt"
        path.write_text(content, encoding="utf-8")
        chunks = TextIngestionPipeline().ingest(path)
        assert len(chunks) == 1
        assert chunks[0].text == content

    def test_parent_doc_id_matches_filename_hash(self, tmp_path: Path) -> None:
        path = tmp_path / "guide.txt"
        path.write_text("Some text content here for the doc.", encoding="utf-8")
        expected = hashlib.md5("guide.txt".encode()).hexdigest()[:16]
        chunks = TextIngestionPipeline().ingest(path)
        assert all(c.parent_doc_id == expected for c in chunks)

    def test_chunks_are_chunk_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.md"
        path.write_text(
            "# Section\n\nThis content section has enough text to be chunked properly.",
            encoding="utf-8",
        )
        chunks = TextIngestionPipeline().ingest(path)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_md_and_txt_produce_same_chunks_for_same_content(
        self, tmp_path: Path
    ) -> None:
        content = "Hello world content for testing purposes."
        txt_path = tmp_path / "doc.txt"
        md_path = tmp_path / "doc.md"
        txt_path.write_text(content, encoding="utf-8")
        md_path.write_text(content, encoding="utf-8")
        txt_chunks = TextIngestionPipeline().ingest(txt_path)
        md_chunks = TextIngestionPipeline().ingest(md_path)
        assert [c.text for c in txt_chunks] == [c.text for c in md_chunks]
