# tests/ingestion/test_json_pipeline.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingestion.json_pipeline import JSONChunkIngestionPipeline
from src.shared.models import Chunk


class TestJSONChunkIngestionPipelineIngest:
    """Tests for JSONChunkIngestionPipeline.ingest()"""

    def test_ingest_valid_json(self, tmp_path: Path) -> None:
        """Ingest a valid JSON file returns list of Chunk objects."""
        data = [
            {
                "chunk_id": "chunk-001",
                "parent_doc_id": "doc-001",
                "text": "Hello world",
                "chunk_index": 0,
                "total_chunks": 1,
            }
        ]
        path = tmp_path / "chunks.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        pipeline = JSONChunkIngestionPipeline()
        chunks = pipeline.ingest(path)

        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].chunk_id == "chunk-001"

    def test_ingest_accepts_string_path(self, tmp_path: Path) -> None:
        """ingest() accepts a string path as well as Path object."""
        data = [
            {
                "chunk_id": "chunk-001",
                "parent_doc_id": "doc-001",
                "text": "Hello world",
                "chunk_index": 0,
                "total_chunks": 1,
            }
        ]
        path = tmp_path / "chunks.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        pipeline = JSONChunkIngestionPipeline()
        chunks = pipeline.ingest(str(path))
        assert len(chunks) == 1

    def test_ingest_file_not_found_raises(self) -> None:
        pipeline = JSONChunkIngestionPipeline()
        with pytest.raises(FileNotFoundError, match="Source file not found"):
            pipeline.ingest("nonexistent.json")

    def test_ingest_wrong_extension_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "chunks.csv"
        path.write_text("id,text", encoding="utf-8")
        pipeline = JSONChunkIngestionPipeline()
        with pytest.raises(ValueError, match="Expected a .json file"):
            pipeline.ingest(path)

    def test_ingest_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "chunks.json"
        path.write_text("not valid json {{{", encoding="utf-8")
        pipeline = JSONChunkIngestionPipeline()
        with pytest.raises(ValueError, match="Invalid JSON"):
            pipeline.ingest(path)

    def test_ingest_json_object_not_array_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "chunks.json"
        path.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        pipeline = JSONChunkIngestionPipeline()
        with pytest.raises(ValueError, match="Expected a JSON array"):
            pipeline.ingest(path)

    def test_ingest_without_embedding_uses_empty_list(self, tmp_path: Path) -> None:
        """Chunks without embedding field default to empty list."""
        data = [
            {
                "chunk_id": "chunk-001",
                "parent_doc_id": "doc-001",
                "text": "Hello world",
                "chunk_index": 0,
                "total_chunks": 1,
            }
        ]
        path = tmp_path / "chunks.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        pipeline = JSONChunkIngestionPipeline()
        chunks = pipeline.ingest(path)
        assert chunks[0].embedding == []
        assert chunks[0].is_embedded is False

    def test_ingest_with_metadata(self, tmp_path: Path) -> None:
        """Chunks with metadata are correctly parsed."""
        data = [
            {
                "chunk_id": "chunk-001",
                "parent_doc_id": "doc-001",
                "text": "Hello world",
                "chunk_index": 0,
                "total_chunks": 1,
                "metadata": {"language": "fr"},
            }
        ]
        path = tmp_path / "chunks.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        pipeline = JSONChunkIngestionPipeline()
        chunks = pipeline.ingest(path)
        assert chunks[0].metadata.get("language") == "fr"


class TestJSONChunkIngestionPipelineValidate:
    """Tests for JSONChunkIngestionPipeline.validate()"""

    def _make_chunk(self, chunk_id: str, text: str) -> Chunk:
        from src.shared.models import ChunkMetadata

        return Chunk(
            chunk_id=chunk_id,
            parent_doc_id="doc-001",
            text=text,
            chunk_index=0,
            total_chunks=1,
            metadata=ChunkMetadata(),
        )

    def test_validate_removes_duplicates(self) -> None:
        """Duplicate chunks by content are removed."""
        pipeline = JSONChunkIngestionPipeline()
        chunks = [
            self._make_chunk("chunk-001", "Same text"),
            self._make_chunk("chunk-002", "Same text"),
            self._make_chunk("chunk-003", "Different text"),
        ]
        result = pipeline.validate(chunks)
        assert len(result) == 2

    def test_validate_keeps_unique_chunks(self) -> None:
        pipeline = JSONChunkIngestionPipeline()
        chunks = [
            self._make_chunk("chunk-001", "Text one"),
            self._make_chunk("chunk-002", "Text two"),
            self._make_chunk("chunk-003", "Text three"),
        ]
        result = pipeline.validate(chunks)
        assert len(result) == 3

    def test_validate_empty_list_raises(self) -> None:
        pipeline = JSONChunkIngestionPipeline()
        with pytest.raises(ValueError, match="No valid chunks found"):
            pipeline.validate([])


class TestJSONChunkIngestionPipelineParseChunk:
    """Tests for JSONChunkIngestionPipeline._parse_chunk()"""

    def test_parse_missing_required_field_raises(self) -> None:
        raw = {
            "chunk_id": "chunk-001",
            "parent_doc_id": "doc-001",
            "text": "Hello",
            # missing chunk_index and total_chunks
        }
        with pytest.raises(ValueError, match="missing required fields"):
            JSONChunkIngestionPipeline._parse_chunk(raw)

    def test_parse_non_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected a JSON object"):
            JSONChunkIngestionPipeline._parse_chunk("not a dict")

    def test_parse_all_required_fields(self) -> None:
        raw = {
            "chunk_id": "chunk-001",
            "parent_doc_id": "doc-001",
            "text": "Hello world",
            "chunk_index": 0,
            "total_chunks": 1,
        }
        chunk = JSONChunkIngestionPipeline._parse_chunk(raw)
        assert chunk.chunk_id == "chunk-001"
        assert chunk.text == "Hello world"
        assert chunk.embedding == []
