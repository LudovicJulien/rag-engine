# src/ingestion/json_pipeline.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.ingestion.pipeline import DataIngestionPipeline
from src.shared.models import Chunk, ChunkMetadata


class JSONChunkIngestionPipeline(DataIngestionPipeline):
    """Ingestion pipeline for pre-chunked JSON files.

    Expects a JSON file containing a list of chunk objects.
    Each chunk must have been pre-processed and chunked externally
    before being ingested into the RAG engine.

    Example JSON format:
        [
            {
                "chunk_id": "chunk-001",
                "parent_doc_id": "doc-001",
                "text": "Some text content...",
                "chunk_index": 0,
                "total_chunks": 3,
                "metadata": {"language": "fr"}
            }
        ]
    """

    def ingest(self, source: str | Path) -> list[Chunk]:
        """Load pre-chunked JSON from a file path.

        Args:
            source: Path to a JSON file containing a list of chunk objects.

        Returns:
            List of validated Chunk objects.

        Raises:
            FileNotFoundError: If the source file does not exist.
            ValueError: If the file is not valid JSON or contains no chunks.
        """
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")

        if path.suffix != ".json":
            raise ValueError(f"Expected a .json file, got: {path.suffix}")

        with path.open("r", encoding="utf-8") as f:
            try:
                raw = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path}: {e}") from e

        if not isinstance(raw, list):
            raise ValueError(
                f"Expected a JSON array at root level, got: {type(raw).__name__}"
            )

        chunks = [self._parse_chunk(item) for item in raw]
        return self.validate(chunks)

    def validate(self, chunks: list[Chunk]) -> list[Chunk]:
        """Validate chunks and deduplicate by content hash.

        Args:
            chunks: Raw list of Chunk objects to validate.

        Returns:
            Deduplicated list of valid chunks.

        Raises:
            ValueError: If no valid chunks remain after validation.
        """
        seen_hashes: set[str] = set()
        valid: list[Chunk] = []

        for chunk in chunks:
            content_hash = hashlib.md5(chunk.text.encode()).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            valid.append(chunk)

        if not valid:
            raise ValueError("No valid chunks found after validation and deduplication")

        return valid

    @staticmethod
    def _parse_chunk(raw: object) -> Chunk:
        """Parse a raw dict into a Chunk object.

        Args:
            raw: Raw dict from JSON deserialization.

        Returns:
            Validated Chunk object.

        Raises:
            ValueError: If required fields are missing or have wrong types.
        """
        if not isinstance(raw, dict):
            raise ValueError(
                f"Expected a JSON object for each chunk, got: {type(raw).__name__}"
            )

        required_fields = {
            "chunk_id",
            "parent_doc_id",
            "text",
            "chunk_index",
            "total_chunks",
        }
        missing = required_fields - raw.keys()
        if missing:
            raise ValueError(f"Chunk is missing required fields: {missing}")

        return Chunk(
            chunk_id=str(raw["chunk_id"]),
            parent_doc_id=str(raw["parent_doc_id"]),
            text=str(raw["text"]),
            chunk_index=int(raw["chunk_index"]),
            total_chunks=int(raw["total_chunks"]),
            embedding=list(raw.get("embedding", [])),
            metadata=ChunkMetadata(metadata=dict(raw.get("metadata", {}))),
        )
