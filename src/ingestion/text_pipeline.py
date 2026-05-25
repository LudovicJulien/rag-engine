from __future__ import annotations

import hashlib
from pathlib import Path

from src.chunking import get_chunker
from src.chunking.chunker import TextChunker
from src.ingestion.ingestion_pipeline import DataIngestionPipeline
from src.shared.models import Chunk, ChunkMetadata, Document

_SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".txt", ".md"})


class TextIngestionPipeline(DataIngestionPipeline):
    """Ingestion pipeline for plain-text and Markdown files.

    Reads a raw .txt or .md file, wraps it in a Document, and delegates
    chunking to a TextChunker. Unlike JSONChunkIngestionPipeline, no
    pre-chunking is required — the pipeline handles it internally.

    Example usage::

        pipeline = TextIngestionPipeline()
        chunks = pipeline.ingest_and_validate("docs/guide.md")
    """

    def __init__(self, chunker: TextChunker | None = None) -> None:
        self._chunker: TextChunker = chunker if chunker is not None else get_chunker()

    def ingest(self, source: str | Path) -> list[Chunk]:
        """Read a .txt or .md file and return deduplicated Chunk objects.

        Steps:
            1. Validate path existence and file extension.
            2. Read text content; reject empty files.
            3. Build a Document with a deterministic doc_id from the filename.
            4. Chunk via the injected TextChunker.
            5. Deduplicate via validate().

        Args:
            source: Path (str or Path) to a .txt or .md file.

        Returns:
            Non-empty deduplicated list of Chunk objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the extension is unsupported, the file is empty,
                        or the chunker produces no valid chunks.
        """
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")

        if path.suffix not in _SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported file type '{path.suffix}'. "
                f"Expected one of: {sorted(_SUPPORTED_SUFFIXES)}"
            )

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"File is empty: {path}")

        doc_id = hashlib.md5(path.name.encode()).hexdigest()[:16]
        document = Document(
            doc_id=doc_id,
            text=text,
            source=str(path),
            metadata=ChunkMetadata({"source": str(path), "filename": path.name}),
        )

        return self.validate(self._chunker.chunk(document))

    def validate(self, chunks: list[Chunk]) -> list[Chunk]:
        """Deduplicate chunks by MD5 of their text content.

        Preserves the first occurrence of each unique text; later duplicates
        are silently dropped. Order is otherwise preserved.

        Args:
            chunks: List of Chunk objects to deduplicate.

        Returns:
            Non-empty deduplicated list.

        Raises:
            ValueError: If no chunks remain after deduplication.
        """
        seen: set[str] = set()
        valid: list[Chunk] = []

        for chunk in chunks:
            h = hashlib.md5(chunk.text.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                valid.append(chunk)

        if not valid:
            raise ValueError("No valid chunks found after deduplication")

        return valid
