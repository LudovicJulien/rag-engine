# src/ingestion/pipeline.py
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.shared.models import Chunk


class DataIngestionPipeline(ABC):
    """Abstract base class for all data ingestion pipelines.

    Any pipeline that feeds data into the RAG engine must implement
    this interface. The engine only depends on this contract — it does
    not care how chunks were produced or where they come from.

    Example usage:
        class JSONChunkIngestionPipeline(DataIngestionPipeline):
            def ingest(self, source: str | Path) -> list[Chunk]:
                ...
    """

    @abstractmethod
    def ingest(self, source: str | Path) -> list[Chunk]:
        """Load and validate chunks from a data source.

        Args:
            source: Path to a file, directory, or any string identifier
                    meaningful to the concrete implementation (URL, db name, etc.)

        Returns:
            A non-empty list of validated Chunk objects ready for embedding.

        Raises:
            FileNotFoundError: If the source path does not exist.
            ValueError: If the source contains no valid chunks.
        """

    @abstractmethod
    def validate(self, chunks: list[Chunk]) -> list[Chunk]:
        """Validate a list of chunks and return only the valid ones.

        Args:
            chunks: Raw chunks to validate.

        Returns:
            Filtered list of valid chunks.

        Raises:
            ValueError: If no valid chunks remain after validation.
        """

    def ingest_and_validate(self, source: str | Path) -> list[Chunk]:
        """Convenience method — ingest then validate in one call.

        This method is not abstract — concrete pipelines get it for free
        but can override if needed.

        Args:
            source: Same as ingest().

        Returns:
            Validated list of Chunk objects.
        """
        chunks = self.ingest(source)
        return self.validate(chunks)
