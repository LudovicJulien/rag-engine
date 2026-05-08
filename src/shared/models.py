# src/shared/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetadataFilter:
    """A single equality filter on a payload field.

    Attributes:
        field: Dot-notation path to the field (e.g. ``"metadata.language"``,
            ``"parent_doc_id"``).
        value: The exact value the field must match.

    Raises:
        ValueError: If *field* is empty.
    """

    field: str
    value: str

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError("MetadataFilter field cannot be empty")


@dataclass
class ChunkMetadata:
    """Domain-agnostic metadata container for a chunk.

    Attributes:
        metadata: Arbitrary key-value pairs inherited from the parent document
                  plus any chunk-level annotations added during processing.
    """

    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Safe metadata access with default fallback."""
        return self.metadata.get(key, default)


@dataclass
class Chunk:
    """Represents a text chunk derived from a parent Document.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        parent_doc_id: ID of the Document this chunk was derived from.
        text: The actual text content of this chunk.
        chunk_index: Position of this chunk within the parent document (0-based).
        total_chunks: Total number of chunks derived from the parent document.
        embedding: Dense vector representation of the chunk text.
        metadata: Domain-agnostic metadata inherited from parent + chunk-level info.
    """

    chunk_id: str
    parent_doc_id: str
    text: str
    chunk_index: int
    total_chunks: int
    embedding: list[float] = field(default_factory=list)
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("Chunk chunk_id cannot be empty")
        if not self.parent_doc_id:
            raise ValueError("Chunk parent_doc_id cannot be empty")
        if not self.text:
            raise ValueError("Chunk text cannot be empty")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be >= 0")
        if self.total_chunks < 1:
            raise ValueError("total_chunks must be >= 1")
        if self.chunk_index >= self.total_chunks:
            raise ValueError("chunk_index must be < total_chunks")

    @property
    def is_embedded(self) -> bool:
        """Returns True if this chunk has been embedded."""
        return len(self.embedding) > 0

    @property
    def word_count(self) -> int:
        """Returns the number of words in the chunk text."""
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        """Returns the number of characters in the chunk text."""
        return len(self.text)

    @property
    def is_first(self) -> bool:
        """Returns True if this is the first chunk of the parent document."""
        return self.chunk_index == 0

    @property
    def is_last(self) -> bool:
        """Returns True if this is the last chunk of the parent document."""
        return self.chunk_index == self.total_chunks - 1
