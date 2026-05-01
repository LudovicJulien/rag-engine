# src/ingestion/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class Document:
    """Represents a raw document before chunking.

    Attributes:
        id: Unique identifier for the document.
        content: Raw text content of the document.
        source: Origin of the document (file path, URL, API name, etc.).
        metadata: Arbitrary key-value pairs for domain-specific information.
        ingested_at: Timestamp of when the document was ingested.
    """

    id: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    ingested_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Document id cannot be empty")
        if not self.content:
            raise ValueError("Document content cannot be empty")
        if not self.source:
            raise ValueError("Document source cannot be empty")

    @property
    def word_count(self) -> int:
        """Returns the number of words in the document content."""
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        """Returns the number of characters in the document content."""
        return len(self.content)
