from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.shared.models import Chunk, Document

_DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n\n",
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
    " ",
    "",
)


@dataclass(frozen=True)
class ChunkConfig:
    """Immutable configuration for a text chunker.

    Attributes:
        chunk_size:    Target chunk length in characters. Chunks may exceed this
                       by at most ``chunk_overlap`` characters due to overlap prefixing.
        chunk_overlap: Characters from the tail of chunk N prepended to chunk N+1.
                       Must be strictly less than ``chunk_size``.
        min_chunk_size: Fragments shorter than this threshold are discarded after
                        splitting. Useful for filtering isolated punctuation or
                        whitespace-only remnants.
        separators:    Ordered sequence of separator strings tried from broadest
                       (section break) to finest (single character). The empty
                       string ``""`` is the absolute fallback: it splits character
                       by character and guarantees termination.
    """

    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 32
    separators: tuple[str, ...] = field(default=_DEFAULT_SEPARATORS)

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < "
                f"chunk_size ({self.chunk_size})"
            )
        if self.min_chunk_size < 0:
            raise ValueError(f"min_chunk_size must be >= 0, got {self.min_chunk_size}")
        if not self.separators:
            raise ValueError("separators must not be empty")


class TextChunker(ABC):
    """Contract for splitting a Document into Chunks.

    Every implementation accepts a Document and returns an ordered list of
    Chunk objects ready for embedding. Callers depend only on this interface.
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks.

        Args:
            document: Raw document to split.

        Returns:
            Non-empty ordered list of Chunk objects (chunk_index 0-based,
            total_chunks consistent across all returned chunks).

        Raises:
            ValueError: If the document produces no chunks >= min_chunk_size.
        """
        ...
