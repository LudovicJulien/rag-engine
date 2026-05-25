from __future__ import annotations

from typing import TYPE_CHECKING

from src.chunking.chunker import ChunkConfig, TextChunker
from src.chunking.recursive import RecursiveTextChunker

if TYPE_CHECKING:
    from src.pipeline.config import Settings

__all__ = ["ChunkConfig", "RecursiveTextChunker", "TextChunker", "get_chunker"]


def get_chunker(settings: Settings | None = None) -> RecursiveTextChunker:
    """Build a RecursiveTextChunker wired from application settings.

    Args:
        settings: Populated Settings instance. Defaults to the module-level
                  singleton returned by ``get_settings()`` when ``None``.

    Returns:
        A ready-to-use :class:`RecursiveTextChunker` configured with
        ``chunker_chunk_size``, ``chunker_chunk_overlap``, and
        ``chunker_min_chunk_size`` from *settings*.

    Raises:
        ValueError: If the settings produce an invalid :class:`ChunkConfig`
                    (e.g. ``chunker_chunk_overlap >= chunker_chunk_size``).
    """
    from src.pipeline.config import get_settings

    s = settings if settings is not None else get_settings()
    config = ChunkConfig(
        chunk_size=s.chunker_chunk_size,
        chunk_overlap=s.chunker_chunk_overlap,
        min_chunk_size=s.chunker_min_chunk_size,
    )
    return RecursiveTextChunker(config)
