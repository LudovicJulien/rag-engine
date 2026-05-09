# src/generation/context_window.py
from __future__ import annotations

import logging
import math

from src.shared.models import Chunk

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4
_MESSAGE_OVERHEAD_TOKENS = 100


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


def guard_context_window(
    chunks: list[Chunk],
    query: str,
    system_prompt: str,
    max_tokens: int,
) -> list[Chunk]:
    """Truncate chunks to fit within the estimated token budget.

    Chunks are assumed to be ordered by descending relevance; the guard
    therefore keeps the highest-relevance chunks first and drops from the
    tail. At least one chunk is always preserved, even when the base
    overhead alone exceeds *max_tokens*. The guard is disabled when
    *max_tokens* <= 0.

    Args:
        chunks: Ordered list of context chunks (highest relevance first).
        query: User query string used in the prompt.
        system_prompt: System prompt text used to estimate the base cost.
        max_tokens: Total token budget for the LLM context.  A value of
            0 or below disables the guard and returns *chunks* unchanged.

    Returns:
        Possibly-truncated list of chunks that fits within *max_tokens*.
    """
    if max_tokens <= 0 or not chunks:
        return chunks

    base_tokens = (
        _estimate_tokens(query)
        + _estimate_tokens(system_prompt)
        + _MESSAGE_OVERHEAD_TOKENS
    )
    budget = max_tokens - base_tokens

    selected: list[Chunk] = []
    for chunk in chunks:
        chunk_tokens = _estimate_tokens(chunk.text)
        if not selected:
            selected.append(chunk)
            budget -= chunk_tokens
        elif chunk_tokens <= budget:
            selected.append(chunk)
            budget -= chunk_tokens
        else:
            break

    if len(selected) < len(chunks):
        logger.warning(
            "Context window guard truncated chunks from %d to %d (max_tokens=%d)",
            len(chunks),
            len(selected),
            max_tokens,
        )

    return selected
