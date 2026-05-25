from __future__ import annotations

import hashlib

from src.chunking.chunker import ChunkConfig, TextChunker
from src.shared.models import Chunk, Document


class RecursiveTextChunker(TextChunker):
    """Splits text using a hierarchy of separators, from broadest to finest.

    Pipeline: _split_recursive → _apply_overlap → strip+filter → Chunk objects.

    For a given text, tries separators[0]. Fragments that still exceed
    chunk_size are recursed with separators[1:]. The empty string ``""``
    at the end of the default separator list is the absolute fallback:
    it splits character by character and guarantees every fragment fits
    within chunk_size. Overlap is then prepended to each fragment from
    the tail of its predecessor so that content spanning boundaries is
    always represented in at least one chunk.
    """

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self._config = config or ChunkConfig()

    @property
    def config(self) -> ChunkConfig:
        return self._config

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into Chunks via the full recursive pipeline.

        Steps:
            1. _split_recursive  — hierarchical separator descent
            2. _apply_overlap    — prefix each fragment with tail of previous
            3. strip + filter    — discard whitespace-only and sub-threshold fragments
            4. build Chunks      — assign deterministic IDs, indices, metadata

        Args:
            document: Raw document to split.

        Returns:
            Non-empty ordered list of Chunk objects.

        Raises:
            ValueError: If all fragments are below min_chunk_size after filtering.
        """
        texts = self._split_recursive(document.text, list(self._config.separators))
        texts = self._apply_overlap(texts)
        texts = [
            stripped
            for t in texts
            if len(stripped := t.strip()) >= self._config.min_chunk_size
        ]

        if not texts:
            raise ValueError(
                f"Document '{document.doc_id}' produced no chunks "
                f"(all fragments < min_chunk_size={self._config.min_chunk_size})"
            )

        total = len(texts)
        return [
            Chunk(
                chunk_id=self._make_chunk_id(document.doc_id, i, text),
                parent_doc_id=document.doc_id,
                text=text,
                chunk_index=i,
                total_chunks=total,
                metadata=document.metadata,
            )
            for i, text in enumerate(texts)
        ]

    @staticmethod
    def _make_chunk_id(doc_id: str, index: int, text: str) -> str:
        """Generate a deterministic chunk ID from document ID, position, and content."""
        h = hashlib.md5(f"{doc_id}:{index}:{text}".encode()).hexdigest()[:12]
        return f"{doc_id}__chunk_{index:04d}_{h}"

    # ------------------------------------------------------------------
    # Internal algorithm
    # ------------------------------------------------------------------

    def _apply_overlap(self, texts: list[str]) -> list[str]:
        """Prefix each chunk (except the first) with the tail of its predecessor.

        Takes the last ``chunk_overlap`` characters of result[N] and prepends
        them to texts[N+1]. The suffix is taken from the *already-overlapped*
        result rather than the original input, so very short intermediate
        fragments propagate their full content forward naturally.

        Args:
            texts: Ordered list of fragments produced by _split_recursive.
                   May be empty or contain a single element.

        Returns:
            New list of the same length as *texts*.
            result[0] is identical to texts[0].
            result[i] == result[i-1][-chunk_overlap:] + texts[i]  for i >= 1.
            Returns the original list unchanged when chunk_overlap == 0 or
            len(texts) <= 1 (no allocation in the common no-overlap path).
        """
        overlap = self._config.chunk_overlap
        if overlap == 0 or len(texts) <= 1:
            return texts

        result: list[str] = [texts[0]]
        for i in range(1, len(texts)):
            suffix = result[i - 1][-overlap:]
            result.append(suffix + texts[i])
        return result

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Split *text* into fragments all <= chunk_size.

        Tries ``separators[0]`` first. Fragments that still exceed
        ``chunk_size`` are recursed with ``separators[1:]``. Adjacent
        small fragments produced by the same separator are re-merged
        (greedy left-to-right) to avoid unnecessarily small chunks.

        Args:
            text:       Text to split. Must be non-empty.
            separators: Ordered list of separators, broadest first.
                        Must contain ``""`` as the last element to
                        guarantee termination on any input.

        Returns:
            Non-empty list of non-empty strings. Every element satisfies
            ``len(element) <= chunk_size`` when ``""`` is reachable in
            the separator list. If no separator reduces the text (e.g.
            separators is empty), returns ``[text]`` unchanged.
        """
        # Fast path: already within budget — no splitting needed.
        if len(text) <= self._config.chunk_size:
            return [text]

        if not separators:
            # No separator left — caller must accept the oversized fragment.
            return [text]

        sep, remaining = separators[0], separators[1:]

        # Split on current separator. The empty string is handled separately
        # because str.split("") raises ValueError in Python.
        raw_splits = list(text) if sep == "" else text.split(sep)

        # Discard empty strings that arise from consecutive separators.
        splits = [s for s in raw_splits if s]

        # Edge case: separator not present in text — try next level directly.
        if not splits or (len(splits) == 1 and splits[0] == text):
            if remaining:
                return self._split_recursive(text, remaining)
            return [text]

        result: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for part in splits:
            part_len = len(part)
            # sep is re-inserted between accumulated parts when flushing.
            sep_add = len(sep) if current_parts else 0

            if current_len + sep_add + part_len <= self._config.chunk_size:
                current_parts.append(part)
                current_len += sep_add + part_len
            else:
                if current_parts:
                    result.append(sep.join(current_parts))
                    current_parts = []
                    current_len = 0

                if part_len > self._config.chunk_size and remaining:
                    # Fragment still too large — recurse with next separator.
                    result.extend(self._split_recursive(part, remaining))
                else:
                    current_parts = [part]
                    current_len = part_len

        if current_parts:
            result.append(sep.join(current_parts))

        return result
