# src/retrieval/retriever.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.embeddings.hybrid_embedder import HybridEmbedding
from src.shared.models import MetadataFilter
from src.vector_store.vector_store import SearchResult


class Retriever(ABC):
    """Abstract base class for all retrieval implementations.

    Defines the contract that any retrieval strategy must fulfill to
    integrate with the RAG engine. The engine depends only on this
    interface — it does not care whether retrieval is dense, sparse,
    hybrid, or re-ranked.

    Example usage::

        class DenseRetriever(Retriever):
            def retrieve(
                self,
                query_embedding: HybridEmbedding,
                *,
                top_k: Optional[int] = None,
                score_threshold: Optional[float] = None,
            ) -> list[SearchResult]:
                ...
    """

    @abstractmethod
    def retrieve(
        self,
        query_embedding: HybridEmbedding,
        *,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filters: Optional[list[MetadataFilter]] = None,
    ) -> list[SearchResult]:
        """Return the most relevant chunks for *query_embedding*.

        Args:
            query_embedding: Hybrid embedding of the user query produced by
                :class:`~src.embeddings.hybrid_embedder.HybridEmbedder`.
            top_k: Maximum number of results to return.  ``None`` falls back
                to the implementation's configured default.
            score_threshold: Minimum relevance score in [0.0, 1.0].
                ``None`` disables filtering so every candidate up to
                *top_k* is returned.
            filters: Optional equality filters applied to payload fields
                before scoring.  All conditions are ANDed together.
                ``None`` disables payload filtering.

        Returns:
            Ordered list of :class:`~src.vector_store.vector_store.SearchResult`,
            highest score first.  May be shorter than *top_k* when fewer
            results satisfy the score threshold.

        Raises:
            ValueError: If *top_k* < 1 or *score_threshold* is out of [0, 1].
            RuntimeError: On network or serialization errors from the backend.
        """
