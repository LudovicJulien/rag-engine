# src/vector_store/store.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.embeddings.hybrid_embedder import HybridEmbedding
from src.shared.models import Chunk


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A retrieved chunk paired with its relevance score.

    Attributes:
        chunk: The matched domain chunk.
        score: Similarity score in [0.0, 1.0] — higher is more relevant.
    """

    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class UpsertResult:
    """Summary of a completed upsert operation.

    Attributes:
        upserted: Number of points successfully written.
        failed: Number of points that could not be written (default 0).
    """

    upserted: int
    failed: int = field(default=0)

    @property
    def total(self) -> int:
        return self.upserted + self.failed


class VectorStore(ABC):
    """Abstract base class for all vector store implementations.

    Defines the contract that any vector store backend must fulfill to
    integrate with the RAG engine. The engine depends only on this
    interface — it does not care whether vectors live in Qdrant,
    Pinecone, Weaviate, or any other backend.

    Concrete implementations must call ``_validate_upsert_inputs`` at
    the start of their ``upsert`` override to enforce the shared
    pre-conditions documented in that method's docstring.

    Example usage::

        class QdrantVectorStore(VectorStore):
            def create_collection(self, dense_dim: int) -> None:
                ...
    """

    @abstractmethod
    def create_collection(self, dense_dim: int) -> None:
        """Create the collection with named dense and sparse vector configs.

        Args:
            dense_dim: Dimensionality of the dense vectors (depends on the
                embedding model, e.g. 768 for ``all-MiniLM-L6-v2``).

        Raises:
            RuntimeError: If the collection already exists or the backend
                returns an error.
        """

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the collection and all its data.

        Raises:
            RuntimeError: If the collection does not exist or the backend
                returns an error.
        """

    @abstractmethod
    def collection_exists(self) -> bool:
        """Return ``True`` if the collection already exists in the backend."""

    @abstractmethod
    def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[HybridEmbedding],
    ) -> UpsertResult:
        """Upsert chunks with their hybrid embeddings.

        Implementations **must** call ``_validate_upsert_inputs`` first.

        Args:
            chunks: Domain chunks to store. Must be non-empty.
            embeddings: Hybrid embeddings aligned with *chunks* (same order,
                same length).

        Returns:
            An :class:`UpsertResult` with the number of points written and
            the number that failed (if the backend supports partial writes).

        Raises:
            ValueError: If *chunks* is empty or lengths do not match.
        """

    @abstractmethod
    def search(
        self,
        query_embedding: HybridEmbedding,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Search for the most similar chunks to the query embedding.

        Args:
            query_embedding: Hybrid embedding of the user query.
            top_k: Maximum number of results to return. Must be ≥ 1.
            score_threshold: Minimum similarity score in [0.0, 1.0].
                ``None`` means no filter is applied; every candidate
                up to *top_k* is returned regardless of score.

        Returns:
            List of :class:`SearchResult` objects ordered by descending
            relevance score (most relevant first).

        Raises:
            ValueError: If *top_k* < 1.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Return ``True`` if the vector store backend is reachable."""

    @abstractmethod
    def count(self) -> int:
        """Return the total number of points in this instance's collection."""

    @abstractmethod
    def delete_chunk_by_id(self, chunk_id: str) -> bool:
        """Delete a single point by its chunk_id.

        Args:
            chunk_id: The unique identifier of the chunk to delete.

        Returns:
            True if the point was deleted, False if the chunk was not found.

        Raises:
            RuntimeError: If the backend returns an error.
        """

    @abstractmethod
    def create_payload_index(self, field_name: str) -> None:
        """Create a keyword payload index on the given field.

        Intended for filterable metadata fields such as language

        Args:
            field_name: The payload field name to index.

        Raises:
            RuntimeError: If the backend returns an error.
        """

    @abstractmethod
    def scroll_all_chunks(self, batch_size: int = 100) -> list[Chunk]:
        """Scroll through the entire collection and return all chunks.

        Paginates internally so the full collection is never loaded in a single
        request.

        Args:
            batch_size: Number of points to fetch per page.

        Returns:
            All chunks in the collection, or an empty list if the collection is
            empty.
        """

    @staticmethod
    def _validate_upsert_inputs(
        chunks: list[Chunk],
        embeddings: list[HybridEmbedding],
    ) -> None:
        """Validate pre-conditions shared by all ``upsert`` implementations.

        Args:
            chunks: The chunk list passed to :meth:`upsert`.
            embeddings: The embedding list passed to :meth:`upsert`.

        Raises:
            ValueError: If *chunks* is empty or the two lists differ in length.
        """
        if not chunks:
            raise ValueError("'chunks' must not be empty.")
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Length mismatch: {len(chunks)} chunk(s) vs "
                f"{len(embeddings)} embedding(s)."
            )
