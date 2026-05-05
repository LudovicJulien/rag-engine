# src/vector_store/qdrant_store.py
from __future__ import annotations

import time
import uuid
from typing import Any, Callable, TypeVar, cast

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    Fusion,
    HnswConfigDiff,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.embeddings.hybrid_embedder import HybridEmbedding
from src.shared.models import Chunk, ChunkMetadata
from src.vector_store.store import SearchResult, UpsertResult, VectorStore

_T = TypeVar("_T")

# Stable namespace for deterministic UUID5 point IDs derived from chunk_id strings.
_POINT_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _retry(fn: Callable[[], _T], *, max_attempts: int, base_delay: float) -> _T:
    """Retry a callable with exponential backoff.

    Args:
        fn: Zero-argument callable to call.
        max_attempts: Maximum number of attempts before raising.
        base_delay: Base delay in seconds; doubles on each retry
            (``base_delay * 2^attempt``).

    Returns:
        The return value of fn on success.

    Raises:
        ConnectionError: If all attempts fail, wrapping the last exception.
    """
    last_exc: BaseException = RuntimeError("No attempts made")
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(base_delay * (2**attempt))
    raise ConnectionError(
        f"Failed to connect to Qdrant after {max_attempts} attempts: {last_exc}"
    ) from last_exc


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store with connection retry logic.

    Stores chunks as named-vector points in a Qdrant collection:
    - ``"dense"``: cosine-distance HNSW index (SentenceTransformer vectors).
    - ``"sparse"``: sparse dot-product index (BM25 vectors).

    At search time, both indexes are queried in parallel via Prefetch and
    fused with Reciprocal Rank Fusion (RRF).

    Example usage:
        store = QdrantVectorStore(
            host="localhost", port=6333, collection_name="rag-collection"
        )
        store.create_collection(dense_dim=1024)
        store.upsert(chunks, embeddings)
        results = store.search(query_embedding, top_k=5)
    """

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self._collection_name = collection_name
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._client = self._connect(host, port)

    def _connect(self, host: str, port: int) -> QdrantClient:
        client = QdrantClient(host=host, port=port)
        _retry(
            lambda: client.get_collections(),
            max_attempts=self._max_retries,
            base_delay=self._retry_delay,
        )
        return client

    def create_collection(self, dense_dim: int) -> None:
        if not self.collection_exists():
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config={
                    "dense": VectorParams(size=dense_dim, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(),
                },
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            )

    def delete_collection(self) -> None:
        self._client.delete_collection(collection_name=self._collection_name)

    def collection_exists(self) -> bool:
        try:
            self._client.get_collection(collection_name=self._collection_name)
            return True
        except UnexpectedResponse:
            return False

    def upsert(
        self, chunks: list[Chunk], embeddings: list[HybridEmbedding]
    ) -> UpsertResult:
        VectorStore._validate_upsert_inputs(chunks, embeddings)

        points = [
            PointStruct(
                id=str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk.chunk_id)),
                vector={
                    "dense": embedding.dense,
                    "sparse": SparseVector(
                        indices=embedding.sparse_indices,
                        values=embedding.sparse_values,
                    ),
                },
                payload={
                    "chunk_id": chunk.chunk_id,
                    "parent_doc_id": chunk.parent_doc_id,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "metadata": chunk.metadata.metadata,
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        self._client.upsert(collection_name=self._collection_name, points=points)
        return UpsertResult(upserted=len(points))

    def search(
        self,
        query_embedding: HybridEmbedding,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        prefetch_limit = top_k * 2
        threshold = score_threshold if score_threshold else None

        response = self._client.query_points(
            collection_name=self._collection_name,
            prefetch=[
                Prefetch(
                    query=query_embedding.dense,
                    using="dense",
                    limit=prefetch_limit,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=query_embedding.sparse_indices,
                        values=query_embedding.sparse_values,
                    ),
                    using="sparse",
                    limit=prefetch_limit,
                ),
            ],
            query=Fusion.RRF,
            limit=top_k,
            score_threshold=threshold,
        )

        return [
            SearchResult(chunk=self._point_to_chunk(p), score=p.score)
            for p in response.points
        ]

    def health_check(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except (UnexpectedResponse, ConnectionError):
            return False

    def count(self) -> int:
        result = self._client.count(collection_name=self._collection_name, exact=True)
        return int(result.count)

    def delete_chunk_by_id(self, chunk_id: str) -> bool:
        """Delete a single point by its chunk_id.

        Args:
            chunk_id: The unique identifier of the chunk to delete.

        Returns:
            True if the point was deleted, False if the chunk was not found.

        Raises:
            RuntimeError: If the backend returns an error.
        """
        point_id = str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))
        try:
            existing = self._client.retrieve(
                collection_name=self._collection_name,
                ids=[point_id],
                with_payload=False,
                with_vectors=False,
            )
            if not existing:
                return False
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=PointIdsList(points=[point_id]),
            )
            return True
        except UnexpectedResponse as exc:
            raise RuntimeError(f"Failed to delete chunk '{chunk_id}': {exc}") from exc

    def create_payload_index(self, field_name: str) -> None:
        """Create a keyword payload index on the given field.

        Args:
            field_name: The payload field name to index.

        Raises:
            RuntimeError: If the backend returns an error.
        """
        try:
            self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except UnexpectedResponse as exc:
            raise RuntimeError(
                f"Failed to create payload index on '{field_name}': {exc}"
            ) from exc

    def scroll_all_chunks(self, batch_size: int = 100) -> list[Chunk]:
        """Scroll through the entire collection and return all chunks.

        Args:
            batch_size: Number of points to fetch per page.

        Returns:
            All chunks in the collection, or an empty list if the collection is
            empty.
        """
        chunks: list[Chunk] = []
        offset = None
        while True:
            records, next_offset = self._client.scroll(
                collection_name=self._collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                chunks.append(self._point_to_chunk(record))
            if next_offset is None:
                break
            offset = next_offset
        return chunks

    @staticmethod
    def _point_to_chunk(point: Any) -> Chunk:
        payload: dict[str, Any] = point.payload or {}
        try:
            return Chunk(
                chunk_id=str(payload["chunk_id"]),
                parent_doc_id=str(payload["parent_doc_id"]),
                text=str(payload["text"]),
                chunk_index=int(payload["chunk_index"]),
                total_chunks=int(payload["total_chunks"]),
                metadata=ChunkMetadata(
                    metadata=cast(dict[str, Any], payload.get("metadata", {}))
                ),
            )
        except KeyError as e:
            raise ValueError(f"Invalid payload: missing {e}") from e
