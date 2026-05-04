# src/embeddings/hybrid_embedder.py
from __future__ import annotations

import hashlib

from src.embeddings.bm25_embedder import BM25SparseEmbedder
from src.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder


class HybridEmbedding:
    """Container for a hybrid embedding result.

    Holds both dense and sparse vectors for a single text,
    ready to be upserted into Qdrant as named vectors.

    Attributes:
        dense: Dense vector from SentenceTransformer.
        sparse_indices: Non-zero indices of the sparse BM25 vector.
        sparse_values: Non-zero values of the sparse BM25 vector.
        text: Original text that was embedded.
    """

    def __init__(
        self,
        dense: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        text: str = "",
    ) -> None:
        self.dense = dense
        self.sparse_indices = sparse_indices
        self.sparse_values = sparse_values
        self.text = text

    @property
    def dense_dim(self) -> int:
        """Dimensionality of the dense vector."""
        return len(self.dense)

    @property
    def sparse_nnz(self) -> int:
        """Number of non-zero elements in the sparse vector."""
        return len(self.sparse_indices)

    def to_qdrant_payload(self) -> dict[str, object]:
        """Serialize to Qdrant-compatible named vector format.

        Returns:
            Dict with 'dense' and 'sparse' keys ready for Qdrant upsert.
        """
        return {
            "dense": self.dense,
            "sparse": {
                "indices": self.sparse_indices,
                "values": self.sparse_values,
            },
        }


class HybridEmbedder:
    """Combines dense and sparse embeddings for hybrid retrieval.

    Wraps a SentenceTransformerEmbedder (dense) and a BM25SparseEmbedder
    (sparse) to produce HybridEmbedding objects ready for Qdrant.

    The alpha parameter controls the balance between dense and sparse
    during score fusion — but note that actual RRF fusion happens at
    retrieval time in Qdrant, not here. HybridEmbedder only produces
    the vectors.

    Example usage:
        dense = SentenceTransformerEmbedder()
        sparse = BM25SparseEmbedder().fit(corpus_texts)

        embedder = HybridEmbedder(dense=dense, sparse=sparse, alpha=0.7)
        result = embedder.embed_text("Le Plateau est branché")

        # result.dense        → dense vector for Qdrant
        # result.sparse_*     → sparse vector for Qdrant
    """

    def __init__(
        self,
        dense: SentenceTransformerEmbedder,
        sparse: BM25SparseEmbedder,
        alpha: float = 0.5,
    ) -> None:
        """Initialize HybridEmbedder.

        Args:
            dense: Fitted SentenceTransformerEmbedder for dense vectors.
            sparse: Fitted BM25SparseEmbedder for sparse vectors.
            alpha: Weight of dense embeddings in hybrid scoring.
                   0.0 = pure sparse (BM25 only)
                   0.5 = equal weight (default)
                   1.0 = pure dense (SentenceTransformer only)
                   Used for documentation and logging — actual fusion
                   is handled by Qdrant RRF at query time.

        Raises:
            ValueError: If alpha is not in [0.0, 1.0].
            RuntimeError: If sparse embedder is not fitted.
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")

        if not sparse.is_fitted:
            raise RuntimeError(
                "BM25SparseEmbedder must be fitted before passing to HybridEmbedder"
            )

        self._dense = dense
        self._sparse = sparse
        self.alpha = alpha

    @property
    def dense_dim(self) -> int:
        """Dimensionality of the dense vectors."""
        return self._dense.embedding_dim

    @property
    def sparse_dim(self) -> int:
        """Vocabulary size of the sparse vectors."""
        return self._sparse.embedding_dim

    @property
    def model_name(self) -> str:
        """Return combined model identifier."""
        return (
            f"hybrid(dense={self._dense.model_name},"
            f"sparse={self._sparse.model_name},"
            f"alpha={self.alpha})"
        )

    def embed_text(self, text: str) -> HybridEmbedding:
        """Embed a single text into a HybridEmbedding.

        Args:
            text: The text to embed. Must be non-empty.

        Returns:
            HybridEmbedding with dense and sparse vectors.

        Raises:
            ValueError: If text is empty.
        """
        if not text:
            raise ValueError("Cannot embed empty text")

        dense_vector = self._dense.embed_text(text)
        sparse_vector = self._sparse.embed_text(text)
        sparse_indices, sparse_values = self._to_sparse_format(sparse_vector)

        return HybridEmbedding(
            dense=dense_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            text=text,
        )

    def embed_query(self, query: str) -> HybridEmbedding:
        """Embed a search query into a HybridEmbedding.

        Uses embed_query() for dense (adds e5 prefix if needed)
        and embed_text() for sparse (no prefix needed for BM25).

        Args:
            query: The search query. Must be non-empty.

        Returns:
            HybridEmbedding with dense and sparse vectors for the query.

        Raises:
            ValueError: If query is empty.
        """
        if not query:
            raise ValueError("Cannot embed empty query")

        dense_vector = self._dense.embed_query(query)
        sparse_vector = self._sparse.embed_text(query)
        sparse_indices, sparse_values = self._to_sparse_format(sparse_vector)

        return HybridEmbedding(
            dense=dense_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            text=query,
        )

    def embed_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[HybridEmbedding]:
        """Embed a list of texts into HybridEmbeddings.

        Args:
            texts: List of texts to embed. Must be non-empty.
            batch_size: Batch size for dense embedding inference.

        Returns:
            List of HybridEmbedding objects in input order.

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts")

        dense_vectors = self._dense.embed_batch(texts, batch_size=batch_size)
        sparse_vectors = self._sparse.embed_batch(texts)

        results = []
        for text, dense, sparse in zip(texts, dense_vectors, sparse_vectors):
            indices, values = self._to_sparse_format(sparse)
            results.append(
                HybridEmbedding(
                    dense=dense,
                    sparse_indices=indices,
                    sparse_values=values,
                    text=text,
                )
            )
        return results

    def get_cache_key(self) -> str:
        """Return a unique cache key for this hybrid configuration.

        Returns:
            MD5 hash of the combined model name.
        """
        return hashlib.md5(self.model_name.encode()).hexdigest()

    @staticmethod
    def _to_sparse_format(
        vector: list[float],
    ) -> tuple[list[int], list[float]]:
        """Convert a dense sparse vector to Qdrant sparse format.

        Extracts non-zero indices and values from a full sparse vector.

        Args:
            vector: Full sparse vector (most values are 0.0).

        Returns:
            Tuple of (indices, values) for non-zero elements only.
        """
        indices = []
        values = []
        for idx, val in enumerate(vector):
            if val != 0:
                indices.append(idx)
                values.append(val)
        return indices, values
