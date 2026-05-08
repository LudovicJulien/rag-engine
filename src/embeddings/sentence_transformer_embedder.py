# src/embeddings/sentence_transformer_embedder.py
from __future__ import annotations

import hashlib
from typing import Any

from sentence_transformers import SentenceTransformer

from src.embeddings.embedding_model import EmbeddingModel


class SentenceTransformerEmbedder(EmbeddingModel):
    """Dense embedding model backed by HuggingFace SentenceTransformers.

    Downloads and caches the model locally on first use.
    Supports any model from the SentenceTransformers model hub.

    Example usage:
        embedder = SentenceTransformerEmbedder(
            model_name="intfloat/multilingual-e5-large"
        )
        vector = embedder.embed_text("Le Plateau est un quartier de Montréal")
        vectors = embedder.embed_batch(["text one", "text two"])
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        """Initialize the SentenceTransformer embedder.

        Args:
            model_name: HuggingFace model name or local path.
                        Defaults to multilingual-e5-large for FR/EN support.
            device: Device to run inference on ('cpu', 'cuda', 'mps').
                    If None, automatically selects the best available device.
            normalize_embeddings: Whether to L2-normalize embeddings.
                                  Normalized embeddings work better with
                                  cosine similarity in Qdrant.
        """
        self._model_name = model_name
        self._normalize = normalize_embeddings
        self._model: SentenceTransformer = SentenceTransformer(
            model_name, device=device
        )
        self._embedding_dim: int = self._model.get_sentence_embedding_dimension() or 0

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        """Return the HuggingFace model name."""
        return self._model_name

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a dense vector.

        Args:
            text: The text to embed. Must be non-empty.

        Returns:
            Dense embedding vector as a list of floats.

        Raises:
            ValueError: If text is empty.
        """
        if not text:
            raise ValueError("Cannot embed empty text")

        result: list[float] = self._model.encode(
            text,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        ).tolist()
        return result

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a list of texts into dense vectors.

        Processes texts in batches to avoid OOM errors on large corpora.
        Order of returned embeddings matches order of input texts.

        Args:
            texts: List of texts to embed. Must be non-empty.
            batch_size: Number of texts per batch. Defaults to 32.

        Returns:
            List of dense embedding vectors in input order.

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts")

        vectors: Any = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query.

        Some models (e.g. e5 family) use different prefixes for
        queries vs documents. This method adds the query prefix
        when relevant.

        Args:
            query: The search query to embed.

        Returns:
            Dense embedding vector for the query.

        Raises:
            ValueError: If query is empty.
        """
        if not query:
            raise ValueError("Cannot embed empty query")

        prefixed = self._add_query_prefix(query)
        return self.embed_text(prefixed)

    def _add_query_prefix(self, text: str) -> str:
        """Add model-specific query prefix if needed.

        The e5 family of models requires 'query: ' prefix for queries
        and 'passage: ' prefix for documents to achieve best performance.

        Args:
            text: Raw query text.

        Returns:
            Text with appropriate prefix applied.
        """
        e5_models = {"intfloat/multilingual-e5-large", "intfloat/e5-large-v2"}
        if self._model_name in e5_models:
            return f"query: {text}"
        return text

    def get_cache_key(self) -> str:
        """Return a unique cache key for this embedder configuration.

        Used by the embedding cache layer to namespace cached vectors
        by model — prevents cross-model cache collisions.

        Returns:
            MD5 hash of the model name.
        """
        return hashlib.md5(self._model_name.encode()).hexdigest()
