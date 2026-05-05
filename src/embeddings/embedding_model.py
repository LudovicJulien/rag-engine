# src/embeddings/embedding_model.py
from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    """Abstract base class for all embedding model implementations.

    Defines the contract that any embedding model must fulfill to integrate
    with the RAG engine. The engine depends only on this interface — it does
    not care whether embeddings come from SentenceTransformers, Ollama,
    OpenAI, or any other provider.

    Example usage:
        class SentenceTransformerEmbedder(EmbeddingModel):
            def embed_text(self, text: str) -> list[float]:
                ...

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                ...
    """

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a dense vector.

        Args:
            text: The text to embed. Must be non-empty.

        Returns:
            A list of floats representing the dense embedding vector.
            The dimensionality depends on the underlying model.

        Raises:
            ValueError: If text is empty.
        """

    @abstractmethod
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a list of texts into dense vectors.

        Implementations should process texts in batches to avoid OOM errors
        on large corpora. The order of returned embeddings must match
        the order of input texts.

        Args:
            texts: List of texts to embed. Must be non-empty.
            batch_size: Number of texts to process per batch.
                        Defaults to 32. Ignored by providers that
                        handle batching internally (e.g. OpenAI API).

        Returns:
            A list of embedding vectors in the same order as input texts.

        Raises:
            ValueError: If texts is empty.
        """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vectors.

        This is used by the vector store to configure the collection
        with the correct vector size before upserting chunks.

        Returns:
            Integer dimension of the embedding vectors.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name or identifier of the underlying model.

        Used for logging, MLflow tracking, and evaluation reports
        to identify which model produced the embeddings.

        Returns:
            String identifier of the model.
        """

    def embed_chunks(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Convenience method — validates input then delegates to embed_batch.

        Concrete implementations get this for free. Override only if
        provider-specific validation is needed.

        Args:
            texts: List of texts to embed.
            batch_size: Batch size passed to embed_batch.

        Returns:
            List of embedding vectors.

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts")
        return self.embed_batch(texts, batch_size=batch_size)
