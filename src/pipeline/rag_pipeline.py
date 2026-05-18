# src/pipeline/rag_pipeline.py
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.embeddings.hybrid_embedder import HybridEmbedder
from src.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.generation.factory import get_generator
from src.generation.generator import LLMGenerator
from src.pipeline.config import Settings
from src.pipeline.ingest_pipeline import IngestionPipeline
from src.retrieval.dense import DenseRetriever, DenseRetrieverConfig
from src.retrieval.retriever import Retriever
from src.shared.models import MetadataFilter
from src.vector_store.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RAGResult:
    query: str
    answer: str
    sources: list[str]
    detected_language: str
    model: str
    chunks_retrieved: int
    tokens_used: int | None


class RAGPipeline:
    """End-to-end RAG query orchestrator.

    Wires together embedding, retrieval, and generation layers into a single
    callable pipeline.  Build it via :meth:`build` — never call the constructor
    directly in application code.

    Example::

        pipeline = RAGPipeline.build(settings)
        result = pipeline.query("Quel est le meilleur quartier de Montréal ?")
    """

    def __init__(
        self,
        embedder: HybridEmbedder,
        retriever: Retriever,
        generator: LLMGenerator,
    ) -> None:
        self._embedder = embedder
        self._retriever = retriever
        self._generator = generator

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, settings: Settings) -> RAGPipeline:
        """Load BM25 from disk and wire all layers from *settings*.

        Instantiation order follows data-flow dependencies:
        1. Load fitted :class:`~src.embeddings.bm25_embedder.BM25SparseEmbedder`
           from :attr:`~Settings.bm25_cache_path`.
        2. Instantiate
           :class:`~src.embeddings.sentence_transformer_embedder.SentenceTransformerEmbedder`.
        3. Wrap both in :class:`~src.embeddings.hybrid_embedder.HybridEmbedder`.
        4. Connect to Qdrant via
           :class:`~src.vector_store.qdrant_store.QdrantVectorStore`.
        5. Wrap the store in :class:`~src.retrieval.dense.DenseRetriever`.
        6. Build the :class:`~src.generation.generator.LLMGenerator` via factory.

        Args:
            settings: Populated application settings instance.

        Returns:
            A ready-to-query :class:`RAGPipeline`.

        Raises:
            FileNotFoundError: If the BM25 pickle does not exist at
                :attr:`~Settings.bm25_cache_path`.
            ConnectionError: If the Qdrant server is unreachable.
        """
        logger.info("Building RAGPipeline")

        sparse = IngestionPipeline.load_bm25(settings.bm25_cache_path)
        logger.info(
            "Loaded BM25 from %s (vocab_size=%d)",
            settings.bm25_cache_path,
            sparse.embedding_dim,
        )

        dense = SentenceTransformerEmbedder(model_name=settings.embedding_model)
        logger.info(
            "Dense embedder ready: %s (dim=%d)",
            settings.embedding_model,
            dense.embedding_dim,
        )

        embedder = HybridEmbedder(dense=dense, sparse=sparse)

        vector_store = QdrantVectorStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.collection_name,
        )
        logger.info(
            "Qdrant connected: %s:%d / collection=%s",
            settings.qdrant_host,
            settings.qdrant_port,
            settings.collection_name,
        )

        retriever = DenseRetriever(
            vector_store=vector_store,
            config=DenseRetrieverConfig(
                top_k=settings.top_k,
                score_threshold=settings.score_threshold,
            ),
        )

        generator = get_generator(settings)
        logger.info(
            "RAGPipeline ready | provider=%s model=%s",
            generator.provider_name,
            generator.model_name,
        )

        return cls(embedder=embedder, retriever=retriever, generator=generator)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filters: list[MetadataFilter] | None = None,
    ) -> RAGResult:
        """Execute the full RAG pipeline for a user query.

        Steps: embed query → retrieve chunks → generate answer.

        Args:
            query: The user's question.  Must be non-empty.
            top_k: Per-call override for the number of retrieved chunks.
                ``None`` falls back to the retriever's configured default.
            score_threshold: Per-call override for the minimum relevance score.
                ``None`` falls back to the retriever's configured default.
            filters: Optional equality filters applied before retrieval.
                ``None`` disables payload filtering.

        Returns:
            A :class:`RAGResult` with the generated answer and metadata.

        Raises:
            ValueError: If *query* is empty or whitespace-only.
            RuntimeError: If retrieval returns no results.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        logger.debug("RAGPipeline.query | query=%r", query)

        query_embedding = self._embedder.embed_query(query)

        search_results = self._retriever.retrieve(
            query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters,
        )

        if not search_results:
            raise RuntimeError(
                "No chunks retrieved for the given query. "
                "Consider lowering score_threshold or verifying "
                "the collection is populated."
            )

        chunks = [r.chunk for r in search_results]
        logger.info("Retrieved %d chunk(s)", len(chunks))

        gen_result = self._generator.generate(query, chunks)

        return RAGResult(
            query=query,
            answer=gen_result.answer,
            sources=gen_result.sources,
            detected_language=gen_result.detected_language,
            model=gen_result.model,
            chunks_retrieved=len(chunks),
            tokens_used=gen_result.tokens_used,
        )
