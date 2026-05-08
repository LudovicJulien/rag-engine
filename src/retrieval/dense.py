# src/retrieval/dense.py
"""
Dense vector retrieval strategy.

Wraps ``VectorStore.search`` and exposes a focused interface for the
retrieval layer: top-k selection and optional score-threshold filtering.

Note on "dense" semantics
--------------------------
The underlying ``QdrantVectorStore`` already performs hybrid search
(dense HNSW + sparse BM25 fused via RRF) — this is intentional.
``DenseRetriever`` is named from the *pipeline* perspective: it is the
retrieval step that consumes a *dense* query embedding (as opposed to
future sparse-only or re-ranker stages).  The vector store handles the
fusion detail transparently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.embeddings.hybrid_embedder import HybridEmbedding
from src.retrieval.retriever import Retriever
from src.shared.models import MetadataFilter
from src.vector_store.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DenseRetrieverConfig:
    """Immutable hyperparameters for the dense retrieval step.

    Attributes:
        top_k: Maximum number of results to return.  Must be ≥ 1.
        score_threshold: Minimum RRF-fused relevance score accepted.
            Results below this value are discarded.  ``None`` disables
            filtering so every ANN candidate up to ``top_k`` is returned.
    """

    top_k: int = 5
    score_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError(f"top_k must be ≥ 1, got {self.top_k}")
        if self.score_threshold is not None and not (
            0.0 <= self.score_threshold <= 1.0
        ):
            raise ValueError(
                f"score_threshold must be in [0.0, 1.0], got {self.score_threshold}"
            )


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class DenseRetriever(Retriever):
    """Retrieve the top-k most relevant chunks for a query embedding.

    Delegates to the injected :class:`VectorStore` — no Qdrant-specific
    code lives here.  Any backend that satisfies the ``VectorStore``
    contract (Qdrant, Pinecone, in-memory test double …) can be swapped in
    without touching this class.

    Instances are stateless beyond their config and are safe to share
    across threads.

    Args:
        vector_store: Initialised backend implementing :class:`VectorStore`.
        config: Retrieval hyperparameters.  Defaults to ``top_k=5`` with
            no score filtering.

    Example::

        retriever = DenseRetriever(store, DenseRetrieverConfig(top_k=10))
        results = retriever.retrieve(query_embedding)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        config: DenseRetrieverConfig = DenseRetrieverConfig(),
    ) -> None:
        self._store = vector_store
        self._config = config
        logger.debug(
            "DenseRetriever ready | top_k=%d score_threshold=%s",
            config.top_k,
            config.score_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_embedding: HybridEmbedding,
        *,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filters: Optional[list[MetadataFilter]] = None,
    ) -> list[SearchResult]:
        """Return the most relevant chunks for *query_embedding*.

        Call-site overrides for ``top_k`` and ``score_threshold`` take
        precedence over the values baked into the config, enabling
        per-request tuning (e.g. RAGAS eval sweeps) without rebuilding
        the retriever.

        Pass ``score_threshold=0.0`` to explicitly disable filtering for a
        single call even when a non-zero default is configured.

        Args:
            query_embedding: Hybrid embedding of the user query produced by
                :class:`~src.embeddings.hybrid_embedder.HybridEmbedder`.
            top_k: Override ``config.top_k`` for this call only.
            score_threshold: Override ``config.score_threshold`` for this
                call only.
            filters: Optional equality filters applied to payload fields
                before scoring.  All conditions are ANDed together.
                ``None`` disables payload filtering.

        Returns:
            Ordered list of :class:`~src.vector_store.vector_store.SearchResult`,
            highest score first.  May be shorter than ``top_k`` when fewer
            results satisfy the score threshold.

        Raises:
            ValueError: Propagated from :class:`VectorStore` if ``top_k`` < 1.
            RuntimeError: Propagated on network or serialization errors.
        """
        effective_top_k: int
        if top_k is None:
            effective_top_k = self._config.top_k
        else:
            effective_top_k = top_k
        effective_threshold = (
            score_threshold
            if score_threshold is not None
            else self._config.score_threshold
        )

        logger.debug(
            "Dense retrieve | top_k=%d score_threshold=%s",
            effective_top_k,
            effective_threshold,
        )

        results = self._store.search(
            query_embedding=query_embedding,
            top_k=effective_top_k,
            score_threshold=effective_threshold,
            filters=filters,
        )

        logger.info(
            "Dense retrieve → %d result(s) | top_score=%s",
            len(results),
            f"{results[0].score:.4f}" if results else "n/a",
        )

        return results
