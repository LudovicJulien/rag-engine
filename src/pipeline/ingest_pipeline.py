# src/pipeline/ingest_pipeline.py
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

from src.embeddings.bm25_embedder import BM25SparseEmbedder
from src.embeddings.hybrid_embedder import HybridEmbedder
from src.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.ingestion.json_pipeline import JSONChunkIngestionPipeline
from src.vector_store.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestResult:
    source: str
    collection_name: str
    chunks_loaded: int
    chunks_upserted: int
    chunks_failed: int
    bm25_cache_path: str


class IngestionPipeline:
    """End-to-end ingestion orchestrator.

    Runs the full ingestion sequence in order:
    1. Load and validate chunks from a pre-chunked JSON file.
    2. Fit BM25 on the corpus texts.
    3. Serialize the fitted BM25 model to disk (pickle).
    4. Ensure the Qdrant collection exists; create it if not.
    5. Embed all chunks via HybridEmbedder (dense + sparse).
    6. Upsert chunks + embeddings into the vector store.

    Example::

        pipeline = IngestionPipeline(
            dense=SentenceTransformerEmbedder(),
            sparse=BM25SparseEmbedder(),
            vector_store=QdrantVectorStore(...),
            collection_name="rag-collection",
        )
        result = pipeline.run("data/chunks.json")
    """

    def __init__(
        self,
        dense: SentenceTransformerEmbedder,
        sparse: BM25SparseEmbedder,
        vector_store: VectorStore,
        collection_name: str,
        batch_size: int = 32,
        bm25_cache_dir: str | Path = ".bm25_cache",
        alpha: float = 0.5,
    ) -> None:
        """Initialize the pipeline.

        Args:
            dense: Dense embedding model (SentenceTransformer).
            sparse: Sparse BM25 embedder — must NOT be pre-fitted; it will be
                fitted inside :meth:`run` on the ingested corpus.
            vector_store: Configured vector store backend.
            collection_name: Qdrant collection name (used in :class:`IngestResult`).
            batch_size: Batch size passed to :meth:`HybridEmbedder.embed_batch`.
            bm25_cache_dir: Directory where the fitted BM25 pickle is stored.
            alpha: Dense-vs-sparse weight forwarded to :class:`HybridEmbedder`.

        Raises:
            ValueError: If *alpha* is not in [0.0, 1.0] or *batch_size* < 1.
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self._dense = dense
        self._sparse = sparse
        self._vector_store = vector_store
        self._collection_name = collection_name
        self._batch_size = batch_size
        self._bm25_cache_dir = Path(bm25_cache_dir)
        self._alpha = alpha

    def run(self, source: str | Path) -> IngestResult:
        """Execute the full ingestion pipeline.

        Args:
            source: Path to a pre-chunked JSON file.

        Returns:
            :class:`IngestResult` with chunk counts and BM25 cache path.

        Raises:
            FileNotFoundError: If the source file does not exist.
            ValueError: If the source file is invalid or yields no chunks.
            ConnectionError: If the vector store is unreachable.
        """
        source = Path(source)
        logger.info("Starting ingestion from %s", source)

        chunks = JSONChunkIngestionPipeline().ingest(source)
        logger.info("Loaded %d chunks", len(chunks))

        texts = [chunk.text for chunk in chunks]
        self._sparse.fit(texts)
        logger.info(
            "Fitted BM25 on %d texts (vocab_size=%d)",
            len(texts),
            self._sparse.embedding_dim,
        )

        bm25_cache_path = self._save_bm25(source)

        embedder = HybridEmbedder(
            dense=self._dense,
            sparse=self._sparse,
            alpha=self._alpha,
        )

        if not self._vector_store.collection_exists():
            self._vector_store.create_collection(dense_dim=embedder.dense_dim)
            logger.info(
                "Created collection '%s' (dense_dim=%d)",
                self._collection_name,
                embedder.dense_dim,
            )

        logger.info(
            "Embedding %d chunks (batch_size=%d)", len(chunks), self._batch_size
        )
        embeddings = embedder.embed_batch(texts, batch_size=self._batch_size)

        upsert_result = self._vector_store.upsert(chunks, embeddings)
        logger.info(
            "Upsert complete — upserted=%d failed=%d",
            upsert_result.upserted,
            upsert_result.failed,
        )

        return IngestResult(
            source=str(source),
            collection_name=self._collection_name,
            chunks_loaded=len(chunks),
            chunks_upserted=upsert_result.upserted,
            chunks_failed=upsert_result.failed,
            bm25_cache_path=bm25_cache_path,
        )

    def _save_bm25(self, source: Path) -> str:
        """Pickle the fitted BM25 model to *bm25_cache_dir*.

        Args:
            source: Ingestion source path — its stem is used as the filename.

        Returns:
            Resolved absolute path to the saved pickle file.
        """
        self._bm25_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._bm25_cache_dir / f"{source.stem}.pkl"
        with cache_path.open("wb") as f:
            pickle.dump(self._sparse, f)
        logger.info("Saved BM25 model to %s", cache_path.resolve())
        return str(cache_path.resolve())
