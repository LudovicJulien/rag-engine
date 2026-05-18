# tests/pipeline/test_rag_pipeline_integration.py
from __future__ import annotations

from pathlib import Path
from typing import Generator

import httpx
import ollama
import pytest
from qdrant_client import QdrantClient

from src.embeddings.bm25_embedder import BM25SparseEmbedder
from src.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.pipeline.config import Settings
from src.pipeline.ingest_pipeline import IngestionPipeline, IngestResult
from src.pipeline.rag_pipeline import RAGPipeline, RAGResult
from src.vector_store.qdrant_store import QdrantVectorStore

_QDRANT_HOST = "localhost"
_QDRANT_PORT = 6333
_OLLAMA_URL = "http://localhost:11434"
_OLLAMA_MODEL = "gemma3:4b"
_TEST_COLLECTION = "rag-pipeline-integration-test"
_DEMO_DATA = Path(__file__).parent.parent.parent / "demo_data" / "sample_chunks.json"


# ── Service availability fixtures ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def qdrant_available() -> None:
    """Skip the module if Qdrant is not reachable."""
    try:
        QdrantClient(host=_QDRANT_HOST, port=_QDRANT_PORT, timeout=2).get_collections()
    except Exception:
        pytest.skip(f"Qdrant not reachable at {_QDRANT_HOST}:{_QDRANT_PORT}")


@pytest.fixture(scope="module")
def ollama_available() -> None:
    """Skip the module if Ollama or the required model is not available."""
    client = ollama.Client(host=_OLLAMA_URL)
    try:
        model_names = [m.model for m in client.list().models]
    except (ConnectionError, httpx.ConnectError, httpx.HTTPError):
        pytest.skip(f"Ollama not reachable at {_OLLAMA_URL}")
    if _OLLAMA_MODEL not in model_names:
        pytest.skip(
            f"Model '{_OLLAMA_MODEL}' not pulled — run: ollama pull {_OLLAMA_MODEL}"
        )


# ── Ingestion fixture — runs once, cleans up after the module ─────────────────


@pytest.fixture(scope="module")
def ingest_result(
    qdrant_available: None,
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[IngestResult, None, None]:
    """Ingest demo_data once; delete the test collection on teardown."""
    bm25_dir = tmp_path_factory.mktemp("bm25")

    store = QdrantVectorStore(
        host=_QDRANT_HOST,
        port=_QDRANT_PORT,
        collection_name=_TEST_COLLECTION,
    )
    if store.collection_exists():
        store.delete_collection()

    pipeline = IngestionPipeline(
        dense=SentenceTransformerEmbedder(model_name="intfloat/multilingual-e5-large"),
        sparse=BM25SparseEmbedder(),
        vector_store=store,
        collection_name=_TEST_COLLECTION,
        bm25_cache_dir=bm25_dir,
    )
    result = pipeline.run(_DEMO_DATA)

    yield result

    if store.collection_exists():
        store.delete_collection()


# ── RAGPipeline fixture — built once on top of the ingested collection ─────────


@pytest.fixture(scope="module")
def rag_pipeline(
    ingest_result: IngestResult,
    ollama_available: None,
) -> RAGPipeline:
    """Build a live RAGPipeline pointing at the test collection."""
    settings = Settings(
        qdrant_host=_QDRANT_HOST,
        qdrant_port=_QDRANT_PORT,
        collection_name=_TEST_COLLECTION,
        bm25_cache_path=Path(ingest_result.bm25_cache_path),
        llm_provider="ollama",
        llm_model=_OLLAMA_MODEL,
        llm_base_url=_OLLAMA_URL,
        top_k=3,
        score_threshold=0.0,
    )
    return RAGPipeline.build(settings)


# ── Module-scoped query results — one LLM call per query, reused across tests ──


@pytest.fixture(scope="module")
def fr_result(rag_pipeline: RAGPipeline) -> RAGResult:
    return rag_pipeline.query("Quel est le quartier branché de Montréal ?")


@pytest.fixture(scope="module")
def en_result(rag_pipeline: RAGPipeline) -> RAGResult:
    return rag_pipeline.query("What is a diverse neighbourhood in Montreal?")


# ── TestIngestionPipelineIntegration ──────────────────────────────────────────


class TestIngestionPipelineIntegration:
    """Full ingestion pipeline against a live Qdrant instance.

    Requires Qdrant running at localhost:6333.
    Skipped automatically when the server is unreachable.
    """

    def test_run_returns_ingest_result(self, ingest_result: IngestResult) -> None:
        assert isinstance(ingest_result, IngestResult)

    def test_chunks_loaded_matches_source_file(
        self, ingest_result: IngestResult
    ) -> None:
        assert ingest_result.chunks_loaded == 3

    def test_all_chunks_upserted_without_failure(
        self, ingest_result: IngestResult
    ) -> None:
        assert ingest_result.chunks_failed == 0
        assert ingest_result.chunks_upserted == 3

    def test_bm25_pickle_exists_on_disk(self, ingest_result: IngestResult) -> None:
        assert Path(ingest_result.bm25_cache_path).exists()

    def test_bm25_pickle_is_loadable(self, ingest_result: IngestResult) -> None:
        loaded = BM25SparseEmbedder.load(ingest_result.bm25_cache_path)
        assert loaded.is_fitted

    def test_qdrant_collection_exists_after_ingest(
        self, ingest_result: IngestResult
    ) -> None:
        store = QdrantVectorStore(
            host=_QDRANT_HOST,
            port=_QDRANT_PORT,
            collection_name=_TEST_COLLECTION,
        )
        assert store.collection_exists()

    def test_qdrant_collection_count_matches_chunks(
        self, ingest_result: IngestResult
    ) -> None:
        store = QdrantVectorStore(
            host=_QDRANT_HOST,
            port=_QDRANT_PORT,
            collection_name=_TEST_COLLECTION,
        )
        assert store.count() == ingest_result.chunks_upserted


# ── TestRAGPipelineQueryIntegration ───────────────────────────────────────────


class TestRAGPipelineQueryIntegration:
    """Full RAG pipeline against live Qdrant + Ollama.

    Requires Qdrant at localhost:6333 and Ollama at localhost:11434
    with gemma3:4b pulled. Both are skipped automatically when unavailable.
    """

    def test_fr_query_returns_rag_result(self, fr_result: RAGResult) -> None:
        assert isinstance(fr_result, RAGResult)

    def test_fr_answer_is_non_empty(self, fr_result: RAGResult) -> None:
        assert fr_result.answer.strip() != ""

    def test_fr_chunks_retrieved_positive(self, fr_result: RAGResult) -> None:
        assert fr_result.chunks_retrieved > 0

    def test_fr_sources_reference_ingested_chunk_ids(
        self, fr_result: RAGResult
    ) -> None:
        valid_ids = {"chunk-001", "chunk-002", "chunk-003"}
        assert set(fr_result.sources).issubset(valid_ids)

    def test_fr_sources_non_empty(self, fr_result: RAGResult) -> None:
        assert len(fr_result.sources) > 0

    def test_fr_detected_language_is_fr(self, fr_result: RAGResult) -> None:
        assert fr_result.detected_language == "fr"

    def test_fr_model_name_matches_config(self, fr_result: RAGResult) -> None:
        assert fr_result.model == _OLLAMA_MODEL

    def test_fr_query_preserved_verbatim_in_result(self, fr_result: RAGResult) -> None:
        assert fr_result.query == "Quel est le quartier branché de Montréal ?"

    def test_en_query_returns_rag_result(self, en_result: RAGResult) -> None:
        assert isinstance(en_result, RAGResult)

    def test_en_answer_is_non_empty(self, en_result: RAGResult) -> None:
        assert en_result.answer.strip() != ""

    def test_en_detected_language_is_en(self, en_result: RAGResult) -> None:
        assert en_result.detected_language == "en"

    def test_top_k_override_limits_chunks_retrieved(
        self, rag_pipeline: RAGPipeline
    ) -> None:
        result = rag_pipeline.query(
            "Quel quartier ?",
            top_k=1,
        )
        assert result.chunks_retrieved == 1
