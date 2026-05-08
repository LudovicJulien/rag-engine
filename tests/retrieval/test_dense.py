# tests/retrieval/test_dense.py
from typing import Optional

import pytest

from src.embeddings.hybrid_embedder import HybridEmbedding
from src.retrieval.dense import DenseRetriever, DenseRetrieverConfig
from src.retrieval.retriever import Retriever
from src.shared.models import Chunk, ChunkMetadata, MetadataFilter
from src.vector_store.vector_store import SearchResult, UpsertResult, VectorStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedding() -> HybridEmbedding:
    return HybridEmbedding(
        dense=[0.1, 0.2, 0.3],
        sparse_indices=[0, 5],
        sparse_values=[0.8, 0.3],
        text="what is the capital of France?",
    )


def _make_result(score: float = 0.9) -> SearchResult:
    chunk = Chunk(
        chunk_id="chunk-001",
        parent_doc_id="doc-001",
        text="Paris is the capital of France.",
        chunk_index=0,
        total_chunks=1,
        metadata=ChunkMetadata(),
    )
    return SearchResult(chunk=chunk, score=score)


class FakeVectorStore(VectorStore):
    """Minimal in-memory test double for VectorStore."""

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self._results = results or []
        self.last_search_call: dict[str, int | float | None] = {}
        self.last_filters: Optional[list[MetadataFilter]] = None

    def create_collection(self, dense_dim: int) -> None:
        return None

    def delete_collection(self) -> None:
        return None

    def collection_exists(self) -> bool:
        return True

    def upsert(
        self, chunks: list[Chunk], embeddings: list[HybridEmbedding]
    ) -> UpsertResult:
        return UpsertResult(upserted=len(chunks))

    def search(
        self,
        query_embedding: HybridEmbedding,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[list[MetadataFilter]] = None,
    ) -> list[SearchResult]:
        self.last_search_call = {
            "top_k": top_k,
            "score_threshold": score_threshold,
        }
        self.last_filters = filters
        return self._results

    def health_check(self) -> bool:
        return True

    def count(self) -> int:
        return len(self._results)

    def delete_chunk_by_id(self, chunk_id: str) -> bool:
        return False

    def create_payload_index(self, field_name: str) -> None:
        return None

    def scroll_all_chunks(self, batch_size: int = 100) -> list[Chunk]:
        return []


# ---------------------------------------------------------------------------
# DenseRetrieverConfig
# ---------------------------------------------------------------------------


class TestDenseRetrieverConfig:
    def test_defaults_are_valid(self) -> None:
        config = DenseRetrieverConfig()
        assert config.top_k == 5
        assert config.score_threshold is None

    def test_raises_on_top_k_zero(self) -> None:
        with pytest.raises(ValueError, match="top_k must be"):
            DenseRetrieverConfig(top_k=0)

    def test_raises_on_top_k_negative(self) -> None:
        with pytest.raises(ValueError, match="top_k must be"):
            DenseRetrieverConfig(top_k=-1)

    def test_raises_on_score_threshold_below_zero(self) -> None:
        with pytest.raises(ValueError, match="score_threshold"):
            DenseRetrieverConfig(score_threshold=-0.1)

    def test_raises_on_score_threshold_above_one(self) -> None:
        with pytest.raises(ValueError, match="score_threshold"):
            DenseRetrieverConfig(score_threshold=1.1)

    def test_score_threshold_at_boundaries_is_valid(self) -> None:
        DenseRetrieverConfig(score_threshold=0.0)
        DenseRetrieverConfig(score_threshold=1.0)


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------


class TestDenseRetrieverContract:
    def test_is_retriever_subclass(self) -> None:
        store = FakeVectorStore()
        retriever = DenseRetriever(store)
        assert isinstance(retriever, Retriever)


class TestDenseRetrieverRetrieve:
    def test_returns_results_from_store(self) -> None:
        expected = [_make_result(0.9), _make_result(0.7)]
        retriever = DenseRetriever(FakeVectorStore(results=expected))
        results = retriever.retrieve(_make_embedding())
        assert results == expected

    def test_uses_config_top_k_by_default(self) -> None:
        store = FakeVectorStore()
        retriever = DenseRetriever(store, DenseRetrieverConfig(top_k=7))
        retriever.retrieve(_make_embedding())
        assert store.last_search_call["top_k"] == 7

    def test_per_call_top_k_overrides_config(self) -> None:
        store = FakeVectorStore()
        retriever = DenseRetriever(store, DenseRetrieverConfig(top_k=7))
        retriever.retrieve(_make_embedding(), top_k=3)
        assert store.last_search_call["top_k"] == 3

    def test_uses_config_score_threshold_by_default(self) -> None:
        store = FakeVectorStore()
        retriever = DenseRetriever(store, DenseRetrieverConfig(score_threshold=0.5))
        retriever.retrieve(_make_embedding())
        assert store.last_search_call["score_threshold"] == pytest.approx(0.5)

    def test_per_call_score_threshold_overrides_config(self) -> None:
        store = FakeVectorStore()
        retriever = DenseRetriever(store, DenseRetrieverConfig(score_threshold=0.5))
        retriever.retrieve(_make_embedding(), score_threshold=0.8)
        assert store.last_search_call["score_threshold"] == pytest.approx(0.8)

    def test_score_threshold_none_passed_to_store_when_unconfigured(self) -> None:
        store = FakeVectorStore()
        retriever = DenseRetriever(store)
        retriever.retrieve(_make_embedding())
        assert store.last_search_call["score_threshold"] is None

    def test_returns_empty_list_when_store_returns_nothing(self) -> None:
        retriever = DenseRetriever(FakeVectorStore(results=[]))
        results = retriever.retrieve(_make_embedding())
        assert results == []

    def test_results_are_forwarded_without_modification(self) -> None:
        expected = [_make_result(0.95)]
        retriever = DenseRetriever(FakeVectorStore(results=expected))
        results = retriever.retrieve(_make_embedding())
        assert results[0].score == pytest.approx(0.95)
        assert results[0].chunk.text == "Paris is the capital of France."

    def test_filters_none_by_default(self) -> None:
        store = FakeVectorStore()
        DenseRetriever(store).retrieve(_make_embedding())
        assert store.last_filters is None

    def test_filters_forwarded_to_store(self) -> None:
        store = FakeVectorStore()
        f = MetadataFilter(field="metadata.language", value="fr")
        DenseRetriever(store).retrieve(_make_embedding(), filters=[f])
        assert store.last_filters == [f]

    def test_multiple_filters_forwarded(self) -> None:
        store = FakeVectorStore()
        filters = [
            MetadataFilter(field="metadata.language", value="fr"),
            MetadataFilter(field="parent_doc_id", value="doc-42"),
        ]
        DenseRetriever(store).retrieve(_make_embedding(), filters=filters)
        assert store.last_filters == filters
