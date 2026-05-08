# tests/embeddings/test_integration.py
from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.embeddings.bm25_embedder import BM25SparseEmbedder
from src.embeddings.embedding_cache import EmbeddingCache
from src.embeddings.hybrid_embedder import HybridEmbedder, HybridEmbedding
from src.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.ingestion.json_pipeline import JSONChunkIngestionPipeline
from src.shared.models import Chunk

DEMO_DATA_PATH = (
    Path(__file__).parent.parent.parent / "demo_data" / "sample_chunks.json"
)


@pytest.fixture(scope="module")
def demo_chunks() -> list[Chunk]:
    """Load demo_data chunks once for all integration tests."""
    pipeline = JSONChunkIngestionPipeline()
    return pipeline.ingest(DEMO_DATA_PATH)


@pytest.fixture(scope="module")
def dense_embedder() -> SentenceTransformerEmbedder:
    """Load SentenceTransformer model once for all integration tests."""
    return SentenceTransformerEmbedder(model_name="intfloat/multilingual-e5-large")


@pytest.fixture(scope="module")
def sparse_embedder(demo_chunks: list[Chunk]) -> BM25SparseEmbedder:
    """Fit BM25 on demo corpus once for all integration tests."""
    embedder = BM25SparseEmbedder()
    corpus = [chunk.text for chunk in demo_chunks]
    embedder.fit(corpus)
    return embedder


@pytest.fixture(scope="module")
def hybrid_embedder(
    dense_embedder: SentenceTransformerEmbedder,
    sparse_embedder: BM25SparseEmbedder,
) -> HybridEmbedder:
    """Create HybridEmbedder once for all integration tests."""
    return HybridEmbedder(
        dense=dense_embedder,
        sparse=sparse_embedder,
        alpha=0.5,
    )


class TestSentenceTransformerEmbedderIntegration:
    """Integration tests for SentenceTransformerEmbedder on real data."""

    def test_embed_batch_returns_correct_count(
        self,
        dense_embedder: SentenceTransformerEmbedder,
        demo_chunks: list[Chunk],
    ) -> None:
        texts = [chunk.text for chunk in demo_chunks]
        vectors = dense_embedder.embed_batch(texts)
        assert len(vectors) == len(demo_chunks)

    def test_embed_batch_no_zero_vectors(
        self,
        dense_embedder: SentenceTransformerEmbedder,
        demo_chunks: list[Chunk],
    ) -> None:
        texts = [chunk.text for chunk in demo_chunks]
        vectors = dense_embedder.embed_batch(texts)
        for i, vector in enumerate(vectors):
            assert any(
                v != 0.0 for v in vector
            ), f"Chunk {i} produced an all-zero vector: {demo_chunks[i].text[:50]}"

    def test_embed_batch_consistent_dimensions(
        self,
        dense_embedder: SentenceTransformerEmbedder,
        demo_chunks: list[Chunk],
    ) -> None:
        texts = [chunk.text for chunk in demo_chunks]
        vectors = dense_embedder.embed_batch(texts)
        dims = {len(v) for v in vectors}
        assert len(dims) == 1, f"Inconsistent dimensions: {dims}"

    def test_embed_batch_dimension_matches_model(
        self,
        dense_embedder: SentenceTransformerEmbedder,
        demo_chunks: list[Chunk],
    ) -> None:
        texts = [chunk.text for chunk in demo_chunks]
        vectors = dense_embedder.embed_batch(texts)
        assert all(len(v) == dense_embedder.embedding_dim for v in vectors)

    def test_single_vs_batch_consistency(
        self,
        dense_embedder: SentenceTransformerEmbedder,
        demo_chunks: list[Chunk],
    ) -> None:
        chunk = demo_chunks[0]
        single = dense_embedder.embed_text(chunk.text)
        batch = dense_embedder.embed_batch([chunk.text])
        assert single == pytest.approx(batch[0], abs=1e-5)

    def test_embed_query_different_from_embed_text_for_e5(
        self,
        dense_embedder: SentenceTransformerEmbedder,
    ) -> None:
        """e5 models produce different vectors for query vs passage."""
        query = "meilleur quartier familial à Montréal"
        query_vector = dense_embedder.embed_query(query)
        text_vector = dense_embedder.embed_text(query)
        assert query_vector != text_vector

    def test_similar_texts_closer_than_dissimilar(
        self,
        dense_embedder: SentenceTransformerEmbedder,
    ) -> None:
        def cosine_similarity(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        similar_1 = dense_embedder.embed_text("Le Plateau est branché")
        similar_2 = dense_embedder.embed_text("Le Plateau est un quartier tendance")
        dissimilar = dense_embedder.embed_text("Les mathématiques sont abstraites")

        sim_score = cosine_similarity(similar_1, similar_2)
        dis_score = cosine_similarity(similar_1, dissimilar)

        assert sim_score > dis_score


class TestBM25SparseEmbedderIntegration:
    """Integration tests for BM25SparseEmbedder on real data."""

    def test_fit_builds_vocabulary_from_corpus(
        self,
        sparse_embedder: BM25SparseEmbedder,
    ) -> None:
        assert sparse_embedder.embedding_dim > 0
        assert sparse_embedder.is_fitted is True

    def test_embed_batch_returns_correct_count(
        self,
        sparse_embedder: BM25SparseEmbedder,
        demo_chunks: list[Chunk],
    ) -> None:
        texts = [chunk.text for chunk in demo_chunks]
        vectors = sparse_embedder.embed_batch(texts)
        assert len(vectors) == len(demo_chunks)

    def test_embed_batch_consistent_dimensions(
        self,
        sparse_embedder: BM25SparseEmbedder,
        demo_chunks: list[Chunk],
    ) -> None:
        texts = [chunk.text for chunk in demo_chunks]
        vectors = sparse_embedder.embed_batch(texts)
        dims = {len(v) for v in vectors}
        assert len(dims) == 1

    def test_embed_batch_dimension_matches_vocabulary(
        self,
        sparse_embedder: BM25SparseEmbedder,
        demo_chunks: list[Chunk],
    ) -> None:
        texts = [chunk.text for chunk in demo_chunks]
        vectors = sparse_embedder.embed_batch(texts)
        assert all(len(v) == sparse_embedder.embedding_dim for v in vectors)

    def test_corpus_terms_have_nonzero_scores(
        self,
        sparse_embedder: BM25SparseEmbedder,
        demo_chunks: list[Chunk],
    ) -> None:
        texts = [chunk.text for chunk in demo_chunks]
        vectors = sparse_embedder.embed_batch(texts)
        for i, vector in enumerate(vectors):
            assert any(
                v != 0.0 for v in vector
            ), f"Chunk {i} produced an all-zero sparse vector"

    def test_single_vs_batch_consistency(
        self,
        sparse_embedder: BM25SparseEmbedder,
        demo_chunks: list[Chunk],
    ) -> None:
        chunk = demo_chunks[0]
        single = sparse_embedder.embed_text(chunk.text)
        batch = sparse_embedder.embed_batch([chunk.text])
        assert single == batch[0]


class TestHybridEmbedderIntegration:
    """Integration tests for HybridEmbedder on real data."""

    def test_embed_batch_returns_hybrid_embeddings(
        self,
        hybrid_embedder: HybridEmbedder,
        demo_chunks: list[Chunk],
    ) -> None:
        texts = [chunk.text for chunk in demo_chunks]
        results = hybrid_embedder.embed_batch(texts)
        assert all(isinstance(r, HybridEmbedding) for r in results)

    def test_embed_batch_correct_count(
        self,
        hybrid_embedder: HybridEmbedder,
        demo_chunks: list[Chunk],
    ) -> None:
        texts = [chunk.text for chunk in demo_chunks]
        results = hybrid_embedder.embed_batch(texts)
        assert len(results) == len(demo_chunks)

    def test_embed_batch_dense_vectors_nonzero(
        self,
        hybrid_embedder: HybridEmbedder,
        demo_chunks: list[Chunk],
    ) -> None:
        texts = [chunk.text for chunk in demo_chunks]
        results = hybrid_embedder.embed_batch(texts)
        for result in results:
            assert any(v != 0.0 for v in result.dense)

    def test_embed_batch_sparse_indices_values_consistent(
        self,
        hybrid_embedder: HybridEmbedder,
        demo_chunks: list[Chunk],
    ) -> None:
        texts = [chunk.text for chunk in demo_chunks]
        results = hybrid_embedder.embed_batch(texts)
        for result in results:
            assert len(result.sparse_indices) == len(result.sparse_values)

    def test_qdrant_payload_valid_for_all_chunks(
        self,
        hybrid_embedder: HybridEmbedder,
        demo_chunks: list[Chunk],
    ) -> None:
        texts = [chunk.text for chunk in demo_chunks]
        results = hybrid_embedder.embed_batch(texts)
        for result in results:
            payload = result.to_qdrant_payload()
            assert "dense" in payload
            assert "sparse" in payload
            sparse = payload["sparse"]
            assert isinstance(sparse, dict)
            assert "indices" in sparse
            assert "values" in sparse

    def test_embed_query_returns_hybrid_embedding(
        self,
        hybrid_embedder: HybridEmbedder,
    ) -> None:
        result = hybrid_embedder.embed_query("meilleur quartier Montréal")
        assert isinstance(result, HybridEmbedding)
        assert len(result.dense) == hybrid_embedder.dense_dim


class TestEmbeddingCacheIntegration:
    """Integration tests for EmbeddingCache with real embeddings."""

    def test_cache_hit_returns_identical_vector(
        self,
        dense_embedder: SentenceTransformerEmbedder,
        tmp_path: Path,
    ) -> None:
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        text = "Le Plateau est branché"
        model_key = dense_embedder.get_cache_key()

        vector = dense_embedder.embed_text(text)
        cache.set(text=text, model_key=model_key, vector=vector)

        cached = cache.get(text=text, model_key=model_key)
        assert cached == pytest.approx(vector, abs=1e-6)

    def test_cache_avoids_reembedding(
        self,
        dense_embedder: SentenceTransformerEmbedder,
        tmp_path: Path,
    ) -> None:
        """hit_rate is computed from calls to get(), not set()
        the assertion relies on exactly one get() call after priming."""
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        text = "Le Plateau est branché"
        model_key = dense_embedder.get_cache_key()

        vector = dense_embedder.embed_text(text)
        cache.set(text=text, model_key=model_key, vector=vector)

        cache.get(text=text, model_key=model_key)
        assert cache.hit_rate == pytest.approx(1.0)

    def test_cache_miss_on_model_change(
        self,
        tmp_path: Path,
    ) -> None:
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        text = "Le Plateau est branché"

        cache.set(text=text, model_key="model-a", vector=[0.1, 0.2])
        result = cache.get(text=text, model_key="model-b")
        assert result is None

    def test_full_corpus_cache_roundtrip(
        self,
        dense_embedder: SentenceTransformerEmbedder,
        demo_chunks: list[Chunk],
        tmp_path: Path,
    ) -> None:
        cache = EmbeddingCache(cache_dir=tmp_path / "cache")
        model_key = dense_embedder.get_cache_key()
        texts = [chunk.text for chunk in demo_chunks]

        vectors = dense_embedder.embed_batch(texts)
        cache.set_batch(texts=texts, model_key=model_key, vectors=vectors)

        cached_vectors = cache.get_batch(texts=texts, model_key=model_key)

        assert all(v is not None for v in cached_vectors)
        assert cache.hit_rate == pytest.approx(1.0)
        assert cache.stats()["cached_count"] == len(demo_chunks)
