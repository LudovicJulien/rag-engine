# tests/embeddings/test_hybrid_embedder.py
from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embeddings.bm25_embedder import BM25SparseEmbedder
from src.embeddings.hybrid_embedder import HybridEmbedder, HybridEmbedding
from src.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder


@pytest.fixture
def fitted_bm25() -> BM25SparseEmbedder:
    """Create a fitted BM25SparseEmbedder."""
    embedder = BM25SparseEmbedder()
    corpus = [
        "Le Plateau est un quartier branché de Montréal",
        "Rosemont est un quartier familial avec des parcs",
        "Le Mile End est connu pour ses cafés et restaurants",
    ]
    embedder.fit(corpus)
    return embedder


@pytest.fixture
def mock_dense() -> Generator[SentenceTransformerEmbedder, None, None]:
    """Create a SentenceTransformerEmbedder with a mocked model."""
    with patch(
        "src.embeddings.sentence_transformer_embedder.SentenceTransformer"
    ) as mock_cls:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 4
        mock_cls.return_value = mock_model
        embedder = SentenceTransformerEmbedder(
            model_name="intfloat/multilingual-e5-large"
        )
        embedder._model = mock_model
        yield embedder


@pytest.fixture
def hybrid_embedder(
    mock_dense: SentenceTransformerEmbedder,
    fitted_bm25: BM25SparseEmbedder,
) -> HybridEmbedder:
    """Create a HybridEmbedder with mocked dense and real sparse."""
    return HybridEmbedder(dense=mock_dense, sparse=fitted_bm25, alpha=0.5)


class TestHybridEmbeddingContainer:
    """Tests for HybridEmbedding container class."""

    def test_dense_dim(self) -> None:
        result = HybridEmbedding(
            dense=[0.1, 0.2, 0.3],
            sparse_indices=[0, 2],
            sparse_values=[1.5, 2.3],
            text="hello",
        )
        assert result.dense_dim == 3

    def test_sparse_nnz(self) -> None:
        result = HybridEmbedding(
            dense=[0.1, 0.2],
            sparse_indices=[0, 2, 5],
            sparse_values=[1.5, 2.3, 0.8],
            text="hello",
        )
        assert result.sparse_nnz == 3

    def test_to_qdrant_payload_structure(self) -> None:
        result = HybridEmbedding(
            dense=[0.1, 0.2],
            sparse_indices=[0, 2],
            sparse_values=[1.5, 2.3],
        )
        payload = result.to_qdrant_payload()
        sparse = payload["sparse"]
        assert isinstance(sparse, dict)
        assert "indices" in sparse
        assert "values" in sparse

    def test_to_qdrant_payload_values(self) -> None:
        result = HybridEmbedding(
            dense=[0.1, 0.2],
            sparse_indices=[0, 2],
            sparse_values=[1.5, 2.3],
        )
        payload = result.to_qdrant_payload()
        assert payload["dense"] == [0.1, 0.2]
        assert payload["sparse"]["indices"] == [0, 2]  # type: ignore[index]
        assert payload["sparse"]["values"] == [1.5, 2.3]  # type: ignore[index]

    def test_empty_sparse(self) -> None:
        """HybridEmbedding with no sparse matches (all zeros) is valid."""
        result = HybridEmbedding(
            dense=[0.1, 0.2],
            sparse_indices=[],
            sparse_values=[],
        )
        assert result.sparse_nnz == 0


class TestHybridEmbedderInit:
    """Tests for HybridEmbedder initialization."""

    def test_valid_alpha(
        self,
        mock_dense: SentenceTransformerEmbedder,
        fitted_bm25: BM25SparseEmbedder,
    ) -> None:
        embedder = HybridEmbedder(dense=mock_dense, sparse=fitted_bm25, alpha=0.7)
        assert embedder.alpha == pytest.approx(0.7)

    def test_alpha_zero_valid(
        self,
        mock_dense: SentenceTransformerEmbedder,
        fitted_bm25: BM25SparseEmbedder,
    ) -> None:
        embedder = HybridEmbedder(dense=mock_dense, sparse=fitted_bm25, alpha=0.0)
        assert embedder.alpha == pytest.approx(0.0)

    def test_alpha_one_valid(
        self,
        mock_dense: SentenceTransformerEmbedder,
        fitted_bm25: BM25SparseEmbedder,
    ) -> None:
        embedder = HybridEmbedder(dense=mock_dense, sparse=fitted_bm25, alpha=1.0)
        assert embedder.alpha == pytest.approx(1.0)

    def test_alpha_above_one_raises(
        self,
        mock_dense: SentenceTransformerEmbedder,
        fitted_bm25: BM25SparseEmbedder,
    ) -> None:
        with pytest.raises(ValueError, match="alpha must be between"):
            HybridEmbedder(dense=mock_dense, sparse=fitted_bm25, alpha=1.5)

    def test_alpha_negative_raises(
        self,
        mock_dense: SentenceTransformerEmbedder,
        fitted_bm25: BM25SparseEmbedder,
    ) -> None:
        with pytest.raises(ValueError, match="alpha must be between"):
            HybridEmbedder(dense=mock_dense, sparse=fitted_bm25, alpha=-0.1)

    def test_unfitted_sparse_raises(
        self,
        mock_dense: SentenceTransformerEmbedder,
    ) -> None:
        unfitted_bm25 = BM25SparseEmbedder()
        with pytest.raises(RuntimeError, match="must be fitted"):
            HybridEmbedder(dense=mock_dense, sparse=unfitted_bm25)

    def test_dense_dim_property(
        self,
        hybrid_embedder: HybridEmbedder,
    ) -> None:
        assert hybrid_embedder.dense_dim == 4

    def test_sparse_dim_property(
        self,
        hybrid_embedder: HybridEmbedder,
        fitted_bm25: BM25SparseEmbedder,
    ) -> None:
        assert hybrid_embedder.sparse_dim == fitted_bm25.embedding_dim

    def test_model_name_contains_alpha(
        self,
        hybrid_embedder: HybridEmbedder,
    ) -> None:
        assert "0.5" in hybrid_embedder.model_name

    def test_model_name_contains_dense_name(
        self,
        hybrid_embedder: HybridEmbedder,
    ) -> None:
        assert "multilingual-e5-large" in hybrid_embedder.model_name


class TestHybridEmbedderEmbedText:
    """Tests for HybridEmbedder.embed_text()."""

    def test_embed_text_returns_hybrid_embedding(
        self,
        hybrid_embedder: HybridEmbedder,
        mock_dense: SentenceTransformerEmbedder,
    ) -> None:
        mock_dense._model.encode.return_value = np.array(  # type: ignore[attr-defined]
            [0.1, 0.2, 0.3, 0.4]
        )
        result = hybrid_embedder.embed_text("Plateau branché")
        assert isinstance(result, HybridEmbedding)

    def test_embed_text_dense_vector_correct_dim(
        self,
        hybrid_embedder: HybridEmbedder,
        mock_dense: SentenceTransformerEmbedder,
    ) -> None:
        mock_dense._model.encode.return_value = np.array(  # type: ignore[attr-defined]
            [0.1, 0.2, 0.3, 0.4]
        )
        result = hybrid_embedder.embed_text("Plateau branché")
        assert result.dense_dim == 4

    def test_embed_text_sparse_indices_and_values_same_length(
        self,
        hybrid_embedder: HybridEmbedder,
        mock_dense: SentenceTransformerEmbedder,
    ) -> None:
        mock_dense._model.encode.return_value = np.array(  # type: ignore[attr-defined]
            [0.1, 0.2, 0.3, 0.4]
        )
        result = hybrid_embedder.embed_text("Plateau branché")
        assert len(result.sparse_indices) == len(result.sparse_values)

    def test_embed_text_preserves_input_text(
        self,
        hybrid_embedder: HybridEmbedder,
        mock_dense: SentenceTransformerEmbedder,
    ) -> None:
        mock_dense._model.encode.return_value = np.array(  # type: ignore[attr-defined]
            [0.1, 0.2, 0.3, 0.4]
        )
        result = hybrid_embedder.embed_text("Plateau branché")
        assert result.text == "Plateau branché"

    def test_embed_text_empty_raises(
        self,
        hybrid_embedder: HybridEmbedder,
    ) -> None:
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            hybrid_embedder.embed_text("")

    def test_embed_text_unknown_term_sparse_all_zero(
        self,
        hybrid_embedder: HybridEmbedder,
        mock_dense: SentenceTransformerEmbedder,
    ) -> None:
        mock_dense._model.encode.return_value = np.array(  # type: ignore[attr-defined]
            [0.1, 0.2, 0.3, 0.4]
        )
        result = hybrid_embedder.embed_text("xyzunknownterm")
        assert result.sparse_nnz == 0


class TestHybridEmbedderEmbedQuery:
    """Tests for HybridEmbedder.embed_query()."""

    def test_embed_query_returns_hybrid_embedding(
        self,
        hybrid_embedder: HybridEmbedder,
        mock_dense: SentenceTransformerEmbedder,
    ) -> None:
        mock_dense._model.encode.return_value = np.array(  # type: ignore[attr-defined]
            [0.1, 0.2, 0.3, 0.4]
        )
        result = hybrid_embedder.embed_query("Plateau branché")
        assert isinstance(result, HybridEmbedding)

    def test_embed_query_empty_raises(
        self,
        hybrid_embedder: HybridEmbedder,
    ) -> None:
        with pytest.raises(ValueError, match="Cannot embed empty query"):
            hybrid_embedder.embed_query("")

    def test_embed_query_adds_e5_prefix_to_dense(
        self,
        hybrid_embedder: HybridEmbedder,
        mock_dense: SentenceTransformerEmbedder,
    ) -> None:
        """embed_query uses embed_query() on dense — adds e5 prefix."""
        mock_dense._model.encode.return_value = np.array(  # type: ignore[attr-defined]
            [0.1, 0.2, 0.3, 0.4]
        )
        hybrid_embedder.embed_query("Plateau branché")
        model = mock_dense._model
        call_args = model.encode.call_args.args[0]  # type: ignore[attr-defined]
        assert call_args.startswith("query: ")


class TestHybridEmbedderEmbedBatch:
    """Tests for HybridEmbedder.embed_batch()."""

    def test_embed_batch_returns_list(
        self,
        hybrid_embedder: HybridEmbedder,
        mock_dense: SentenceTransformerEmbedder,
    ) -> None:
        mock_dense._model.encode.return_value = np.array(  # type: ignore[attr-defined]
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        )
        result = hybrid_embedder.embed_batch(["Plateau", "Rosemont"])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_embed_batch_all_hybrid_embeddings(
        self,
        hybrid_embedder: HybridEmbedder,
        mock_dense: SentenceTransformerEmbedder,
    ) -> None:
        mock_dense._model.encode.return_value = np.array(  # type: ignore[attr-defined]
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        )
        result = hybrid_embedder.embed_batch(["Plateau", "Rosemont"])
        assert all(isinstance(r, HybridEmbedding) for r in result)

    def test_embed_batch_preserves_order(
        self,
        hybrid_embedder: HybridEmbedder,
        mock_dense: SentenceTransformerEmbedder,
    ) -> None:
        mock_dense._model.encode.return_value = np.array(  # type: ignore[attr-defined]
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        )
        texts = ["Plateau branché", "Rosemont familial"]
        result = hybrid_embedder.embed_batch(texts)
        assert result[0].text == texts[0]
        assert result[1].text == texts[1]

    def test_embed_batch_empty_raises(
        self,
        hybrid_embedder: HybridEmbedder,
    ) -> None:
        with pytest.raises(ValueError, match="Cannot embed an empty list"):
            hybrid_embedder.embed_batch([])


class TestHybridEmbedderToSparseFormat:
    """Tests for HybridEmbedder._to_sparse_format()."""

    def test_extracts_nonzero_indices(self) -> None:
        vector = [0.0, 1.5, 0.0, 2.3, 0.0]
        indices, _ = HybridEmbedder._to_sparse_format(vector)
        assert indices == [1, 3]

    def test_extracts_nonzero_values(self) -> None:
        vector = [0.0, 1.5, 0.0, 2.3, 0.0]
        _, values = HybridEmbedder._to_sparse_format(vector)
        assert values == [1.5, 2.3]

    def test_all_zeros_returns_empty(self) -> None:
        vector = [0.0, 0.0, 0.0]
        indices, values = HybridEmbedder._to_sparse_format(vector)
        assert indices == []
        assert values == []

    def test_no_zeros_returns_all(self) -> None:
        vector = [1.0, 2.0, 3.0]
        indices, values = HybridEmbedder._to_sparse_format(vector)
        assert indices == [0, 1, 2]
        assert values == [1.0, 2.0, 3.0]

    def test_indices_and_values_same_length(self) -> None:
        vector = [0.0, 1.5, 0.0, 2.3, 0.0, 0.8]
        indices, values = HybridEmbedder._to_sparse_format(vector)
        assert len(indices) == len(values)


class TestHybridEmbedderCacheKey:
    """Tests for HybridEmbedder.get_cache_key()."""

    def test_cache_key_is_string(
        self,
        hybrid_embedder: HybridEmbedder,
    ) -> None:
        assert isinstance(hybrid_embedder.get_cache_key(), str)

    def test_cache_key_is_deterministic(
        self,
        hybrid_embedder: HybridEmbedder,
    ) -> None:
        assert hybrid_embedder.get_cache_key() == hybrid_embedder.get_cache_key()

    def test_different_alpha_different_cache_key(
        self,
        mock_dense: SentenceTransformerEmbedder,
        fitted_bm25: BM25SparseEmbedder,
    ) -> None:
        embedder1 = HybridEmbedder(dense=mock_dense, sparse=fitted_bm25, alpha=0.3)
        embedder2 = HybridEmbedder(dense=mock_dense, sparse=fitted_bm25, alpha=0.7)
        assert embedder1.get_cache_key() != embedder2.get_cache_key()
