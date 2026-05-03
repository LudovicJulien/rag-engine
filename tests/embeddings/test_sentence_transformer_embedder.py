# tests/embeddings/test_sentence_transformer_embedder.py
from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder


@pytest.fixture
def mock_embedder() -> Generator[SentenceTransformerEmbedder, None, None]:
    """Create a SentenceTransformerEmbedder with a mocked model."""
    with patch(
        "src.embeddings.sentence_transformer_embedder.SentenceTransformer"
    ) as mock_cls:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_cls.return_value = mock_model
        embedder = SentenceTransformerEmbedder(
            model_name="intfloat/multilingual-e5-large"
        )
        embedder._model = mock_model
        yield embedder


class TestSentenceTransformerEmbedderInit:
    """Tests for SentenceTransformerEmbedder initialization."""

    def test_default_model_name(
        self, mock_embedder: SentenceTransformerEmbedder
    ) -> None:
        assert mock_embedder.model_name == "intfloat/multilingual-e5-large"

    def test_embedding_dim_from_model(
        self, mock_embedder: SentenceTransformerEmbedder
    ) -> None:
        assert mock_embedder.embedding_dim == 1024

    def test_embedding_dim_is_positive(
        self, mock_embedder: SentenceTransformerEmbedder
    ) -> None:
        assert mock_embedder.embedding_dim > 0


class TestSentenceTransformerEmbedderEmbedText:
    """Tests for embed_text()."""

    def test_embed_text_returns_list_of_floats(
        self, mock_embedder: SentenceTransformerEmbedder
    ) -> None:
        mock_model = MagicMock(wraps=mock_embedder._model)
        mock_embedder._model = mock_model
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        result = mock_embedder.embed_text("Hello world")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_embed_text_returns_correct_vector(
        self, mock_embedder: SentenceTransformerEmbedder
    ) -> None:
        mock_model = MagicMock()
        mock_embedder._model = mock_model
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        result = mock_embedder.embed_text("Hello world")
        assert result == pytest.approx([0.1, 0.2, 0.3])

    def test_embed_text_empty_raises(
        self, mock_embedder: SentenceTransformerEmbedder
    ) -> None:
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            mock_embedder.embed_text("")

    def test_embed_text_calls_model_encode(
        self, mock_embedder: SentenceTransformerEmbedder
    ) -> None:
        mock_model = MagicMock()
        mock_embedder._model = mock_model
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_embedder.embed_text("Hello world")
        mock_model.encode.assert_called_once()


class TestSentenceTransformerEmbedderEmbedBatch:
    """Tests for embed_batch()."""

    def test_embed_batch_returns_list_of_vectors(
        self, mock_embedder: SentenceTransformerEmbedder
    ) -> None:
        mock_model = MagicMock()
        mock_embedder._model = mock_model
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = mock_embedder.embed_batch(["text one", "text two"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(v, list) for v in result)

    def test_embed_batch_preserves_order(
        self, mock_embedder: SentenceTransformerEmbedder
    ) -> None:
        """Order of returned vectors must match order of input texts."""
        mock_model = MagicMock()
        mock_embedder._model = mock_model
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        result = mock_embedder.embed_batch(["a", "b", "c"])
        assert len(result) == 3
        assert result[0] == pytest.approx([0.1, 0.2])
        assert result[1] == pytest.approx([0.3, 0.4])
        assert result[2] == pytest.approx([0.5, 0.6])

    def test_embed_batch_empty_raises(
        self, mock_embedder: SentenceTransformerEmbedder
    ) -> None:
        with pytest.raises(ValueError, match="Cannot embed an empty list"):
            mock_embedder.embed_batch([])

    def test_embed_batch_uses_batch_size(
        self, mock_embedder: SentenceTransformerEmbedder
    ) -> None:
        """batch_size parameter is passed to the underlying model."""
        mock_model = MagicMock()
        mock_embedder._model = mock_model
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        mock_embedder.embed_batch(["text"], batch_size=16)
        call_kwargs = mock_model.encode.call_args.kwargs
        assert call_kwargs["batch_size"] == 16


class TestSentenceTransformerEmbedderEmbedQuery:
    """Tests for embed_query() and query prefix behavior."""

    def test_embed_query_empty_raises(
        self, mock_embedder: SentenceTransformerEmbedder
    ) -> None:
        with pytest.raises(ValueError, match="Cannot embed empty query"):
            mock_embedder.embed_query("")

    def test_embed_query_adds_e5_prefix(
        self, mock_embedder: SentenceTransformerEmbedder
    ) -> None:
        """e5 models require 'query: ' prefix for queries."""
        mock_model = MagicMock()
        mock_embedder._model = mock_model
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_embedder.embed_query("meilleur quartier à Montréal")
        call_args = mock_model.encode.call_args.args[0]
        assert call_args.startswith("query: ")

    def test_embed_query_no_prefix_for_other_models(self) -> None:
        """Non-e5 models should not have query prefix added."""
        with patch(
            "src.embeddings.sentence_transformer_embedder.SentenceTransformer"
        ) as mock_cls:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
            mock_cls.return_value = mock_model

            embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
            embedder._model = mock_model
            embedder.embed_query("some query")

            call_args = mock_model.encode.call_args.args[0]
            assert not call_args.startswith("query: ")


class TestSentenceTransformerEmbedderCacheKey:
    """Tests for get_cache_key()."""

    def test_cache_key_is_string(
        self, mock_embedder: SentenceTransformerEmbedder
    ) -> None:
        assert isinstance(mock_embedder.get_cache_key(), str)

    def test_cache_key_is_deterministic(
        self, mock_embedder: SentenceTransformerEmbedder
    ) -> None:
        """Same model always produces same cache key."""
        assert mock_embedder.get_cache_key() == mock_embedder.get_cache_key()

    def test_different_models_have_different_cache_keys(self) -> None:
        """Different models must have different cache keys."""
        with patch(
            "src.embeddings.sentence_transformer_embedder.SentenceTransformer"
        ) as mock_cls:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_cls.return_value = mock_model

            embedder1 = SentenceTransformerEmbedder(model_name="model-a")
            embedder2 = SentenceTransformerEmbedder(model_name="model-b")

            assert embedder1.get_cache_key() != embedder2.get_cache_key()
