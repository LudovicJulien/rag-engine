# tests/embeddings/test_bm25_embedder.py
from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from src.embeddings.bm25_embedder import BM25SparseEmbedder


@pytest.fixture
def fitted_embedder() -> BM25SparseEmbedder:
    """Create a BM25SparseEmbedder fitted on a small corpus."""
    embedder = BM25SparseEmbedder()
    corpus = [
        "Le Plateau est un quartier branché de Montréal",
        "Rosemont est un quartier familial avec des parcs",
        "Le Mile End est connu pour ses cafés et restaurants",
        "Outremont est un quartier résidentiel et calme",
        "Verdun est un quartier en pleine transformation",
    ]
    embedder.fit(corpus)
    return embedder


class TestBM25SparseEmbedderInit:
    """Tests for BM25SparseEmbedder initialization."""

    def test_default_hyperparameters(self) -> None:
        embedder = BM25SparseEmbedder()
        assert embedder.k1 == 1.5
        assert embedder.b == 0.75
        assert embedder.delta == 1.0

    def test_custom_hyperparameters(self) -> None:
        embedder = BM25SparseEmbedder(k1=1.2, b=0.5, delta=0.5)
        assert embedder.k1 == 1.2
        assert embedder.b == 0.5
        assert embedder.delta == 0.5

    def test_not_fitted_by_default(self) -> None:
        embedder = BM25SparseEmbedder()
        assert embedder.is_fitted is False

    def test_embedding_dim_zero_before_fit(self) -> None:
        embedder = BM25SparseEmbedder()
        assert embedder.embedding_dim == 0

    def test_model_name_encodes_hyperparameters(self) -> None:
        embedder = BM25SparseEmbedder(k1=1.2, b=0.5, delta=0.5)
        assert "1.2" in embedder.model_name
        assert "0.5" in embedder.model_name


class TestBM25SparseEmbedderFit:
    """Tests for BM25SparseEmbedder.fit()."""

    def test_fit_sets_fitted_flag(self) -> None:
        embedder = BM25SparseEmbedder()
        embedder.fit(["hello world", "foo bar"])
        assert embedder.is_fitted is True

    def test_fit_builds_vocabulary(self) -> None:
        embedder = BM25SparseEmbedder()
        embedder.fit(["hello world", "foo bar"])
        assert embedder.embedding_dim > 0

    def test_fit_empty_corpus_raises(self) -> None:
        embedder = BM25SparseEmbedder()
        with pytest.raises(ValueError, match="Cannot fit BM25 on an empty corpus"):
            embedder.fit([])

    def test_fit_returns_self_for_chaining(self) -> None:
        """fit() returns self to allow method chaining."""
        embedder = BM25SparseEmbedder()
        result = embedder.fit(["hello world"])
        assert result is embedder

    def test_fit_method_chaining(self) -> None:
        result = BM25SparseEmbedder().fit(["hello world"]).embed_text("hello")
        assert isinstance(result, list)

    def test_fit_vocabulary_size_matches_unique_terms(self) -> None:
        embedder = BM25SparseEmbedder()
        embedder.fit(["hello world", "hello foo"])
        assert embedder.embedding_dim == 3  # hello, world, foo

    def test_fit_larger_corpus_larger_vocabulary(self) -> None:
        embedder_small = BM25SparseEmbedder()
        embedder_small.fit(["hello world"])

        embedder_large = BM25SparseEmbedder()
        embedder_large.fit(["hello world", "foo bar baz qux"])

        assert embedder_large.embedding_dim > embedder_small.embedding_dim


class TestBM25SparseEmbedderEmbedText:
    """Tests for BM25SparseEmbedder.embed_text()."""

    def test_embed_text_returns_list_of_floats(
        self, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        result = fitted_embedder.embed_text("Plateau Montréal")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_embed_text_length_equals_vocabulary_size(
        self, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        result = fitted_embedder.embed_text("Plateau Montréal")
        assert len(result) == fitted_embedder.embedding_dim

    def test_embed_text_sparse_most_values_are_zero(
        self, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        result = fitted_embedder.embed_text("Plateau")
        non_zero = sum(1 for v in result if v != 0.0)
        assert non_zero < len(result)

    def test_embed_text_known_term_has_nonzero_score(
        self, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        result = fitted_embedder.embed_text("Plateau")
        assert any(v > 0.0 for v in result)

    def test_embed_text_unknown_term_all_zeros(
        self, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        result = fitted_embedder.embed_text("xyzunknownterm")
        assert all(v == 0.0 for v in result)

    def test_embed_text_empty_raises(self, fitted_embedder: BM25SparseEmbedder) -> None:
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            fitted_embedder.embed_text("")

    def test_embed_text_not_fitted_raises(self) -> None:
        embedder = BM25SparseEmbedder()
        with pytest.raises(RuntimeError, match="must be fitted"):
            embedder.embed_text("hello")

    def test_embed_text_rare_term_higher_score_than_common_term(
        self, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        """Rare terms have higher IDF
        BM25 assigns them larger scores than frequent terms."""
        vocab = fitted_embedder._vocabulary

        rare_term = "branché"
        common_term = "est"

        if rare_term not in vocab or common_term not in vocab:
            pytest.skip("Terms not in vocabulary")

        rare_vector = fitted_embedder.embed_text(rare_term)
        common_vector = fitted_embedder.embed_text(common_term)

        rare_score = rare_vector[vocab[rare_term]]
        common_score = common_vector[vocab[common_term]]

        assert rare_score > common_score

    def test_embed_text_higher_tf_higher_score(self) -> None:
        """Higher term frequency yields a higher BM25 score."""
        embedder = BM25SparseEmbedder()
        embedder.fit(["hello world foo bar baz"])

        single = embedder.embed_text("hello")
        repeated = embedder.embed_text("hello hello hello")

        vocab = embedder._vocabulary
        idx = vocab["hello"]

        assert repeated[idx] > single[idx]


class TestBM25SparseEmbedderEmbedBatch:
    """Tests for BM25SparseEmbedder.embed_batch()."""

    def test_embed_batch_returns_list_of_vectors(
        self, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        result = fitted_embedder.embed_batch(["Plateau", "Rosemont"])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_embed_batch_preserves_order(
        self, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        """Order of returned vectors must match order of input texts."""
        texts = ["Plateau branché", "Rosemont familial", "Mile End cafés"]
        result = fitted_embedder.embed_batch(texts)

        for i, text in enumerate(texts):
            single = fitted_embedder.embed_text(text)
            assert result[i] == single

    def test_embed_batch_empty_raises(
        self, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        with pytest.raises(ValueError, match="Cannot embed an empty list"):
            fitted_embedder.embed_batch([])

    def test_embed_batch_not_fitted_raises(self) -> None:
        embedder = BM25SparseEmbedder()
        with pytest.raises(RuntimeError, match="must be fitted"):
            embedder.embed_batch(["hello"])

    def test_embed_batch_single_item_matches_embed_text(
        self, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        single = fitted_embedder.embed_text("Plateau")
        batch = fitted_embedder.embed_batch(["Plateau"])
        assert batch[0] == single


class TestBM25SparseEmbedderCacheKey:
    """Tests for get_cache_key()."""

    def test_cache_key_is_string(self) -> None:
        assert isinstance(BM25SparseEmbedder().get_cache_key(), str)

    def test_cache_key_is_deterministic(self) -> None:
        embedder = BM25SparseEmbedder()
        assert embedder.get_cache_key() == embedder.get_cache_key()

    def test_different_hyperparameters_different_cache_key(self) -> None:
        embedder1 = BM25SparseEmbedder(k1=1.2)
        embedder2 = BM25SparseEmbedder(k1=2.0)
        assert embedder1.get_cache_key() != embedder2.get_cache_key()

    def test_same_hyperparameters_same_cache_key(self) -> None:
        embedder1 = BM25SparseEmbedder(k1=1.5, b=0.75)
        embedder2 = BM25SparseEmbedder(k1=1.5, b=0.75)
        assert embedder1.get_cache_key() == embedder2.get_cache_key()


class TestBM25SparseEmbedderTokenize:
    """Tests for BM25SparseEmbedder._tokenize()."""

    def test_tokenize_lowercase(self) -> None:
        result = BM25SparseEmbedder._tokenize("Hello World")
        assert result == ["hello", "world"]

    def test_tokenize_removes_punctuation(self) -> None:
        result = BM25SparseEmbedder._tokenize("hello, world!")
        assert result == ["hello", "world"]

    def test_tokenize_empty_string(self) -> None:
        result = BM25SparseEmbedder._tokenize("")
        assert result == []

    def test_tokenize_hyphenated_words(self) -> None:
        """Hyphenated words are split into separate tokens."""
        result = BM25SparseEmbedder._tokenize("Plateau-Mont-Royal")
        assert "plateau" in result
        assert "mont" in result
        assert "royal" in result

    def test_tokenize_numbers(self) -> None:
        result = BM25SparseEmbedder._tokenize("5490 boul Saint-Laurent")
        assert "5490" in result
        assert "saint" in result


class TestBM25SparseEmbedderSave:
    """Tests for BM25SparseEmbedder.save()."""

    def test_save_raises_if_not_fitted(self, tmp_path: Path) -> None:
        embedder = BM25SparseEmbedder()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            embedder.save(tmp_path / "model.pkl")

    def test_save_creates_file(
        self, tmp_path: Path, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        path = tmp_path / "model.pkl"
        fitted_embedder.save(path)
        assert path.exists()

    def test_save_returns_resolved_path(
        self, tmp_path: Path, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        path = tmp_path / "model.pkl"
        returned = fitted_embedder.save(path)
        assert returned == path.resolve()

    def test_save_creates_nested_parent_dirs(
        self, tmp_path: Path, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        nested = tmp_path / "a" / "b" / "c" / "model.pkl"
        fitted_embedder.save(nested)
        assert nested.exists()

    def test_save_file_contains_valid_pickle(
        self, tmp_path: Path, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        path = tmp_path / "model.pkl"
        fitted_embedder.save(path)
        with path.open("rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, BM25SparseEmbedder)

    def test_save_overwrites_existing_file(
        self, tmp_path: Path, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        path = tmp_path / "model.pkl"
        path.write_bytes(b"stale")
        fitted_embedder.save(path)
        with path.open("rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, BM25SparseEmbedder)


class TestBM25SparseEmbedderLoad:
    """Tests for BM25SparseEmbedder.load()."""

    def test_load_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="BM25 cache not found"):
            BM25SparseEmbedder.load(tmp_path / "missing.pkl")

    def test_load_wrong_type_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "wrong.pkl"
        with path.open("wb") as f:
            pickle.dump({"not": "a bm25"}, f)
        with pytest.raises(TypeError, match="Expected BM25SparseEmbedder"):
            BM25SparseEmbedder.load(path)

    def test_load_returns_bm25_sparse_embedder(
        self, tmp_path: Path, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        path = tmp_path / "model.pkl"
        fitted_embedder.save(path)
        loaded = BM25SparseEmbedder.load(path)
        assert isinstance(loaded, BM25SparseEmbedder)

    def test_load_is_fitted(
        self, tmp_path: Path, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        path = tmp_path / "model.pkl"
        fitted_embedder.save(path)
        loaded = BM25SparseEmbedder.load(path)
        assert loaded.is_fitted

    def test_load_preserves_hyperparameters(self, tmp_path: Path) -> None:
        original = BM25SparseEmbedder(k1=1.2, b=0.6, delta=0.8)
        original.fit(["hello world", "foo bar"])
        path = tmp_path / "model.pkl"
        original.save(path)
        loaded = BM25SparseEmbedder.load(path)
        assert loaded.k1 == pytest.approx(1.2)
        assert loaded.b == pytest.approx(0.6)
        assert loaded.delta == pytest.approx(0.8)

    def test_load_preserves_vocabulary_size(
        self, tmp_path: Path, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        path = tmp_path / "model.pkl"
        fitted_embedder.save(path)
        loaded = BM25SparseEmbedder.load(path)
        assert loaded.embedding_dim == fitted_embedder.embedding_dim

    def test_load_embedding_matches_original(
        self, tmp_path: Path, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        path = tmp_path / "model.pkl"
        fitted_embedder.save(path)
        loaded = BM25SparseEmbedder.load(path)
        text = "Plateau Montréal"
        assert loaded.embed_text(text) == pytest.approx(
            fitted_embedder.embed_text(text)
        )

    def test_load_accepts_str_path(
        self, tmp_path: Path, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        path = tmp_path / "model.pkl"
        fitted_embedder.save(path)
        loaded = BM25SparseEmbedder.load(str(path))
        assert isinstance(loaded, BM25SparseEmbedder)


class TestBM25SparseEmbedderSaveLoadRoundtrip:
    """End-to-end save → load → embed consistency tests."""

    def test_roundtrip_embed_text_is_identical(
        self, tmp_path: Path, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        path = tmp_path / "model.pkl"
        fitted_embedder.save(path)
        loaded = BM25SparseEmbedder.load(path)
        for text in ["Plateau", "Rosemont familial", "Mile End cafés"]:
            assert loaded.embed_text(text) == pytest.approx(
                fitted_embedder.embed_text(text)
            )

    def test_roundtrip_embed_batch_is_identical(
        self, tmp_path: Path, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        path = tmp_path / "model.pkl"
        fitted_embedder.save(path)
        loaded = BM25SparseEmbedder.load(path)
        texts = ["Le Plateau", "Rosemont", "Outremont calme"]
        original_batch = fitted_embedder.embed_batch(texts)
        loaded_batch = loaded.embed_batch(texts)
        for orig, reloaded in zip(original_batch, loaded_batch):
            assert reloaded == pytest.approx(orig)

    def test_roundtrip_model_name_is_preserved(
        self, tmp_path: Path, fitted_embedder: BM25SparseEmbedder
    ) -> None:
        path = tmp_path / "model.pkl"
        fitted_embedder.save(path)
        loaded = BM25SparseEmbedder.load(path)
        assert loaded.model_name == fitted_embedder.model_name
