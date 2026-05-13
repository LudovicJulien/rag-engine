# src/embeddings/bm25_embedder.py
from __future__ import annotations

import hashlib
import math
import pickle
from collections import Counter
from pathlib import Path

from src.embeddings.embedding_model import EmbeddingModel


class BM25SparseEmbedder(EmbeddingModel):
    """Sparse embedding model using BM25 (Best Match 25) algorithm.

    BM25 produces sparse vectors where each dimension corresponds to a term
    in the corpus vocabulary. Unlike dense embeddings, BM25 excels at
    exact keyword matching and is complementary to dense embeddings in
    hybrid retrieval.

    This implementation uses BM25+ scoring which addresses the lower-bounding
    problem of the original BM25 formula.

    Example usage:
        corpus = ["Le Plateau est branché", "Rosemont est familial"]
        embedder = BM25SparseEmbedder()
        embedder.fit(corpus)
        vector = embedder.embed_text("Plateau Montréal")
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
    ) -> None:
        """Initialize BM25SparseEmbedder.

        Args:
            k1: Term frequency saturation parameter.
                Higher values increase the influence of term frequency.
                Typical range: [1.2, 2.0]. Defaults to 1.5.
            b: Length normalization parameter.
                1.0 = full normalization, 0.0 = no normalization.
                Defaults to 0.75.
            delta: BM25+ lower bound parameter.
                Prevents zero scores for terms present in a document.
                Defaults to 1.0.
        """
        self.k1 = k1
        self.b = b
        self.delta = delta

        self._vocabulary: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._avg_doc_len: float = 0.0
        self._corpus_size: int = 0
        self._fitted: bool = False

    @property
    def embedding_dim(self) -> int:
        """Return vocabulary size — the dimensionality of sparse vectors."""
        return len(self._vocabulary)

    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return f"bm25-k1={self.k1}-b={self.b}-delta={self.delta}"

    @property
    def is_fitted(self) -> bool:
        """Return True if the model has been fitted on a corpus."""
        return self._fitted

    def fit(self, corpus: list[str]) -> BM25SparseEmbedder:
        """Fit the BM25 model on a corpus.

        Builds the vocabulary and computes IDF scores for all terms.
        Must be called before embed_text() or embed_batch().

        Args:
            corpus: List of text documents to fit on. Must be non-empty.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If corpus is empty.
        """
        if not corpus:
            raise ValueError("Cannot fit BM25 on an empty corpus")

        self._corpus_size = len(corpus)
        tokenized = [self._tokenize(doc) for doc in corpus]

        total_len = sum(len(tokens) for tokens in tokenized)
        self._avg_doc_len = total_len / self._corpus_size

        all_terms: set[str] = set()
        for tokens in tokenized:
            all_terms.update(tokens)

        self._vocabulary = {term: idx for idx, term in enumerate(sorted(all_terms))}

        doc_frequencies: Counter[str] = Counter()
        for tokens in tokenized:
            doc_frequencies.update(set(tokens))

        self._idf = {}
        for term, df in doc_frequencies.items():
            self._idf[term] = math.log((self._corpus_size - df + 0.5) / (df + 0.5) + 1)

        self._fitted = True
        return self

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text into a sparse BM25 vector.

        Args:
            text: The text to embed. Must be non-empty.

        Returns:
            Sparse vector of length equal to vocabulary size.
            Most values are 0.0 — only matching terms have non-zero scores.

        Raises:
            ValueError: If text is empty or model is not fitted.
        """
        if not text:
            raise ValueError("Cannot embed empty text")
        if not self._fitted:
            raise RuntimeError("BM25SparseEmbedder must be fitted before embedding")

        tokens = self._tokenize(text)
        term_freq = Counter(tokens)
        doc_len = len(tokens)

        vector = [0.0] * len(self._vocabulary)

        for term, tf in term_freq.items():
            if term not in self._vocabulary:
                continue
            idx = self._vocabulary[term]
            idf = self._idf.get(term, 0.0)
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_len)
            ) + self.delta
            vector[idx] = idf * tf_norm

        return vector

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a list of texts into sparse BM25 vectors.

        Args:
            texts: List of texts to embed. Must be non-empty.
            batch_size: Ignored for BM25 — included for interface compatibility.

        Returns:
            List of sparse vectors in input order.

        Raises:
            ValueError: If texts is empty or model is not fitted.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts")
        if not self._fitted:
            raise RuntimeError("BM25SparseEmbedder must be fitted before embedding")

        return [self.embed_text(text) for text in texts]

    def save(self, path: str | Path) -> Path:
        """Serialize the fitted model to a pickle file.

        Parent directories are created automatically if they do not exist.

        Args:
            path: Destination path for the pickle file
                (e.g. ``.bm25_cache/corpus.pkl``).

        Returns:
            Resolved absolute path to the written file.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError(
                "Cannot save an unfitted BM25SparseEmbedder — call fit() first."
            )
        resolved_path = Path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with resolved_path.open("wb") as f:
            pickle.dump(self, f)
        return resolved_path.resolve()

    @classmethod
    def load(cls, path: str | Path) -> BM25SparseEmbedder:
        """Load a fitted BM25SparseEmbedder from a pickle file.

        Args:
            path: Path to a file produced by :meth:`save`.

        Returns:
            The restored :class:`BM25SparseEmbedder` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            TypeError: If the file does not contain a :class:`BM25SparseEmbedder`.
        """
        resolved_path = Path(path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"BM25 cache not found: {resolved_path}")
        with resolved_path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj

    def get_cache_key(self) -> str:
        """Return a unique cache key for this BM25 configuration.

        Returns:
            MD5 hash of the model name.
        """
        return hashlib.md5(self.model_name.encode()).hexdigest()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase terms.

        Simple whitespace + punctuation tokenizer.
        Can be overridden for language-specific tokenization.

        Args:
            text: Raw text to tokenize.

        Returns:
            List of lowercase tokens.
        """
        import re

        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens
