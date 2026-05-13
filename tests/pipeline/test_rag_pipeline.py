# tests/pipeline/test_rag_pipeline.py
from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from src.pipeline.rag_pipeline import RAGResult


def _make_rag_result(**kwargs: Any) -> RAGResult:
    defaults: dict[str, Any] = {
        "query": "Quelle est la capitale de la France ?",
        "answer": "Paris",
        "sources": ["chunk-001", "chunk-002"],
        "detected_language": "fr",
        "model": "gemma3:4b",
        "chunks_retrieved": 5,
        "tokens_used": 120,
    }
    defaults.update(kwargs)
    return RAGResult(**defaults)


class TestRAGResultCreation:
    def test_query_stored(self) -> None:
        result = _make_rag_result(query="What is RAG?")
        assert result.query == "What is RAG?"

    def test_answer_stored(self) -> None:
        result = _make_rag_result(
            answer="RAG stands for Retrieval-Augmented Generation."
        )
        assert result.answer == "RAG stands for Retrieval-Augmented Generation."

    def test_sources_stored(self) -> None:
        result = _make_rag_result(sources=["chunk-001", "chunk-002"])
        assert result.sources == ["chunk-001", "chunk-002"]

    def test_detected_language_stored(self) -> None:
        result = _make_rag_result(detected_language="en")
        assert result.detected_language == "en"

    def test_model_stored(self) -> None:
        result = _make_rag_result(model="llama3:8b")
        assert result.model == "llama3:8b"

    def test_chunks_retrieved_stored(self) -> None:
        result = _make_rag_result(chunks_retrieved=10)
        assert result.chunks_retrieved == 10

    def test_tokens_used_stored(self) -> None:
        result = _make_rag_result(tokens_used=256)
        assert result.tokens_used == 256


class TestRAGResultImmutability:
    def test_query_is_frozen(self) -> None:
        result = _make_rag_result()
        with pytest.raises(FrozenInstanceError):
            result.query = "other"  # type: ignore[misc]

    def test_answer_is_frozen(self) -> None:
        result = _make_rag_result()
        with pytest.raises(FrozenInstanceError):
            result.answer = "other"  # type: ignore[misc]

    def test_sources_field_is_frozen(self) -> None:
        result = _make_rag_result()
        with pytest.raises(FrozenInstanceError):
            result.sources = []  # type: ignore[misc]

    def test_tokens_used_is_frozen(self) -> None:
        result = _make_rag_result()
        with pytest.raises(FrozenInstanceError):
            result.tokens_used = 0  # type: ignore[misc]


class TestRAGResultEdgeCases:
    def test_tokens_used_none_when_provider_does_not_report(self) -> None:
        result = _make_rag_result(tokens_used=None)
        assert result.tokens_used is None

    def test_sources_empty_list(self) -> None:
        result = _make_rag_result(sources=[])
        assert result.sources == []

    def test_chunks_retrieved_zero(self) -> None:
        result = _make_rag_result(chunks_retrieved=0)
        assert result.chunks_retrieved == 0

    def test_equality_on_identical_instances(self) -> None:
        a = _make_rag_result()
        b = _make_rag_result()
        assert a == b

    def test_inequality_on_different_query(self) -> None:
        a = _make_rag_result(query="question A")
        b = _make_rag_result(query="question B")
        assert a != b
