# tests/pipeline/test_rag_pipeline.py
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.embeddings.bm25_embedder import BM25SparseEmbedder
from src.embeddings.hybrid_embedder import HybridEmbedder, HybridEmbedding
from src.generation.generator import GenerationResult, LLMGenerator
from src.pipeline.config import Settings
from src.pipeline.rag_pipeline import RAGPipeline, RAGResult
from src.retrieval.retriever import Retriever
from src.shared.models import Chunk, ChunkMetadata, MetadataFilter
from src.vector_store.vector_store import SearchResult

# ── Helpers ───────────────────────────────────────────────────────────────────


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


def _make_chunk(chunk_id: str = "chunk-001") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        parent_doc_id="doc-001",
        text="Paris est la capitale de la France.",
        chunk_index=0,
        total_chunks=1,
        metadata=ChunkMetadata(),
    )


def _make_search_results(n: int = 2) -> list[SearchResult]:
    return [
        SearchResult(chunk=_make_chunk(f"chunk-{i:03d}"), score=0.9 - i * 0.05)
        for i in range(n)
    ]


def _make_generation_result(
    answer: str = "Paris est la capitale.",
    sources: list[str] | None = None,
    tokens_used: int | None = 100,
) -> GenerationResult:
    return GenerationResult(
        answer=answer,
        sources=sources if sources is not None else ["chunk-000", "chunk-001"],
        detected_language="fr",
        model="gemma3:4b",
        prompt_version="v1",
        tokens_used=tokens_used,
    )


def _make_query_embedding() -> HybridEmbedding:
    return HybridEmbedding(
        dense=[0.1, 0.2, 0.3, 0.4],
        sparse_indices=[0, 2],
        sparse_values=[0.5, 0.3],
        text="capitale de la France",
    )


def _make_fitted_bm25(path: Path) -> Path:
    bm25 = BM25SparseEmbedder()
    bm25.fit(["Le Plateau est branché", "Rosemont est familial"])
    bm25.save(path)
    return path


def _make_pipeline_under_test() -> tuple[RAGPipeline, MagicMock, MagicMock, MagicMock]:
    embedder = MagicMock(spec=HybridEmbedder)
    retriever = MagicMock(spec=Retriever)
    generator = MagicMock(spec=LLMGenerator)

    embedder.embed_query.return_value = _make_query_embedding()
    retriever.retrieve.return_value = _make_search_results(n=2)
    generator.generate.return_value = _make_generation_result()
    generator.provider_name = "ollama"
    generator.model_name = "gemma3:4b"

    pipeline = RAGPipeline(embedder=embedder, retriever=retriever, generator=generator)
    return pipeline, embedder, retriever, generator


# ── RAGResult — creation ──────────────────────────────────────────────────────


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


# ── RAGResult — immutability ──────────────────────────────────────────────────


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


# ── RAGResult — edge cases ────────────────────────────────────────────────────


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


# ── RAGPipeline.build ─────────────────────────────────────────────────────────


class TestRAGPipelineBuild:
    def test_build_returns_rag_pipeline(self, tmp_path: Path) -> None:
        _make_fitted_bm25(tmp_path / "bm25.pkl")
        settings = Settings(bm25_cache_path=tmp_path / "bm25.pkl")

        with (
            patch(
                "src.pipeline.rag_pipeline.SentenceTransformerEmbedder"
            ) as mock_dense,
            patch("src.pipeline.rag_pipeline.QdrantVectorStore"),
            patch("src.pipeline.rag_pipeline.get_generator") as mock_gen_factory,
        ):
            mock_dense.return_value.embedding_dim = 4
            mock_dense.return_value.model_name = "mock-dense"
            mock_gen_factory.return_value = MagicMock(spec=LLMGenerator)
            mock_gen_factory.return_value.provider_name = "ollama"
            mock_gen_factory.return_value.model_name = "gemma3:4b"

            pipeline = RAGPipeline.build(settings)

        assert isinstance(pipeline, RAGPipeline)

    def test_build_raises_file_not_found_when_bm25_missing(
        self, tmp_path: Path
    ) -> None:
        settings = Settings(bm25_cache_path=tmp_path / "nonexistent.pkl")
        with pytest.raises(FileNotFoundError):
            RAGPipeline.build(settings)

    def test_build_passes_embedding_model_to_dense_embedder(
        self, tmp_path: Path
    ) -> None:
        _make_fitted_bm25(tmp_path / "bm25.pkl")
        settings = Settings(
            bm25_cache_path=tmp_path / "bm25.pkl",
            embedding_model="intfloat/multilingual-e5-large",
        )

        with (
            patch(
                "src.pipeline.rag_pipeline.SentenceTransformerEmbedder"
            ) as mock_dense,
            patch("src.pipeline.rag_pipeline.QdrantVectorStore"),
            patch("src.pipeline.rag_pipeline.get_generator") as mock_gen_factory,
        ):
            mock_dense.return_value.embedding_dim = 4
            mock_dense.return_value.model_name = "intfloat/multilingual-e5-large"
            mock_gen_factory.return_value = MagicMock(spec=LLMGenerator)
            mock_gen_factory.return_value.provider_name = "ollama"
            mock_gen_factory.return_value.model_name = "gemma3:4b"

            RAGPipeline.build(settings)

        mock_dense.assert_called_once_with(model_name="intfloat/multilingual-e5-large")

    def test_build_passes_qdrant_params_to_vector_store(self, tmp_path: Path) -> None:
        _make_fitted_bm25(tmp_path / "bm25.pkl")
        settings = Settings(
            bm25_cache_path=tmp_path / "bm25.pkl",
            qdrant_host="my-host",
            qdrant_port=6334,
            collection_name="my-collection",
        )

        with (
            patch(
                "src.pipeline.rag_pipeline.SentenceTransformerEmbedder"
            ) as mock_dense,
            patch("src.pipeline.rag_pipeline.QdrantVectorStore") as mock_store,
            patch("src.pipeline.rag_pipeline.get_generator") as mock_gen_factory,
        ):
            mock_dense.return_value.embedding_dim = 4
            mock_dense.return_value.model_name = "mock-dense"
            mock_gen_factory.return_value = MagicMock(spec=LLMGenerator)
            mock_gen_factory.return_value.provider_name = "ollama"
            mock_gen_factory.return_value.model_name = "gemma3:4b"

            RAGPipeline.build(settings)

        mock_store.assert_called_once_with(
            host="my-host",
            port=6334,
            collection_name="my-collection",
        )

    def test_build_passes_settings_to_generator_factory(self, tmp_path: Path) -> None:
        _make_fitted_bm25(tmp_path / "bm25.pkl")
        settings = Settings(bm25_cache_path=tmp_path / "bm25.pkl")

        with (
            patch(
                "src.pipeline.rag_pipeline.SentenceTransformerEmbedder"
            ) as mock_dense,
            patch("src.pipeline.rag_pipeline.QdrantVectorStore"),
            patch("src.pipeline.rag_pipeline.get_generator") as mock_gen_factory,
        ):
            mock_dense.return_value.embedding_dim = 4
            mock_dense.return_value.model_name = "mock-dense"
            mock_gen_factory.return_value = MagicMock(spec=LLMGenerator)
            mock_gen_factory.return_value.provider_name = "ollama"
            mock_gen_factory.return_value.model_name = "gemma3:4b"

            RAGPipeline.build(settings)

        mock_gen_factory.assert_called_once_with(settings)

    def test_build_raises_connection_error_when_qdrant_unreachable(
        self, tmp_path: Path
    ) -> None:
        _make_fitted_bm25(tmp_path / "bm25.pkl")
        settings = Settings(bm25_cache_path=tmp_path / "bm25.pkl")

        with (
            patch(
                "src.pipeline.rag_pipeline.SentenceTransformerEmbedder"
            ) as mock_dense,
            patch(
                "src.pipeline.rag_pipeline.QdrantVectorStore",
                side_effect=ConnectionError("Qdrant unreachable"),
            ),
        ):
            mock_dense.return_value.embedding_dim = 4
            mock_dense.return_value.model_name = "mock-dense"

            with pytest.raises(ConnectionError, match="Qdrant unreachable"):
                RAGPipeline.build(settings)


# ── RAGPipeline.query — result fields ────────────────────────────────────────


class TestRAGPipelineQueryResult:
    def test_query_returns_rag_result(self) -> None:
        pipeline, _, _, _ = _make_pipeline_under_test()
        result = pipeline.query("capitale de la France ?")
        assert isinstance(result, RAGResult)

    def test_query_field_matches_input(self) -> None:
        pipeline, _, _, _ = _make_pipeline_under_test()
        result = pipeline.query("capitale de la France ?")
        assert result.query == "capitale de la France ?"

    def test_answer_comes_from_generator(self) -> None:
        pipeline, _, _, generator = _make_pipeline_under_test()
        generator.generate.return_value = _make_generation_result(
            answer="Paris est la capitale."
        )
        result = pipeline.query("capitale ?")
        assert result.answer == "Paris est la capitale."

    def test_sources_come_from_generator(self) -> None:
        pipeline, _, _, generator = _make_pipeline_under_test()
        generator.generate.return_value = _make_generation_result(
            sources=["chunk-000", "chunk-001"]
        )
        result = pipeline.query("capitale ?")
        assert result.sources == ["chunk-000", "chunk-001"]

    def test_chunks_retrieved_matches_retrieval_count(self) -> None:
        pipeline, _, retriever, _ = _make_pipeline_under_test()
        retriever.retrieve.return_value = _make_search_results(n=4)
        result = pipeline.query("capitale ?")
        assert result.chunks_retrieved == 4

    def test_detected_language_comes_from_generator(self) -> None:
        pipeline, _, _, generator = _make_pipeline_under_test()
        generator.generate.return_value = GenerationResult(
            answer="Paris.",
            sources=["chunk-000"],
            detected_language="en",
            model="gemma3:4b",
            prompt_version="v1",
        )
        result = pipeline.query("what is the capital?")
        assert result.detected_language == "en"

    def test_model_comes_from_generator(self) -> None:
        pipeline, _, _, generator = _make_pipeline_under_test()
        generator.generate.return_value = GenerationResult(
            answer="Paris.",
            sources=["chunk-000"],
            detected_language="fr",
            model="llama3:8b",
            prompt_version="v1",
        )
        result = pipeline.query("capitale ?")
        assert result.model == "llama3:8b"

    def test_tokens_used_propagated_from_generator(self) -> None:
        pipeline, _, _, generator = _make_pipeline_under_test()
        generator.generate.return_value = _make_generation_result(tokens_used=250)
        result = pipeline.query("capitale ?")
        assert result.tokens_used == 250

    def test_tokens_used_none_when_not_reported(self) -> None:
        pipeline, _, _, generator = _make_pipeline_under_test()
        generator.generate.return_value = _make_generation_result(tokens_used=None)
        result = pipeline.query("capitale ?")
        assert result.tokens_used is None


# ── RAGPipeline.query — layer wiring ─────────────────────────────────────────


class TestRAGPipelineQueryLayerWiring:
    def test_embed_query_called_with_user_query(self) -> None:
        pipeline, embedder, _, _ = _make_pipeline_under_test()
        pipeline.query("capitale de la France")
        embedder.embed_query.assert_called_once_with("capitale de la France")

    def test_retriever_receives_query_embedding(self) -> None:
        pipeline, embedder, retriever, _ = _make_pipeline_under_test()
        expected = _make_query_embedding()
        embedder.embed_query.return_value = expected

        pipeline.query("capitale de la France")

        assert retriever.retrieve.call_args.args[0] is expected

    def test_generator_receives_chunks_from_search_results(self) -> None:
        pipeline, _, retriever, generator = _make_pipeline_under_test()
        results = _make_search_results(n=3)
        retriever.retrieve.return_value = results

        pipeline.query("capitale ?")

        chunks_arg = generator.generate.call_args.args[1]
        assert len(chunks_arg) == 3
        assert all(chunks_arg[i] is results[i].chunk for i in range(3))

    def test_generator_receives_user_query(self) -> None:
        pipeline, _, _, generator = _make_pipeline_under_test()
        pipeline.query("ma question")
        assert generator.generate.call_args.args[0] == "ma question"

    def test_top_k_override_forwarded_to_retriever(self) -> None:
        pipeline, _, retriever, _ = _make_pipeline_under_test()
        pipeline.query("test", top_k=10)
        assert retriever.retrieve.call_args.kwargs["top_k"] == 10

    def test_score_threshold_override_forwarded_to_retriever(self) -> None:
        pipeline, _, retriever, _ = _make_pipeline_under_test()
        pipeline.query("test", score_threshold=0.5)
        assert retriever.retrieve.call_args.kwargs["score_threshold"] == pytest.approx(
            0.5
        )

    def test_filters_forwarded_to_retriever(self) -> None:
        pipeline, _, retriever, _ = _make_pipeline_under_test()
        filters = [MetadataFilter(field="metadata.language", value="fr")]
        pipeline.query("test", filters=filters)
        assert retriever.retrieve.call_args.kwargs["filters"] is filters

    def test_none_overrides_forwarded_to_retriever(self) -> None:
        pipeline, _, retriever, _ = _make_pipeline_under_test()
        pipeline.query("test")
        kwargs = retriever.retrieve.call_args.kwargs
        assert kwargs["top_k"] is None
        assert kwargs["score_threshold"] is None
        assert kwargs["filters"] is None


# ── RAGPipeline.query — error handling ───────────────────────────────────────


class TestRAGPipelineQueryErrors:
    def test_empty_query_raises_value_error(self) -> None:
        pipeline, _, _, _ = _make_pipeline_under_test()
        with pytest.raises(ValueError, match="empty"):
            pipeline.query("")

    def test_whitespace_only_query_raises_value_error(self) -> None:
        pipeline, _, _, _ = _make_pipeline_under_test()
        with pytest.raises(ValueError, match="empty"):
            pipeline.query("   ")

    def test_no_results_raises_runtime_error(self) -> None:
        pipeline, _, retriever, _ = _make_pipeline_under_test()
        retriever.retrieve.return_value = []
        with pytest.raises(RuntimeError, match="No chunks retrieved"):
            pipeline.query("question sans résultat")

    def test_empty_query_does_not_call_retriever(self) -> None:
        pipeline, _, retriever, _ = _make_pipeline_under_test()
        with pytest.raises(ValueError):
            pipeline.query("")
        retriever.retrieve.assert_not_called()

    def test_retriever_runtime_error_propagates(self) -> None:
        pipeline, _, retriever, _ = _make_pipeline_under_test()
        retriever.retrieve.side_effect = RuntimeError("Qdrant connection lost")
        with pytest.raises(RuntimeError, match="Qdrant connection lost"):
            pipeline.query("capitale ?")

    def test_generator_runtime_error_propagates(self) -> None:
        pipeline, _, _, generator = _make_pipeline_under_test()
        generator.generate.side_effect = RuntimeError("LLM server unreachable")
        with pytest.raises(RuntimeError, match="LLM server unreachable"):
            pipeline.query("capitale ?")

    def test_no_results_does_not_call_generator(self) -> None:
        pipeline, _, retriever, generator = _make_pipeline_under_test()
        retriever.retrieve.return_value = []
        with pytest.raises(RuntimeError):
            pipeline.query("question sans résultat")
        generator.generate.assert_not_called()


# ── RAGPipeline.query — edge cases ────────────────────────────────────────────


class TestRAGPipelineQueryEdgeCases:
    def test_single_chunk_retrieved(self) -> None:
        pipeline, _, retriever, _ = _make_pipeline_under_test()
        retriever.retrieve.return_value = _make_search_results(n=1)
        result = pipeline.query("capitale ?")
        assert result.chunks_retrieved == 1

    def test_chunks_retrieved_equals_generator_context_size(self) -> None:
        pipeline, _, retriever, generator = _make_pipeline_under_test()
        retriever.retrieve.return_value = _make_search_results(n=5)
        pipeline.query("capitale ?")
        context_passed = generator.generate.call_args.args[1]
        assert len(context_passed) == 5

    def test_query_string_preserved_verbatim_in_result(self) -> None:
        pipeline, _, _, _ = _make_pipeline_under_test()
        raw = "  Quel est le meilleur café ? "
        result = pipeline.query(raw)
        assert result.query == raw

    def test_chunk_order_from_retriever_preserved_to_generator(self) -> None:
        pipeline, _, retriever, generator = _make_pipeline_under_test()
        results = _make_search_results(n=3)
        retriever.retrieve.return_value = results
        pipeline.query("capitale ?")
        context = generator.generate.call_args.args[1]
        assert [c.chunk_id for c in context] == [r.chunk.chunk_id for r in results]
