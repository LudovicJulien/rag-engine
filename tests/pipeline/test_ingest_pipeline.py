# tests/pipeline/test_ingest_pipeline.py
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.embeddings.bm25_embedder import BM25SparseEmbedder
from src.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.pipeline.config import Settings
from src.pipeline.ingest_pipeline import IngestionPipeline, IngestResult
from src.vector_store.vector_store import UpsertResult, VectorStore

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_ingest_result(**kwargs: Any) -> IngestResult:
    defaults: dict[str, Any] = {
        "source": "data/chunks.json",
        "collection_name": "rag-collection",
        "chunks_loaded": 100,
        "chunks_upserted": 98,
        "chunks_failed": 2,
        "bm25_cache_path": "/home/user/.cache/rag/bm25.pkl",
    }
    defaults.update(kwargs)
    return IngestResult(**defaults)


def _write_chunks_json(path: Path, n: int = 3) -> Path:
    """Write a minimal valid chunks JSON file to *path*."""
    chunks = [
        {
            "chunk_id": f"chunk-{i:03d}",
            "parent_doc_id": "doc-001",
            "text": f"Chunk number {i} with some unique content for BM25 fitting.",
            "chunk_index": i,
            "total_chunks": n,
        }
        for i in range(n)
    ]
    path.write_text(json.dumps(chunks), encoding="utf-8")
    return path


def _make_dense_mock(dim: int = 4) -> Mock:
    mock = Mock(spec=SentenceTransformerEmbedder)
    mock.embedding_dim = dim
    mock.model_name = "mock-dense"
    mock.embed_batch.side_effect = lambda texts, batch_size=32: [
        [0.1] * dim for _ in texts
    ]
    mock.embed_text.side_effect = lambda text: [0.1] * dim
    mock.embed_query.side_effect = lambda text: [0.1] * dim
    return mock


def _make_store_mock(exists: bool = True) -> MagicMock:
    mock = MagicMock(spec=VectorStore)
    mock.collection_exists.return_value = exists
    mock.create_collection.return_value = None
    mock.upsert.return_value = UpsertResult(upserted=0, failed=0)
    return mock


def _make_pipeline(
    tmp_path: Path,
    *,
    store_exists: bool = True,
    upserted: int = 3,
    failed: int = 0,
    alpha: float = 0.5,
    batch_size: int = 32,
) -> tuple[IngestionPipeline, MagicMock]:
    dense = _make_dense_mock()
    sparse = BM25SparseEmbedder()
    store = _make_store_mock(exists=store_exists)
    store.upsert.return_value = UpsertResult(upserted=upserted, failed=failed)

    pipeline = IngestionPipeline(
        dense=dense,
        sparse=sparse,
        vector_store=store,
        collection_name="test-collection",
        batch_size=batch_size,
        bm25_cache_dir=tmp_path / "bm25",
        alpha=alpha,
    )
    return pipeline, store


# ── IngestResult tests ────────────────────────────────────────────────────────


class TestIngestResultCreation:
    def test_source_stored(self) -> None:
        result = _make_ingest_result(source="data/tourism.json")
        assert result.source == "data/tourism.json"

    def test_collection_name_stored(self) -> None:
        result = _make_ingest_result(collection_name="my-collection")
        assert result.collection_name == "my-collection"

    def test_chunks_loaded_stored(self) -> None:
        result = _make_ingest_result(chunks_loaded=200)
        assert result.chunks_loaded == 200

    def test_chunks_upserted_stored(self) -> None:
        result = _make_ingest_result(chunks_upserted=195)
        assert result.chunks_upserted == 195

    def test_chunks_failed_stored(self) -> None:
        result = _make_ingest_result(chunks_failed=5)
        assert result.chunks_failed == 5

    def test_bm25_cache_path_stored(self) -> None:
        result = _make_ingest_result(bm25_cache_path="/tmp/bm25.pkl")
        assert result.bm25_cache_path == "/tmp/bm25.pkl"


class TestIngestResultImmutability:
    def test_source_is_frozen(self) -> None:
        result = _make_ingest_result()
        with pytest.raises(FrozenInstanceError):
            result.source = "other.json"  # type: ignore[misc]

    def test_chunks_upserted_is_frozen(self) -> None:
        result = _make_ingest_result()
        with pytest.raises(FrozenInstanceError):
            result.chunks_upserted = 0  # type: ignore[misc]

    def test_bm25_cache_path_is_frozen(self) -> None:
        result = _make_ingest_result()
        with pytest.raises(FrozenInstanceError):
            result.bm25_cache_path = "/other/bm25.pkl"  # type: ignore[misc]


class TestIngestResultEdgeCases:
    def test_chunks_failed_zero_on_perfect_run(self) -> None:
        result = _make_ingest_result(
            chunks_loaded=50, chunks_upserted=50, chunks_failed=0
        )
        assert result.chunks_failed == 0

    def test_equality_on_identical_instances(self) -> None:
        a = _make_ingest_result()
        b = _make_ingest_result()
        assert a == b

    def test_inequality_on_different_source(self) -> None:
        a = _make_ingest_result(source="data/a.json")
        b = _make_ingest_result(source="data/b.json")
        assert a != b


# ── IngestionPipeline.__init__ tests ─────────────────────────────────────────


class TestIngestionPipelineInit:
    def test_valid_construction_succeeds(self, tmp_path: Path) -> None:
        pipeline, _ = _make_pipeline(tmp_path)
        assert pipeline is not None

    def test_alpha_below_zero_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="alpha"):
            _make_pipeline(tmp_path, alpha=-0.1)

    def test_alpha_above_one_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="alpha"):
            _make_pipeline(tmp_path, alpha=1.1)

    def test_alpha_zero_is_valid(self, tmp_path: Path) -> None:
        pipeline, _ = _make_pipeline(tmp_path, alpha=0.0)
        assert pipeline is not None

    def test_alpha_one_is_valid(self, tmp_path: Path) -> None:
        pipeline, _ = _make_pipeline(tmp_path, alpha=1.0)
        assert pipeline is not None

    def test_batch_size_zero_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            _make_pipeline(tmp_path, batch_size=0)

    def test_batch_size_negative_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            _make_pipeline(tmp_path, batch_size=-1)

    def test_batch_size_one_is_valid(self, tmp_path: Path) -> None:
        pipeline, _ = _make_pipeline(tmp_path, batch_size=1)
        assert pipeline is not None


# ── IngestionPipeline.run — result fields ─────────────────────────────────────


class TestIngestionPipelineRunResult:
    def test_run_returns_ingest_result(self, tmp_path: Path) -> None:
        src = _write_chunks_json(tmp_path / "chunks.json", n=3)
        pipeline, store = _make_pipeline(tmp_path, upserted=3)

        result = pipeline.run(src)

        assert isinstance(result, IngestResult)

    def test_source_path_in_result(self, tmp_path: Path) -> None:
        src = _write_chunks_json(tmp_path / "chunks.json", n=3)
        pipeline, store = _make_pipeline(tmp_path, upserted=3)

        result = pipeline.run(src)

        assert result.source == str(src)

    def test_collection_name_in_result(self, tmp_path: Path) -> None:
        src = _write_chunks_json(tmp_path / "chunks.json", n=3)
        pipeline, store = _make_pipeline(tmp_path, upserted=3)

        result = pipeline.run(src)

        assert result.collection_name == "test-collection"

    def test_chunks_loaded_matches_json_count(self, tmp_path: Path) -> None:
        src = _write_chunks_json(tmp_path / "chunks.json", n=5)
        pipeline, store = _make_pipeline(tmp_path, upserted=5)

        result = pipeline.run(src)

        assert result.chunks_loaded == 5

    def test_chunks_upserted_from_store_result(self, tmp_path: Path) -> None:
        src = _write_chunks_json(tmp_path / "chunks.json", n=3)
        pipeline, store = _make_pipeline(tmp_path, upserted=3, failed=0)

        result = pipeline.run(src)

        assert result.chunks_upserted == 3

    def test_chunks_failed_from_store_result(self, tmp_path: Path) -> None:
        src = _write_chunks_json(tmp_path / "chunks.json", n=3)
        pipeline, store = _make_pipeline(tmp_path, upserted=2, failed=1)

        result = pipeline.run(src)

        assert result.chunks_failed == 1

    def test_bm25_cache_path_is_non_empty(self, tmp_path: Path) -> None:
        src = _write_chunks_json(tmp_path / "chunks.json", n=3)
        pipeline, _ = _make_pipeline(tmp_path)

        result = pipeline.run(src)

        assert result.bm25_cache_path != ""


# ── IngestionPipeline.run — vector store interactions ─────────────────────────


class TestIngestionPipelineRunVectorStore:
    def test_creates_collection_when_missing(self, tmp_path: Path) -> None:
        src = _write_chunks_json(tmp_path / "chunks.json", n=3)
        pipeline, store = _make_pipeline(tmp_path, store_exists=False, upserted=3)

        pipeline.run(src)

        store.create_collection.assert_called_once()

    def test_does_not_create_collection_when_exists(self, tmp_path: Path) -> None:
        src = _write_chunks_json(tmp_path / "chunks.json", n=3)
        pipeline, store = _make_pipeline(tmp_path, store_exists=True, upserted=3)

        pipeline.run(src)

        store.create_collection.assert_not_called()

    def test_upsert_called_once(self, tmp_path: Path) -> None:
        src = _write_chunks_json(tmp_path / "chunks.json", n=3)
        pipeline, store = _make_pipeline(tmp_path, upserted=3)

        pipeline.run(src)

        store.upsert.assert_called_once()

    def test_upsert_receives_all_chunks(self, tmp_path: Path) -> None:
        n = 4
        src = _write_chunks_json(tmp_path / "chunks.json", n=n)
        pipeline, store = _make_pipeline(tmp_path, upserted=n)

        pipeline.run(src)

        chunks_arg, embeddings_arg = store.upsert.call_args.args
        assert len(chunks_arg) == n
        assert len(embeddings_arg) == n

    def test_create_collection_receives_dense_dim(self, tmp_path: Path) -> None:
        src = _write_chunks_json(tmp_path / "chunks.json", n=2)
        pipeline, store = _make_pipeline(tmp_path, store_exists=False, upserted=2)

        pipeline.run(src)

        assert (
            store.create_collection.call_args.kwargs["dense_dim"] == 4
        )  # matches _make_dense_mock


# ── IngestionPipeline.run — error handling ────────────────────────────────────


class TestIngestionPipelineRunErrors:
    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        pipeline, _ = _make_pipeline(tmp_path)

        with pytest.raises(FileNotFoundError):
            pipeline.run(tmp_path / "missing.json")

    def test_invalid_json_raises_value_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json", encoding="utf-8")
        pipeline, _ = _make_pipeline(tmp_path)

        with pytest.raises(ValueError):
            pipeline.run(bad)

    def test_non_json_extension_raises_value_error(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "chunks.txt"
        txt_file.write_text("[]", encoding="utf-8")
        pipeline, _ = _make_pipeline(tmp_path)

        with pytest.raises(ValueError):
            pipeline.run(txt_file)


# ── IngestionPipeline — BM25 serialization ────────────────────────────────────


class TestIngestionPipelineBM25Serialization:
    def test_bm25_pickle_file_is_created(self, tmp_path: Path) -> None:
        src = _write_chunks_json(tmp_path / "chunks.json", n=3)
        pipeline, _ = _make_pipeline(tmp_path)

        result = pipeline.run(src)

        assert Path(result.bm25_cache_path).exists()

    def test_bm25_cache_path_stem_matches_source(self, tmp_path: Path) -> None:
        src = _write_chunks_json(tmp_path / "my_corpus.json", n=3)
        pipeline, _ = _make_pipeline(tmp_path)

        result = pipeline.run(src)

        assert Path(result.bm25_cache_path).stem == "my_corpus"

    def test_bm25_is_loadable_via_public_api(self, tmp_path: Path) -> None:
        src = _write_chunks_json(tmp_path / "chunks.json", n=3)
        pipeline, _ = _make_pipeline(tmp_path)

        result = pipeline.run(src)

        loaded = BM25SparseEmbedder.load(result.bm25_cache_path)
        assert isinstance(loaded, BM25SparseEmbedder)

    def test_loaded_bm25_is_fitted(self, tmp_path: Path) -> None:
        src = _write_chunks_json(tmp_path / "chunks.json", n=3)
        pipeline, _ = _make_pipeline(tmp_path)

        result = pipeline.run(src)

        loaded = BM25SparseEmbedder.load(result.bm25_cache_path)
        assert loaded.is_fitted

    def test_bm25_cache_dir_is_created_if_missing(self, tmp_path: Path) -> None:
        src = _write_chunks_json(tmp_path / "chunks.json", n=2)
        nested_cache = tmp_path / "deep" / "cache" / "bm25"
        dense = _make_dense_mock()
        sparse = BM25SparseEmbedder()
        store = _make_store_mock()
        store.upsert.return_value = UpsertResult(upserted=2)

        pipeline = IngestionPipeline(
            dense=dense,
            sparse=sparse,
            vector_store=store,
            collection_name="test-collection",
            bm25_cache_dir=nested_cache,
        )
        result = pipeline.run(src)

        assert nested_cache.exists()
        assert Path(result.bm25_cache_path).exists()


# ── IngestionPipeline.load_bm25 ───────────────────────────────────────────────


class TestIngestionPipelineLoadBM25:
    def test_load_bm25_returns_bm25_sparse_embedder(self, tmp_path: Path) -> None:
        src = _write_chunks_json(tmp_path / "chunks.json", n=3)
        pipeline, _ = _make_pipeline(tmp_path)
        result = pipeline.run(src)

        loaded = IngestionPipeline.load_bm25(result.bm25_cache_path)

        assert isinstance(loaded, BM25SparseEmbedder)

    def test_load_bm25_is_fitted(self, tmp_path: Path) -> None:
        src = _write_chunks_json(tmp_path / "chunks.json", n=3)
        pipeline, _ = _make_pipeline(tmp_path)
        result = pipeline.run(src)

        loaded = IngestionPipeline.load_bm25(result.bm25_cache_path)

        assert loaded.is_fitted

    def test_load_bm25_embedding_matches_pipeline_sparse(self, tmp_path: Path) -> None:
        src = _write_chunks_json(tmp_path / "chunks.json", n=3)
        pipeline, _ = _make_pipeline(tmp_path)
        result = pipeline.run(src)

        loaded = IngestionPipeline.load_bm25(result.bm25_cache_path)

        text = "Chunk number 1 with some unique content for BM25 fitting."
        assert loaded.embed_text(text) == pytest.approx(
            pipeline._sparse.embed_text(text)
        )

    def test_load_bm25_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            IngestionPipeline.load_bm25(tmp_path / "missing.pkl")

    def test_load_bm25_wrong_type_raises(self, tmp_path: Path) -> None:
        import pickle

        bad = tmp_path / "bad.pkl"
        with bad.open("wb") as f:
            pickle.dump({"not": "a bm25"}, f)

        with pytest.raises(TypeError):
            IngestionPipeline.load_bm25(bad)


def _make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = dict(
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="rag-test",
        llm_provider="ollama",
        llm_base_url="http://localhost:11434",
        llm_api_key="",
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# IngestionPipeline.build
# ---------------------------------------------------------------------------


class TestIngestionPipelineBuild:
    def test_returns_ingestion_pipeline_instance(self) -> None:
        with (
            patch("src.pipeline.ingest_pipeline.SentenceTransformerEmbedder"),
            patch("src.pipeline.ingest_pipeline.QdrantVectorStore"),
        ):
            result = IngestionPipeline.build(_make_settings())
        assert isinstance(result, IngestionPipeline)

    def test_collection_name_matches_settings(self) -> None:
        with (
            patch("src.pipeline.ingest_pipeline.SentenceTransformerEmbedder"),
            patch("src.pipeline.ingest_pipeline.QdrantVectorStore"),
        ):
            pipeline = IngestionPipeline.build(_make_settings(collection_name="my-col"))
        assert pipeline._collection_name == "my-col"

    def test_batch_size_matches_embedding_batch_size(self) -> None:
        with (
            patch("src.pipeline.ingest_pipeline.SentenceTransformerEmbedder"),
            patch("src.pipeline.ingest_pipeline.QdrantVectorStore"),
        ):
            pipeline = IngestionPipeline.build(_make_settings(embedding_batch_size=64))
        assert pipeline._batch_size == 64

    def test_bm25_cache_dir_is_parent_of_cache_path(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "bm25" / "model.pkl"
        with (
            patch("src.pipeline.ingest_pipeline.SentenceTransformerEmbedder"),
            patch("src.pipeline.ingest_pipeline.QdrantVectorStore"),
        ):
            pipeline = IngestionPipeline.build(
                _make_settings(bm25_cache_path=cache_path)
            )
        assert pipeline._bm25_cache_dir == cache_path.parent

    def test_qdrant_vector_store_created_with_correct_host(self) -> None:
        with (
            patch("src.pipeline.ingest_pipeline.SentenceTransformerEmbedder"),
            patch("src.pipeline.ingest_pipeline.QdrantVectorStore") as mock_store_cls,
        ):
            IngestionPipeline.build(_make_settings(qdrant_host="qdrant.internal"))
        call_kwargs = mock_store_cls.call_args.kwargs
        assert call_kwargs["host"] == "qdrant.internal"

    def test_qdrant_vector_store_created_with_correct_port(self) -> None:
        with (
            patch("src.pipeline.ingest_pipeline.SentenceTransformerEmbedder"),
            patch("src.pipeline.ingest_pipeline.QdrantVectorStore") as mock_store_cls,
        ):
            IngestionPipeline.build(_make_settings(qdrant_port=9999))
        call_kwargs = mock_store_cls.call_args.kwargs
        assert call_kwargs["port"] == 9999

    def test_dense_embedder_created_with_embedding_model(self) -> None:
        with (
            patch(
                "src.pipeline.ingest_pipeline.SentenceTransformerEmbedder"
            ) as mock_dense_cls,
            patch("src.pipeline.ingest_pipeline.QdrantVectorStore"),
        ):
            IngestionPipeline.build(
                _make_settings(embedding_model="intfloat/multilingual-e5-large")
            )
        mock_dense_cls.assert_called_once_with(
            model_name="intfloat/multilingual-e5-large"
        )


# ---------------------------------------------------------------------------
# IngestionPipeline.run — reset flag
# ---------------------------------------------------------------------------


class TestIngestionPipelineRunReset:
    def test_reset_false_does_not_delete_when_collection_exists(
        self, tmp_path: Path
    ) -> None:
        pipeline, store = _make_pipeline(tmp_path, store_exists=True)
        src = _write_chunks_json(tmp_path / "chunks.json")
        pipeline.run(src, reset=False)
        store.delete_collection.assert_not_called()

    def test_reset_true_deletes_existing_collection(self, tmp_path: Path) -> None:
        pipeline, store = _make_pipeline(tmp_path, store_exists=True)
        src = _write_chunks_json(tmp_path / "chunks.json")
        pipeline.run(src, reset=True)
        store.delete_collection.assert_called_once()

    def test_reset_true_does_not_delete_when_collection_missing(
        self, tmp_path: Path
    ) -> None:
        pipeline, store = _make_pipeline(tmp_path, store_exists=False)
        src = _write_chunks_json(tmp_path / "chunks.json")
        pipeline.run(src, reset=True)
        store.delete_collection.assert_not_called()

    def test_reset_true_creates_collection_after_delete(self, tmp_path: Path) -> None:
        pipeline, store = _make_pipeline(tmp_path, store_exists=True)
        store.collection_exists.side_effect = [True, False]
        src = _write_chunks_json(tmp_path / "chunks.json")
        pipeline.run(src, reset=True)
        store.create_collection.assert_called_once()

    def test_default_reset_is_false(self, tmp_path: Path) -> None:
        pipeline, store = _make_pipeline(tmp_path, store_exists=True)
        src = _write_chunks_json(tmp_path / "chunks.json")
        pipeline.run(src)
        store.delete_collection.assert_not_called()
