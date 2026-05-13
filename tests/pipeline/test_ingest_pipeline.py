# tests/pipeline/test_ingest_pipeline.py
from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from src.pipeline.ingest_pipeline import IngestResult


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
