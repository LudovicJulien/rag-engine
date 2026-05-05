# tests/vector_store/test_store.py
import pytest

from src.embeddings.hybrid_embedder import HybridEmbedding
from src.shared.models import Chunk, ChunkMetadata
from src.vector_store.store import VectorStore


def _make_chunk() -> Chunk:
    return Chunk(
        chunk_id="chunk-001",
        parent_doc_id="doc-001",
        text="Sample text content",
        chunk_index=0,
        total_chunks=1,
        metadata=ChunkMetadata(),
    )


def _make_embedding() -> HybridEmbedding:
    return HybridEmbedding(
        dense=[0.1, 0.2, 0.3],
        sparse_indices=[0, 5],
        sparse_values=[0.8, 0.3],
        text="Sample text content",
    )


class TestValidateUpsertInputs:
    def test_raises_on_empty_chunks(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            VectorStore._validate_upsert_inputs([], [])

    def test_raises_on_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            VectorStore._validate_upsert_inputs([_make_chunk()], [])

    def test_passes_on_valid_inputs(self) -> None:
        VectorStore._validate_upsert_inputs([_make_chunk()], [_make_embedding()])
