# tests/vector_store/test_qdrant_store.py
from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from httpx import Headers
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PayloadSchemaType

from src.embeddings.hybrid_embedder import HybridEmbedding
from src.shared.models import Chunk, ChunkMetadata, MetadataFilter
from src.vector_store.qdrant_store import QdrantVectorStore


def _make_chunk(chunk_id: str = "chunk-001") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
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


@pytest.fixture
def mock_client() -> Generator[MagicMock, None, None]:
    """Patch QdrantClient and yield the mock instance."""
    with patch("src.vector_store.qdrant_store.QdrantClient") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def store(mock_client: MagicMock) -> QdrantVectorStore:
    """Return a QdrantVectorStore backed by a mocked Qdrant client."""
    return QdrantVectorStore(
        host="localhost",
        port=6333,
        collection_name="test-col",
    )


class TestQdrantVectorStoreConnect:
    """Tests for QdrantVectorStore connection and retry logic."""

    @patch("src.vector_store.qdrant_store.QdrantClient")
    def test_connect_verifies_connection_on_init(self, mock_cls: MagicMock) -> None:
        """get_collections() is called on init to verify the connection."""
        instance = MagicMock()
        mock_cls.return_value = instance

        QdrantVectorStore(host="localhost", port=6333, collection_name="test")

        instance.get_collections.assert_called_once()

    @patch("src.vector_store.qdrant_store.time.sleep")
    @patch("src.vector_store.qdrant_store.QdrantClient")
    def test_connect_retries_on_transient_failure(
        self, mock_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Constructor retries when get_collections() fails once then succeeds."""
        instance = MagicMock()
        instance.get_collections.side_effect = [
            ConnectionError("transient"),
            MagicMock(),
        ]
        mock_cls.return_value = instance

        QdrantVectorStore(
            host="localhost",
            port=6333,
            collection_name="test",
            max_retries=3,
            retry_delay=1.0,
        )

        assert instance.get_collections.call_count == 2
        mock_sleep.assert_called_once_with(1.0)  # base_delay * 2^0

    @patch("src.vector_store.qdrant_store.time.sleep")
    @patch("src.vector_store.qdrant_store.QdrantClient")
    def test_connect_raises_after_max_retries(
        self, mock_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Constructor raises ConnectionError after exhausting all retries."""
        instance = MagicMock()
        instance.get_collections.side_effect = ConnectionError("refused")
        mock_cls.return_value = instance

        with pytest.raises(
            ConnectionError, match="Failed to connect to Qdrant after 3 attempts"
        ):
            QdrantVectorStore(
                host="localhost",
                port=6333,
                collection_name="test",
                max_retries=3,
                retry_delay=1.0,
            )

        assert instance.get_collections.call_count == 3
        assert mock_sleep.call_count == 2  # 2 sleeps between 3 attempts

    @patch("src.vector_store.qdrant_store.time.sleep")
    @patch("src.vector_store.qdrant_store.QdrantClient")
    def test_connect_uses_exponential_backoff(
        self, mock_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Retry delays follow exponential backoff: base_delay * 2^attempt."""
        instance = MagicMock()
        instance.get_collections.side_effect = ConnectionError("refused")
        mock_cls.return_value = instance

        with pytest.raises(ConnectionError):
            QdrantVectorStore(
                host="localhost",
                port=6333,
                collection_name="test",
                max_retries=3,
                retry_delay=2.0,
            )

        sleep_delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_delays == [2.0, 4.0]  # 2.0*2^0, 2.0*2^1

    @patch("src.vector_store.qdrant_store.time.sleep")
    @patch("src.vector_store.qdrant_store.QdrantClient")
    def test_connect_single_attempt_no_sleep(
        self, mock_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """No sleep when max_retries=1 — only one attempt is made."""
        instance = MagicMock()
        instance.get_collections.side_effect = ConnectionError("refused")
        mock_cls.return_value = instance

        with pytest.raises(ConnectionError):
            QdrantVectorStore(
                host="localhost",
                port=6333,
                collection_name="test",
                max_retries=1,
                retry_delay=1.0,
            )

        mock_sleep.assert_not_called()


class TestQdrantVectorStoreCreateCollection:
    """Tests for QdrantVectorStore.create_collection()."""

    @pytest.fixture(autouse=True)
    def collection_does_not_exist(self, mock_client: MagicMock) -> None:
        from httpx import Headers
        from qdrant_client.http.exceptions import UnexpectedResponse

        mock_client.get_collection.side_effect = UnexpectedResponse(
            status_code=404, reason_phrase="Not Found", content=b"", headers=Headers()
        )

    def test_create_collection_calls_client(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        mock_client.get_collection.side_effect = UnexpectedResponse(
            status_code=404,
            reason_phrase="Not Found",
            content=b"",
            headers=Headers(),
        )
        store.create_collection(dense_dim=1024)
        mock_client.create_collection.assert_called_once()

    def test_create_collection_uses_correct_collection_name(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        store.create_collection(dense_dim=512)
        kwargs = mock_client.create_collection.call_args.kwargs
        assert kwargs["collection_name"] == "test-col"

    def test_create_collection_configures_dense_named_vector(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        store.create_collection(dense_dim=768)
        kwargs = mock_client.create_collection.call_args.kwargs
        vectors_config = kwargs["vectors_config"]
        assert "dense" in vectors_config
        assert vectors_config["dense"].size == 768

    def test_create_collection_configures_sparse_named_vector(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        store.create_collection(dense_dim=768)
        kwargs = mock_client.create_collection.call_args.kwargs
        assert "sparse" in kwargs["sparse_vectors_config"]


class TestQdrantVectorStoreDeleteCollection:
    """Tests for QdrantVectorStore.delete_collection()."""

    def test_delete_collection_calls_client_with_collection_name(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        store.delete_collection()
        mock_client.delete_collection.assert_called_once_with(
            collection_name="test-col"
        )


class TestQdrantVectorStoreCollectionExists:
    """Tests for QdrantVectorStore.collection_exists()."""

    def test_collection_exists_returns_true_when_found(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        mock_client.get_collection.return_value = MagicMock()
        assert store.collection_exists() is True

    def test_collection_exists_returns_false_on_exception(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        mock_client.get_collection.side_effect = UnexpectedResponse(
            status_code=404, reason_phrase="Not Found", content=b"", headers=Headers()
        )
        assert store.collection_exists() is False


class TestQdrantVectorStoreUpsert:
    """Tests for QdrantVectorStore.upsert()."""

    def test_upsert_returns_upserted_count(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        chunks = [_make_chunk("c1"), _make_chunk("c2")]
        embeddings = [_make_embedding(), _make_embedding()]

        result = store.upsert(chunks, embeddings)

        assert result.upserted == 2
        mock_client.upsert.assert_called_once()

    def test_upsert_empty_list_raises(self, store: QdrantVectorStore) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            store.upsert([], [])

    def test_upsert_mismatched_lengths_raises(self, store: QdrantVectorStore) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            store.upsert([_make_chunk()], [])

    def test_upsert_payload_contains_chunk_fields(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        chunk = _make_chunk("chunk-xyz")
        embedding = _make_embedding()

        store.upsert([chunk], [embedding])

        points = mock_client.upsert.call_args.kwargs["points"]
        payload = points[0].payload
        assert payload["chunk_id"] == "chunk-xyz"
        assert payload["text"] == "Sample text content"
        assert payload["parent_doc_id"] == "doc-001"
        assert payload["chunk_index"] == 0
        assert payload["total_chunks"] == 1

    def test_upsert_uses_correct_collection_name(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        store.upsert([_make_chunk()], [_make_embedding()])
        kwargs = mock_client.upsert.call_args.kwargs
        assert kwargs["collection_name"] == "test-col"

    def test_upsert_point_ids_are_deterministic(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        """Same chunk_id always produces the same Qdrant point ID."""
        store.upsert([_make_chunk("c1")], [_make_embedding()])
        first_id = mock_client.upsert.call_args.kwargs["points"][0].id

        mock_client.reset_mock()
        store.upsert([_make_chunk("c1")], [_make_embedding()])
        second_id = mock_client.upsert.call_args.kwargs["points"][0].id

        assert first_id == second_id


class TestQdrantVectorStoreSearch:
    """Tests for QdrantVectorStore.search()."""

    @staticmethod
    def _make_mock_point(chunk_id: str = "c1") -> MagicMock:
        point = MagicMock()
        point.payload = {
            "chunk_id": chunk_id,
            "parent_doc_id": "doc-001",
            "text": "Sample text content",
            "chunk_index": 0,
            "total_chunks": 1,
            "metadata": {},
        }
        return point

    def test_search_returns_chunks(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.points = [
            self._make_mock_point("c1"),
            self._make_mock_point("c2"),
        ]
        mock_client.query_points.return_value = mock_response

        results = store.search(_make_embedding(), top_k=5)

        assert len(results) == 2
        assert all(isinstance(r.chunk, Chunk) for r in results)
        assert results[0].chunk.chunk_id == "c1"
        assert results[1].chunk.chunk_id == "c2"

    def test_search_empty_result_returns_empty_list(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        assert store.search(_make_embedding(), top_k=5) == []

    def test_search_passes_top_k_as_limit(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        mock_client.query_points.return_value.points = []

        store.search(_make_embedding(), top_k=10)

        kwargs = mock_client.query_points.call_args.kwargs
        assert kwargs["limit"] == 10

    def test_search_passes_none_threshold_when_zero(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        """score_threshold=0.0 is converted to None so Qdrant returns all results."""
        mock_client.query_points.return_value.points = []

        store.search(_make_embedding(), top_k=5, score_threshold=0.0)

        kwargs = mock_client.query_points.call_args.kwargs
        assert kwargs["score_threshold"] is None

    def test_search_passes_nonzero_threshold(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        mock_client.query_points.return_value.points = []

        store.search(_make_embedding(), top_k=5, score_threshold=0.7)

        kwargs = mock_client.query_points.call_args.kwargs
        assert kwargs["score_threshold"] == pytest.approx(0.7)

    def test_search_reconstructs_chunk_metadata(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        point = self._make_mock_point()
        point.payload["metadata"] = {"language": "fr"}
        mock_response = MagicMock()
        mock_response.points = [point]
        mock_client.query_points.return_value = mock_response

        results = store.search(_make_embedding(), top_k=1)

        assert results[0].chunk.metadata.metadata.get("language") == "fr"

    def test_search_passes_no_filter_when_filters_none(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        mock_client.query_points.return_value.points = []

        store.search(_make_embedding(), top_k=5, filters=None)

        kwargs = mock_client.query_points.call_args.kwargs
        assert kwargs["query_filter"] is None

    def test_search_passes_qdrant_filter_when_filters_provided(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        from qdrant_client.models import FieldCondition

        mock_client.query_points.return_value.points = []

        store.search(
            _make_embedding(),
            top_k=5,
            filters=[MetadataFilter(field="metadata.language", value="fr")],
        )
        kwargs = mock_client.query_points.call_args.kwargs
        must = kwargs["query_filter"].must
        assert isinstance(must, list)
        assert len(must) == 1
        assert isinstance(must[0], FieldCondition)

    def test_search_filter_uses_correct_field_and_value(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        from qdrant_client.models import MatchValue

        mock_client.query_points.return_value.points = []

        store.search(
            _make_embedding(),
            top_k=5,
            filters=[MetadataFilter(field="metadata.language", value="fr")],
        )

        condition = mock_client.query_points.call_args.kwargs["query_filter"].must[0]
        assert condition.key == "metadata.language"
        assert condition.match == MatchValue(value="fr")

    def test_search_ands_multiple_filters(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        mock_client.query_points.return_value.points = []

        store.search(
            _make_embedding(),
            top_k=5,
            filters=[
                MetadataFilter(field="metadata.language", value="fr"),
                MetadataFilter(field="parent_doc_id", value="doc-42"),
            ],
        )

        must_conditions = mock_client.query_points.call_args.kwargs["query_filter"].must
        assert len(must_conditions) == 2


class TestQdrantVectorStoreHealthCheck:
    """Tests for QdrantVectorStore.health_check()."""

    def test_health_check_returns_true_when_reachable(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        mock_client.get_collections.return_value = MagicMock()
        assert store.health_check() is True

    def test_health_check_returns_false_when_unreachable(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        mock_client.get_collections.side_effect = ConnectionError("refused")
        assert store.health_check() is False


class TestQdrantVectorStoreCount:
    """Tests for QdrantVectorStore.count()."""

    def test_count_returns_point_count(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        mock_result = MagicMock()
        mock_result.count = 42
        mock_client.count.return_value = mock_result

        assert store.count() == 42

    def test_count_uses_exact_mode(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        mock_client.count.return_value.count = 0

        store.count()

        kwargs = mock_client.count.call_args.kwargs
        assert kwargs["exact"] is True
        assert kwargs["collection_name"] == "test-col"


class TestQdrantVectorStoreDeleteChunkById:
    """Tests for QdrantVectorStore.delete_chunk_by_id()."""

    def test_delete_chunk_by_id_calls_client_delete(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        mock_client.retrieve.return_value = [MagicMock()]
        store.delete_chunk_by_id("chunk-001")
        mock_client.delete.assert_called_once()

    def test_delete_chunk_by_id_uses_deterministic_point_id(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        """Same chunk_id always produces the same Qdrant point ID."""
        mock_client.retrieve.return_value = [MagicMock()]
        store.delete_chunk_by_id("chunk-abc")
        first_selector = mock_client.delete.call_args.kwargs["points_selector"]

        mock_client.reset_mock()
        mock_client.retrieve.return_value = [MagicMock()]
        store.delete_chunk_by_id("chunk-abc")
        second_selector = mock_client.delete.call_args.kwargs["points_selector"]

        assert first_selector.points == second_selector.points

    def test_delete_chunk_by_id_returns_true_when_deleted(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        mock_client.retrieve.return_value = [MagicMock()]
        result = store.delete_chunk_by_id("chunk-001")
        assert result is True

    def test_delete_chunk_by_id_returns_false_when_not_found(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        mock_client.retrieve.return_value = []
        result = store.delete_chunk_by_id("chunk-001")
        assert result is False
        mock_client.delete.assert_not_called()


class TestQdrantVectorStoreCreatePayloadIndex:
    """Tests for QdrantVectorStore.create_payload_index()."""

    def test_create_payload_index_calls_client(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        store.create_payload_index("language")
        mock_client.create_payload_index.assert_called_once()

    def test_create_payload_index_uses_correct_field_name(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        store.create_payload_index("season")
        kwargs = mock_client.create_payload_index.call_args.kwargs
        assert kwargs["field_name"] == "season"

    def test_create_payload_index_uses_keyword_schema(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        store.create_payload_index("language")
        kwargs = mock_client.create_payload_index.call_args.kwargs
        assert kwargs["field_schema"] == PayloadSchemaType.KEYWORD

    def test_create_payload_index_uses_correct_collection_name(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        store.create_payload_index("target_audience")
        kwargs = mock_client.create_payload_index.call_args.kwargs
        assert kwargs["collection_name"] == "test-col"


class TestQdrantVectorStoreScrollAllChunks:
    """Tests for QdrantVectorStore.scroll_all_chunks()."""

    @staticmethod
    def _make_mock_record(chunk_id: str = "c1") -> MagicMock:
        record = MagicMock()
        record.payload = {
            "chunk_id": chunk_id,
            "parent_doc_id": "doc-001",
            "text": "Sample text content",
            "chunk_index": 0,
            "total_chunks": 1,
            "metadata": {},
        }
        return record

    def test_scroll_all_chunks_returns_empty_list_when_no_points(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        mock_client.scroll.return_value = ([], None)
        result = store.scroll_all_chunks()
        assert result == []

    def test_scroll_all_chunks_returns_all_chunks_single_page(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        records = [self._make_mock_record("c1"), self._make_mock_record("c2")]
        mock_client.scroll.return_value = (records, None)

        result = store.scroll_all_chunks()

        assert len(result) == 2
        assert all(isinstance(c, Chunk) for c in result)

    def test_scroll_all_chunks_paginates_until_offset_is_none(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        page1 = [self._make_mock_record("c1")]
        page2 = [self._make_mock_record("c2")]
        mock_client.scroll.side_effect = [(page1, "offset-1"), (page2, None)]

        result = store.scroll_all_chunks()

        assert mock_client.scroll.call_count == 2
        assert len(result) == 2

    def test_scroll_all_chunks_reconstructs_chunk_fields(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ) -> None:
        record = self._make_mock_record("chunk-xyz")
        record.payload["text"] = "Hello world"
        mock_client.scroll.return_value = ([record], None)

        result = store.scroll_all_chunks()

        assert result[0].chunk_id == "chunk-xyz"
        assert result[0].text == "Hello world"
