from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from qdrant_client.http.exceptions import UnexpectedResponse

from src.cli.commands import (
    ConfigStatus,
    DiagnosticsReport,
    HealthStatus,
    ServiceHealth,
    _get_bm25_vocab_size,
    _get_collection_count,
    _probe_bm25_cache,
    _probe_collection,
    build_diagnostics,
    check_config,
)
from src.pipeline.config import Settings


def _settings(**overrides: object) -> Settings:
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


def _not_found() -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=404,
        reason_phrase="Not Found",
        content=b"",
        headers=httpx.Headers({}),
    )


def _ok_svc(detail: str = "ok") -> ServiceHealth:
    return ServiceHealth(ok=True, detail=detail)


def _fail_svc(detail: str = "fail") -> ServiceHealth:
    return ServiceHealth(ok=False, detail=detail)


# ---------------------------------------------------------------------------
# _probe_bm25_cache
# ---------------------------------------------------------------------------


class TestProbeBm25Cache:
    def test_ok_true_when_file_exists(self, tmp_path: Path) -> None:
        pkl = tmp_path / "bm25.pkl"
        pkl.write_bytes(b"dummy")
        result = _probe_bm25_cache(_settings(bm25_cache_path=pkl))
        assert result.ok is True

    def test_detail_contains_path_on_success(self, tmp_path: Path) -> None:
        pkl = tmp_path / "bm25.pkl"
        pkl.write_bytes(b"dummy")
        result = _probe_bm25_cache(_settings(bm25_cache_path=pkl))
        assert str(pkl) in result.detail

    def test_ok_false_when_file_missing(self, tmp_path: Path) -> None:
        result = _probe_bm25_cache(_settings(bm25_cache_path=tmp_path / "gone.pkl"))
        assert result.ok is False

    def test_detail_contains_path_on_failure(self, tmp_path: Path) -> None:
        path = tmp_path / "gone.pkl"
        result = _probe_bm25_cache(_settings(bm25_cache_path=path))
        assert str(path) in result.detail

    def test_returns_service_health_instance(self, tmp_path: Path) -> None:
        pkl = tmp_path / "bm25.pkl"
        pkl.write_bytes(b"dummy")
        result = _probe_bm25_cache(_settings(bm25_cache_path=pkl))
        assert isinstance(result, ServiceHealth)


# ---------------------------------------------------------------------------
# _probe_collection
# ---------------------------------------------------------------------------


class TestProbeCollection:
    def test_ok_true_when_collection_exists(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.get_collection.return_value = MagicMock()
            result = _probe_collection(_settings())
        assert result.ok is True

    def test_detail_contains_collection_name_on_success(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.get_collection.return_value = MagicMock()
            result = _probe_collection(_settings(collection_name="my-col"))
        assert "my-col" in result.detail

    def test_ok_false_when_collection_not_found(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.get_collection.side_effect = _not_found()
            result = _probe_collection(_settings())
        assert result.ok is False

    def test_detail_says_not_found_for_missing_collection(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.get_collection.side_effect = _not_found()
            result = _probe_collection(_settings(collection_name="my-col"))
        assert "not found" in result.detail
        assert "my-col" in result.detail

    def test_ok_false_when_qdrant_unreachable(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.get_collection.side_effect = ConnectionError("down")
            result = _probe_collection(_settings())
        assert result.ok is False

    def test_detail_says_unreachable_when_qdrant_down(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.get_collection.side_effect = OSError("refused")
            result = _probe_collection(_settings())
        assert "unreachable" in result.detail

    def test_detail_contains_addr_when_qdrant_down(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.get_collection.side_effect = OSError("refused")
            result = _probe_collection(_settings(qdrant_host="db", qdrant_port=9999))
        assert "db:9999" in result.detail

    def test_never_raises(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.side_effect = RuntimeError("unexpected")
            result = _probe_collection(_settings())
        assert result.ok is False


# ---------------------------------------------------------------------------
# check_config
# ---------------------------------------------------------------------------


class TestCheckConfig:
    def test_returns_config_status_instance(self, tmp_path: Path) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.get_collection.return_value = MagicMock()
            pkl = tmp_path / "bm25.pkl"
            pkl.write_bytes(b"x")
            result = check_config(_settings(bm25_cache_path=pkl))
        assert isinstance(result, ConfigStatus)

    def test_all_ok_when_file_exists_and_collection_exists(
        self, tmp_path: Path
    ) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.get_collection.return_value = MagicMock()
            pkl = tmp_path / "bm25.pkl"
            pkl.write_bytes(b"x")
            result = check_config(_settings(bm25_cache_path=pkl))
        assert result.all_ok is True

    def test_not_ok_when_bm25_file_missing(self, tmp_path: Path) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.get_collection.return_value = MagicMock()
            result = check_config(_settings(bm25_cache_path=tmp_path / "missing.pkl"))
        assert result.bm25_cache.ok is False
        assert result.all_ok is False

    def test_not_ok_when_collection_missing(self, tmp_path: Path) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.get_collection.side_effect = _not_found()
            pkl = tmp_path / "bm25.pkl"
            pkl.write_bytes(b"x")
            result = check_config(_settings(bm25_cache_path=pkl))
        assert result.collection.ok is False
        assert result.all_ok is False

    def test_not_ok_when_qdrant_unreachable(self, tmp_path: Path) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.get_collection.side_effect = ConnectionError()
            pkl = tmp_path / "bm25.pkl"
            pkl.write_bytes(b"x")
            result = check_config(_settings(bm25_cache_path=pkl))
        assert result.collection.ok is False


# ---------------------------------------------------------------------------
# _get_collection_count
# ---------------------------------------------------------------------------


class TestGetCollectionCount:
    def test_returns_count_when_accessible(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.count.return_value = MagicMock(count=142)
            result = _get_collection_count(_settings())
        assert result == 142

    def test_returns_zero_count_correctly(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.count.return_value = MagicMock(count=0)
            result = _get_collection_count(_settings())
        assert result == 0

    def test_returns_none_on_connection_error(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.count.side_effect = ConnectionError("down")
            result = _get_collection_count(_settings())
        assert result is None

    def test_returns_none_when_client_init_fails(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.side_effect = RuntimeError("unexpected")
            result = _get_collection_count(_settings())
        assert result is None


# ---------------------------------------------------------------------------
# _get_bm25_vocab_size
# ---------------------------------------------------------------------------


class TestGetBm25VocabSize:
    def test_returns_embedding_dim_when_file_valid(self) -> None:
        mock_bm25 = MagicMock()
        mock_bm25.embedding_dim = 4821
        with patch("src.cli.commands.BM25SparseEmbedder.load", return_value=mock_bm25):
            result = _get_bm25_vocab_size(_settings())
        assert result == 4821

    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        result = _get_bm25_vocab_size(
            _settings(bm25_cache_path=tmp_path / "missing.pkl")
        )
        assert result is None

    def test_returns_none_on_corrupt_file(self) -> None:
        with patch(
            "src.cli.commands.BM25SparseEmbedder.load",
            side_effect=TypeError("bad pickle"),
        ):
            result = _get_bm25_vocab_size(_settings())
        assert result is None

    def test_returns_none_on_any_unexpected_error(self) -> None:
        with patch(
            "src.cli.commands.BM25SparseEmbedder.load",
            side_effect=RuntimeError("unexpected"),
        ):
            result = _get_bm25_vocab_size(_settings())
        assert result is None


# ---------------------------------------------------------------------------
# build_diagnostics
# ---------------------------------------------------------------------------


class TestBuildDiagnostics:
    def _healthy(self) -> HealthStatus:
        ok = _ok_svc()
        return HealthStatus(qdrant=ok, llm=ok)

    def _qdrant_down(self) -> HealthStatus:
        return HealthStatus(qdrant=_fail_svc(), llm=_ok_svc())

    def _config_ok(self) -> ConfigStatus:
        ok = _ok_svc()
        return ConfigStatus(bm25_cache=ok, collection=ok)

    def _config_fail(self) -> ConfigStatus:
        return ConfigStatus(bm25_cache=_fail_svc(), collection=_fail_svc())

    def test_returns_diagnostics_report_instance(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=self._healthy()),
            patch("src.cli.commands.check_config", return_value=self._config_ok()),
            patch("src.cli.commands._get_collection_count", return_value=10),
            patch("src.cli.commands._get_bm25_vocab_size", return_value=500),
        ):
            result = build_diagnostics(_settings())
        assert isinstance(result, DiagnosticsReport)

    def test_collection_count_populated_when_qdrant_ok(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=self._healthy()),
            patch("src.cli.commands.check_config", return_value=self._config_ok()),
            patch("src.cli.commands._get_collection_count", return_value=142),
            patch("src.cli.commands._get_bm25_vocab_size", return_value=500),
        ):
            result = build_diagnostics(_settings())
        assert result.collection_count == 142

    def test_collection_count_is_none_when_qdrant_down(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=self._qdrant_down()),
            patch("src.cli.commands.check_config", return_value=self._config_ok()),
            patch("src.cli.commands._get_collection_count") as mock_count,
            patch("src.cli.commands._get_bm25_vocab_size", return_value=None),
        ):
            result = build_diagnostics(_settings())
        assert result.collection_count is None
        mock_count.assert_not_called()

    def test_bm25_vocab_size_populated_when_file_present(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=self._healthy()),
            patch("src.cli.commands.check_config", return_value=self._config_ok()),
            patch("src.cli.commands._get_collection_count", return_value=10),
            patch("src.cli.commands._get_bm25_vocab_size", return_value=4821),
        ):
            result = build_diagnostics(_settings())
        assert result.bm25_vocab_size == 4821

    def test_bm25_vocab_size_is_none_when_file_missing(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=self._healthy()),
            patch("src.cli.commands.check_config", return_value=self._config_ok()),
            patch("src.cli.commands._get_collection_count", return_value=None),
            patch("src.cli.commands._get_bm25_vocab_size", return_value=None),
        ):
            result = build_diagnostics(_settings())
        assert result.bm25_vocab_size is None

    def test_all_ok_true_when_all_checks_pass(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=self._healthy()),
            patch("src.cli.commands.check_config", return_value=self._config_ok()),
            patch("src.cli.commands._get_collection_count", return_value=10),
            patch("src.cli.commands._get_bm25_vocab_size", return_value=500),
        ):
            result = build_diagnostics(_settings())
        assert result.all_ok is True

    def test_all_ok_false_when_qdrant_down(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=self._qdrant_down()),
            patch("src.cli.commands.check_config", return_value=self._config_ok()),
            patch("src.cli.commands._get_collection_count", return_value=None),
            patch("src.cli.commands._get_bm25_vocab_size", return_value=None),
        ):
            result = build_diagnostics(_settings())
        assert result.all_ok is False

    def test_all_ok_false_when_config_fails(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=self._healthy()),
            patch("src.cli.commands.check_config", return_value=self._config_fail()),
            patch("src.cli.commands._get_collection_count", return_value=10),
            patch("src.cli.commands._get_bm25_vocab_size", return_value=None),
        ):
            result = build_diagnostics(_settings())
        assert result.all_ok is False

    def test_health_and_config_embedded_in_report(self) -> None:
        health = self._healthy()
        config = self._config_ok()
        with (
            patch("src.cli.commands.check_health", return_value=health),
            patch("src.cli.commands.check_config", return_value=config),
            patch("src.cli.commands._get_collection_count", return_value=0),
            patch("src.cli.commands._get_bm25_vocab_size", return_value=0),
        ):
            result = build_diagnostics(_settings())
        assert result.health is health
        assert result.config is config
