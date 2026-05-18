from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.cli.commands import (
    HealthStatus,
    ServiceHealth,
    _probe_api_key,
    _probe_llm,
    _probe_ollama,
    _probe_qdrant,
    check_health,
)
from src.pipeline.config import Settings


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = dict(
        qdrant_host="localhost",
        qdrant_port=6333,
        llm_provider="ollama",
        llm_base_url="http://localhost:11434",
        llm_api_key="",
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _probe_qdrant
# ---------------------------------------------------------------------------


class TestProbeQdrant:
    def test_returns_ok_true_when_connected(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.get_collections.return_value = MagicMock()
            result = _probe_qdrant(_settings())
        assert result.ok is True

    def test_detail_contains_host_and_port_on_success(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.get_collections.return_value = MagicMock()
            result = _probe_qdrant(_settings(qdrant_host="myhost", qdrant_port=6334))
        assert "myhost:6334" in result.detail

    def test_returns_ok_false_on_connection_error(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.get_collections.side_effect = ConnectionError("down")
            result = _probe_qdrant(_settings())
        assert result.ok is False

    def test_detail_contains_addr_on_failure(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.get_collections.side_effect = OSError("refused")
            result = _probe_qdrant(_settings())
        assert "localhost:6333" in result.detail

    def test_never_raises_on_unexpected_exception(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.side_effect = RuntimeError("unexpected")
            result = _probe_qdrant(_settings())
        assert result.ok is False

    def test_returns_service_health_instance(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_cls:
            mock_cls.return_value.get_collections.return_value = MagicMock()
            result = _probe_qdrant(_settings())
        assert isinstance(result, ServiceHealth)


# ---------------------------------------------------------------------------
# _probe_ollama
# ---------------------------------------------------------------------------


class TestProbeOllama:
    def test_returns_ok_true_when_reachable(self) -> None:
        with patch("src.cli.commands.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            result = _probe_ollama("http://localhost:11434")
        assert result.ok is True

    def test_detail_contains_base_url_on_success(self) -> None:
        with patch("src.cli.commands.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            result = _probe_ollama("http://myserver:11434")
        assert "http://myserver:11434" in result.detail

    def test_hits_api_tags_endpoint(self) -> None:
        with patch("src.cli.commands.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            _probe_ollama("http://localhost:11434")
        url = mock_get.call_args.args[0]
        assert url.endswith("/api/tags")

    def test_returns_ok_false_on_connect_error(self) -> None:
        with patch("src.cli.commands.httpx.get") as mock_get:
            mock_get.side_effect = OSError("connection refused")
            result = _probe_ollama("http://localhost:11434")
        assert result.ok is False

    def test_returns_ok_false_on_timeout(self) -> None:
        with patch("src.cli.commands.httpx.get") as mock_get:
            mock_get.side_effect = TimeoutError("timed out")
            result = _probe_ollama("http://localhost:11434")
        assert result.ok is False

    def test_detail_contains_base_url_on_failure(self) -> None:
        with patch("src.cli.commands.httpx.get") as mock_get:
            mock_get.side_effect = OSError("refused")
            result = _probe_ollama("http://myserver:11434")
        assert "http://myserver:11434" in result.detail

    def test_never_raises_on_unexpected_exception(self) -> None:
        with patch("src.cli.commands.httpx.get") as mock_get:
            mock_get.side_effect = RuntimeError("unexpected")
            result = _probe_ollama("http://localhost:11434")
        assert result.ok is False


# ---------------------------------------------------------------------------
# _probe_api_key
# ---------------------------------------------------------------------------


class TestProbeApiKey:
    def test_returns_ok_true_when_key_present(self) -> None:
        result = _probe_api_key("anthropic", "sk-ant-abc123")
        assert result.ok is True

    def test_returns_ok_false_when_key_empty(self) -> None:
        result = _probe_api_key("anthropic", "")
        assert result.ok is False

    def test_detail_includes_provider_name_on_success(self) -> None:
        result = _probe_api_key("huggingface", "hf_token")
        assert "huggingface" in result.detail

    def test_detail_includes_provider_name_on_failure(self) -> None:
        result = _probe_api_key("openai", "")
        assert "openai" in result.detail

    @pytest.mark.parametrize("provider", ["anthropic", "huggingface", "openai"])
    def test_key_present_is_ok_for_all_key_based_providers(self, provider: str) -> None:
        result = _probe_api_key(provider, "some-key")
        assert result.ok is True

    @pytest.mark.parametrize("provider", ["anthropic", "huggingface", "openai"])
    def test_key_missing_is_not_ok_for_all_key_based_providers(
        self, provider: str
    ) -> None:
        result = _probe_api_key(provider, "")
        assert result.ok is False


# ---------------------------------------------------------------------------
# _probe_llm (dispatch)
# ---------------------------------------------------------------------------


class TestProbeLlm:
    def test_ollama_provider_calls_probe_ollama(self) -> None:
        with patch("src.cli.commands.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            result = _probe_llm(_settings(llm_provider="ollama"))
        mock_get.assert_called_once()
        assert result.ok is True

    def test_anthropic_provider_does_not_call_http(self) -> None:
        with patch("src.cli.commands.httpx.get") as mock_get:
            result = _probe_llm(
                _settings(llm_provider="anthropic", llm_api_key="sk-ant-x")
            )
        mock_get.assert_not_called()
        assert result.ok is True

    def test_key_based_provider_returns_ok_false_when_key_empty(self) -> None:
        result = _probe_llm(_settings(llm_provider="anthropic", llm_api_key=""))
        assert result.ok is False


# ---------------------------------------------------------------------------
# check_health (integration of probes)
# ---------------------------------------------------------------------------


class TestCheckHealth:
    def test_returns_health_status_instance(self) -> None:
        with (
            patch("src.cli.commands.QdrantClient") as mock_qdrant,
            patch("src.cli.commands.httpx.get") as mock_get,
        ):
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            mock_get.return_value = MagicMock(status_code=200)
            result = check_health(_settings())
        assert isinstance(result, HealthStatus)

    def test_all_ok_true_when_all_services_healthy(self) -> None:
        with (
            patch("src.cli.commands.QdrantClient") as mock_qdrant,
            patch("src.cli.commands.httpx.get") as mock_get,
        ):
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            mock_get.return_value = MagicMock(status_code=200)
            result = check_health(_settings())
        assert result.all_ok is True

    def test_all_ok_false_when_qdrant_down(self) -> None:
        with (
            patch("src.cli.commands.QdrantClient") as mock_qdrant,
            patch("src.cli.commands.httpx.get") as mock_get,
        ):
            mock_qdrant.return_value.get_collections.side_effect = ConnectionError()
            mock_get.return_value = MagicMock(status_code=200)
            result = check_health(_settings())
        assert result.all_ok is False

    def test_all_ok_false_when_ollama_unreachable(self) -> None:
        with (
            patch("src.cli.commands.QdrantClient") as mock_qdrant,
            patch("src.cli.commands.httpx.get") as mock_get,
        ):
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            mock_get.side_effect = OSError("refused")
            result = check_health(_settings())
        assert result.all_ok is False

    def test_anthropic_ok_when_qdrant_ok_and_key_set(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_qdrant:
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = check_health(
                _settings(llm_provider="anthropic", llm_api_key="sk-ant-x")
            )
        assert result.all_ok is True

    def test_anthropic_not_ok_when_key_missing(self) -> None:
        with patch("src.cli.commands.QdrantClient") as mock_qdrant:
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = check_health(_settings(llm_provider="anthropic", llm_api_key=""))
        assert result.all_ok is False

    def test_qdrant_health_carries_correct_addr(self) -> None:
        with (
            patch("src.cli.commands.QdrantClient") as mock_qdrant,
            patch("src.cli.commands.httpx.get") as mock_get,
        ):
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            mock_get.return_value = MagicMock(status_code=200)
            result = check_health(
                _settings(qdrant_host="qdrant.internal", qdrant_port=6335)
            )
        assert "qdrant.internal:6335" in result.qdrant.detail
