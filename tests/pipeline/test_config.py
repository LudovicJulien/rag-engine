# tests/pipeline/test_config.py
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError

import src.pipeline.config as config_module
from src.pipeline.config import Settings


class TestSettingsTypes:
    """Tests that Settings fields expose expected Python types."""

    def test_qdrant_host_is_string(self) -> None:
        assert isinstance(Settings().qdrant_host, str)

    def test_qdrant_port_is_int(self) -> None:
        assert isinstance(Settings().qdrant_port, int)

    def test_collection_name_is_string(self) -> None:
        assert isinstance(Settings().collection_name, str)

    def test_embedding_model_is_string(self) -> None:
        assert isinstance(Settings().embedding_model, str)

    def test_embedding_batch_size_is_int(self) -> None:
        assert isinstance(Settings().embedding_batch_size, int)

    def test_llm_provider_is_string(self) -> None:
        assert isinstance(Settings().llm_provider, str)

    def test_llm_base_url_is_string(self) -> None:
        assert isinstance(Settings().llm_base_url, str)

    def test_llm_model_is_string(self) -> None:
        assert isinstance(Settings().llm_model, str)

    def test_llm_api_key_is_string(self) -> None:
        assert isinstance(Settings().llm_api_key, str)

    def test_top_k_is_int(self) -> None:
        assert isinstance(Settings().top_k, int)

    def test_score_threshold_is_float(self) -> None:
        assert isinstance(Settings().score_threshold, float)

    def test_log_level_is_string(self) -> None:
        assert isinstance(Settings().log_level, str)


class TestSettingsInvariants:
    """Tests that Settings always respects domain invariants."""

    def test_qdrant_port_is_valid(self) -> None:
        assert 1 <= Settings().qdrant_port <= 65535

    def test_embedding_batch_size_is_positive(self) -> None:
        assert Settings().embedding_batch_size >= 1

    def test_top_k_is_positive(self) -> None:
        assert Settings().top_k >= 1

    def test_score_threshold_is_between_zero_and_one(self) -> None:
        assert 0.0 <= Settings().score_threshold <= 1.0

    def test_llm_provider_is_allowed_value(self) -> None:
        assert Settings().llm_provider in {
            "ollama",
            "huggingface",
            "openai",
            "anthropic",
            "gemini",
        }

    def test_qdrant_host_is_not_empty(self) -> None:
        assert Settings().qdrant_host.strip() != ""

    def test_collection_name_is_not_empty(self) -> None:
        assert Settings().collection_name.strip() != ""

    def test_embedding_model_is_not_empty(self) -> None:
        assert Settings().embedding_model.strip() != ""

    def test_llm_provider_is_not_empty(self) -> None:
        assert Settings().llm_provider.strip() != ""

    def test_llm_model_is_not_empty(self) -> None:
        assert Settings().llm_model.strip() != ""

    def test_log_level_is_not_empty(self) -> None:
        assert Settings().log_level.strip() != ""

    def test_log_level_is_allowed_value(self) -> None:
        assert Settings().log_level in {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }


class TestSettingsFromEnv:
    """Tests environment variable overrides."""

    def test_qdrant_host_from_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("QDRANT_HOST", "custom-host")
        assert Settings().qdrant_host == "custom-host"

    def test_qdrant_port_from_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("QDRANT_PORT", "6334")
        assert Settings().qdrant_port == 6334

    def test_top_k_from_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TOP_K", "10")
        assert Settings().top_k == 10

    def test_score_threshold_from_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SCORE_THRESHOLD", "0.85")
        assert Settings().score_threshold == pytest.approx(0.85)

    def test_llm_provider_from_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        assert Settings().llm_provider == "anthropic"

    def test_llm_base_url_from_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
        assert Settings().llm_base_url == "https://api.openai.com/v1"

    def test_llm_model_from_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-20250514")
        assert Settings().llm_model == "claude-sonnet-4-20250514"

    def test_llm_api_key_from_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_API_KEY", "sk-test-key")
        assert Settings().llm_api_key == "sk-test-key"

    def test_log_level_from_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        assert Settings().log_level == "DEBUG"

    def test_env_overrides_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original = Settings().top_k
        monkeypatch.setenv("TOP_K", str(original + 1))
        assert Settings().top_k == original + 1


class TestSettingsValidation:
    """Tests Settings validation rules."""

    def test_invalid_qdrant_port_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("QDRANT_PORT", "0")
        with pytest.raises(ValidationError):
            Settings()

    def test_invalid_qdrant_port_too_high(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("QDRANT_PORT", "99999")
        with pytest.raises(ValidationError):
            Settings()

    def test_invalid_qdrant_port_negative(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("QDRANT_PORT", "-1")
        with pytest.raises(ValidationError):
            Settings()

    def test_invalid_llm_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "unknown_provider")
        with pytest.raises(ValidationError):
            Settings()

    def test_invalid_log_level(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
        with pytest.raises(ValidationError):
            Settings()

    def test_invalid_top_k_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TOP_K", "0")
        with pytest.raises(ValidationError):
            Settings()

    def test_invalid_top_k_negative(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TOP_K", "-1")
        with pytest.raises(ValidationError):
            Settings()

    def test_invalid_score_threshold_above_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SCORE_THRESHOLD", "1.5")
        with pytest.raises(ValidationError):
            Settings()

    def test_invalid_score_threshold_negative(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SCORE_THRESHOLD", "-0.1")
        with pytest.raises(ValidationError):
            Settings()

    def test_invalid_embedding_batch_size_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "0")
        with pytest.raises(ValidationError):
            Settings()

    def test_empty_llm_provider_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "   ")
        with pytest.raises(ValidationError):
            Settings()

    def test_empty_embedding_model_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EMBEDDING_MODEL", "")
        with pytest.raises(ValidationError):
            Settings()


class TestSettingsNormalization:
    """Tests normalization behavior."""

    def test_log_level_lowercase_normalized_to_uppercase(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LOG_LEVEL", "debug")
        assert Settings().log_level == "DEBUG"

    def test_log_level_mixed_case_normalized(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LOG_LEVEL", "Warning")
        assert Settings().log_level == "WARNING"

    def test_llm_provider_uppercase_normalized_to_lowercase(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "OLLAMA")
        assert Settings().llm_provider == "ollama"

    def test_llm_provider_mixed_case_normalized(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "Anthropic")
        assert Settings().llm_provider == "anthropic"


class TestSettingsSingleton:
    """Tests module-level singleton behavior."""

    def test_singleton_is_settings_instance(self) -> None:
        assert isinstance(config_module.settings, Settings)

    def test_singleton_respects_invariants(self) -> None:
        s = config_module.settings
        assert 1 <= s.qdrant_port <= 65535
        assert s.top_k >= 1
        assert 0.0 <= s.score_threshold <= 1.0
        assert s.llm_provider in {
            "ollama",
            "huggingface",
            "openai",
            "anthropic",
            "gemini",
        }
        assert s.log_level in {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

    def test_singleton_reloads_with_new_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TOP_K", "15")
        importlib.reload(config_module)
        assert config_module.settings.top_k == 15

    def test_singleton_import_works(self) -> None:
        from src.pipeline.config import settings

        assert settings is not None
        assert hasattr(settings, "qdrant_host")


class TestBM25CachePathType:
    """Tests that bm25_cache_path exposes the expected Python type."""

    def test_bm25_cache_path_is_path(self) -> None:
        assert isinstance(Settings().bm25_cache_path, Path)


class TestBM25CachePathInvariants:
    """Tests structural invariants of the bm25_cache_path field."""

    def test_default_filename_is_bm25_pkl(self) -> None:
        assert Settings().bm25_cache_path.name == "bm25.pkl"

    def test_default_parent_directory(self) -> None:
        assert Settings().bm25_cache_path.parent.name == "rag"

    def test_default_tilde_is_expanded(self) -> None:
        path = Settings().bm25_cache_path
        assert "~" not in str(path)

    def test_default_is_absolute(self) -> None:
        assert Settings().bm25_cache_path.is_absolute()


class TestBM25CachePathFromEnv:
    """Tests environment variable override for bm25_cache_path."""

    def test_bm25_cache_path_from_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("BM25_CACHE_PATH", "/tmp/custom/bm25.pkl")
        assert Settings().bm25_cache_path == Path("/tmp/custom/bm25.pkl")

    def test_bm25_cache_path_tilde_expanded_from_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("BM25_CACHE_PATH", "~/.cache/custom/bm25.pkl")
        result = Settings().bm25_cache_path
        assert "~" not in str(result)
        assert result.is_absolute()
