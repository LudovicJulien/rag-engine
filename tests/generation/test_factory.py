# tests/generation/test_factory.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.generation.anthropic_generator import AnthropicGenerator
from src.generation.factory import get_generator
from src.generation.huggingface_generator import HuggingFaceGenerator
from src.generation.ollama_generator import OllamaGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(provider: str, model: str = "some-model") -> MagicMock:
    settings = MagicMock()
    settings.llm_provider = provider
    settings.llm_model = model
    settings.llm_api_key = ""
    settings.llm_base_url = "http://localhost:11434"
    return settings


# ---------------------------------------------------------------------------
# Implemented providers — correct type returned
# ---------------------------------------------------------------------------


class TestGetGeneratorImplementedProviders:
    def test_ollama_returns_ollama_generator(self) -> None:
        with patch("src.generation.ollama_generator.ollama.Client"):
            gen = get_generator(_make_settings("ollama", "gemma3:4b"))
        assert isinstance(gen, OllamaGenerator)

    def test_huggingface_returns_huggingface_generator(self) -> None:
        with patch("src.generation.huggingface_generator.InferenceClient"):
            gen = get_generator(
                _make_settings("huggingface", "mistralai/Mistral-7B-Instruct-v0.3")
            )
        assert isinstance(gen, HuggingFaceGenerator)

    def test_anthropic_returns_anthropic_generator(self) -> None:
        with patch("src.generation.anthropic_generator.anthropic.Anthropic"):
            gen = get_generator(_make_settings("anthropic", "claude-haiku-4-5"))
        assert isinstance(gen, AnthropicGenerator)

    def test_ollama_model_forwarded_from_settings(self) -> None:
        with patch("src.generation.ollama_generator.ollama.Client"):
            gen = get_generator(_make_settings("ollama", "llama3:8b"))
        assert gen.model_name == "llama3:8b"

    def test_huggingface_model_forwarded_from_settings(self) -> None:
        with patch("src.generation.huggingface_generator.InferenceClient"):
            gen = get_generator(_make_settings("huggingface", "org/my-model"))
        assert gen.model_name == "org/my-model"

    def test_anthropic_model_forwarded_from_settings(self) -> None:
        with patch("src.generation.anthropic_generator.anthropic.Anthropic"):
            gen = get_generator(_make_settings("anthropic", "claude-sonnet-4-6"))
        assert gen.model_name == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Not-yet-implemented providers — NotImplementedError
# ---------------------------------------------------------------------------


class TestGetGeneratorNotImplementedProviders:
    def test_openai_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="openai"):
            get_generator(_make_settings("openai"))

    def test_gemini_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="gemini"):
            get_generator(_make_settings("gemini"))

    def test_not_implemented_error_lists_available_providers(self) -> None:
        with pytest.raises(NotImplementedError, match="ollama"):
            get_generator(_make_settings("openai"))


# ---------------------------------------------------------------------------
# Unknown provider — ValueError
# ---------------------------------------------------------------------------


class TestGetGeneratorUnknownProvider:
    def test_unknown_provider_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_generator(_make_settings("cohere"))

    def test_value_error_includes_provider_name(self) -> None:
        with pytest.raises(ValueError, match="cohere"):
            get_generator(_make_settings("cohere"))
