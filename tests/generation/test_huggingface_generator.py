# tests/generation/test_huggingface_generator.py
from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from src.generation.huggingface_generator import (
    HuggingFaceGenerator,
    HuggingFaceGeneratorConfig,
)
from src.shared.models import Chunk, ChunkMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str = "c1", text: str = "Paris is the capital of France."
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        parent_doc_id="doc1",
        text=text,
        chunk_index=0,
        total_chunks=1,
        metadata=ChunkMetadata(),
    )


def _make_api_response(
    content: str, prompt_tokens: int = 10, completion_tokens: int = 5
) -> MagicMock:
    """Build a minimal chat_completion response mirroring the HF SDK structure."""
    response = MagicMock()
    response.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
    response.usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return response


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestHuggingFaceGeneratorConfig:
    def test_defaults_are_valid(self) -> None:
        config = HuggingFaceGeneratorConfig()
        assert config.model == "mistralai/Mistral-7B-Instruct-v0.3"
        assert config.temperature == pytest.approx(0.1)
        assert config.max_new_tokens == 512
        assert config.api_token == ""

    def test_empty_model_raises(self) -> None:
        with pytest.raises(ValueError, match="model cannot be empty"):
            HuggingFaceGeneratorConfig(model="   ")

    def test_temperature_below_range_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be in"):
            HuggingFaceGeneratorConfig(temperature=-0.1)

    def test_temperature_above_range_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be in"):
            HuggingFaceGeneratorConfig(temperature=2.1)

    def test_max_new_tokens_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_new_tokens must be >= 1"):
            HuggingFaceGeneratorConfig(max_new_tokens=0)

    def test_boundary_temperature_zero_is_valid(self) -> None:
        config = HuggingFaceGeneratorConfig(temperature=0.0)
        assert config.temperature == pytest.approx(0.0)

    def test_boundary_temperature_two_is_valid(self) -> None:
        config = HuggingFaceGeneratorConfig(temperature=2.0)
        assert config.temperature == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Provider identity
# ---------------------------------------------------------------------------


class TestHuggingFaceGeneratorProperties:
    def test_provider_name(self) -> None:
        with patch("src.generation.huggingface_generator.InferenceClient"):
            gen = HuggingFaceGenerator(HuggingFaceGeneratorConfig(model="org/model"))
        assert gen.provider_name == "huggingface"

    def test_model_name_matches_config(self) -> None:
        with patch("src.generation.huggingface_generator.InferenceClient"):
            gen = HuggingFaceGenerator(HuggingFaceGeneratorConfig(model="org/model"))
        assert gen.model_name == "org/model"


# ---------------------------------------------------------------------------
# generate — happy path
# ---------------------------------------------------------------------------


class TestHuggingFaceGeneratorGenerate:
    def _make_generator(
        self, model: str = "org/model"
    ) -> tuple[HuggingFaceGenerator, MagicMock]:
        with patch("src.generation.huggingface_generator.InferenceClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            gen = HuggingFaceGenerator(HuggingFaceGeneratorConfig(model=model))
        gen._client = mock_client
        return gen, mock_client

    def test_generate_returns_result_with_answer(self) -> None:
        gen, mock_client = self._make_generator()
        mock_client.chat_completion.return_value = _make_api_response(
            "Paris is the capital."
        )

        result = gen.generate(query="What is Paris?", context=[_make_chunk()])

        assert result.answer == "Paris is the capital."

    def test_generate_sources_equal_chunk_ids(self) -> None:
        gen, mock_client = self._make_generator()
        chunks = [_make_chunk("c1"), _make_chunk("c2", "Another fact.")]
        mock_client.chat_completion.return_value = _make_api_response("Answer.")

        result = gen.generate(query="Q?", context=chunks)

        assert result.sources == ["c1", "c2"]

    def test_generate_token_count_summed(self) -> None:
        gen, mock_client = self._make_generator()
        mock_client.chat_completion.return_value = _make_api_response(
            "Answer.", prompt_tokens=20, completion_tokens=8
        )

        result = gen.generate(query="Q?", context=[_make_chunk()])

        assert result.tokens_used == 28

    def test_generate_tokens_none_when_usage_missing(self) -> None:
        gen, mock_client = self._make_generator()
        response = _make_api_response("Answer.")
        response.usage = None
        mock_client.chat_completion.return_value = response

        result = gen.generate(query="Q?", context=[_make_chunk()])

        assert result.tokens_used is None

    def test_generate_strips_whitespace_from_answer(self) -> None:
        gen, mock_client = self._make_generator()
        mock_client.chat_completion.return_value = _make_api_response("  Answer.  \n")

        result = gen.generate(query="Q?", context=[_make_chunk()])

        assert result.answer == "Answer."

    def test_generate_model_stored_in_result(self) -> None:
        gen, mock_client = self._make_generator(model="org/custom-model")
        mock_client.chat_completion.return_value = _make_api_response("Answer.")

        result = gen.generate(query="Q?", context=[_make_chunk()])

        assert result.model == "org/custom-model"

    def test_generate_calls_chat_completion_with_system_and_user_messages(self) -> None:
        gen, mock_client = self._make_generator()
        mock_client.chat_completion.return_value = _make_api_response("Answer.")

        gen.generate(query="What is Paris?", context=[_make_chunk("c1", "Paris text.")])

        call_kwargs = mock_client.chat_completion.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "What is Paris?" in messages[1]["content"]
        assert "Paris text." in messages[1]["content"]

    def test_generate_passes_temperature_and_max_tokens(self) -> None:
        config = HuggingFaceGeneratorConfig(
            model="org/model", temperature=0.5, max_new_tokens=256
        )
        with patch("src.generation.huggingface_generator.InferenceClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            gen = HuggingFaceGenerator(config)
        gen._client = mock_client
        mock_client.chat_completion.return_value = _make_api_response("Answer.")

        gen.generate(query="Q?", context=[_make_chunk()])

        call_kwargs = mock_client.chat_completion.call_args.kwargs
        assert call_kwargs["temperature"] == pytest.approx(0.5)
        assert call_kwargs["max_tokens"] == 256


# ---------------------------------------------------------------------------
# generate — input validation
# ---------------------------------------------------------------------------


class TestHuggingFaceGeneratorInputValidation:
    def _make_generator(self) -> HuggingFaceGenerator:
        with patch("src.generation.huggingface_generator.InferenceClient"):
            gen = HuggingFaceGenerator()
        gen._client = MagicMock()
        return gen

    def test_empty_query_raises(self) -> None:
        gen = self._make_generator()
        with pytest.raises(ValueError, match="query"):
            gen.generate(query="", context=[_make_chunk()])

    def test_whitespace_only_query_raises(self) -> None:
        gen = self._make_generator()
        with pytest.raises(ValueError, match="query"):
            gen.generate(query="   ", context=[_make_chunk()])

    def test_empty_context_raises(self) -> None:
        gen = self._make_generator()
        with pytest.raises(ValueError, match="context"):
            gen.generate(query="Q?", context=[])

    def test_empty_api_response_raises(self) -> None:
        gen = self._make_generator()
        mock_client = cast(MagicMock, gen._client)
        mock_client.chat_completion.return_value = _make_api_response("   ")
        with pytest.raises(RuntimeError, match="empty response"):
            gen.generate(query="Q?", context=[_make_chunk()])


# ---------------------------------------------------------------------------
# generate — error handling
# ---------------------------------------------------------------------------


class TestHuggingFaceGeneratorErrorHandling:
    def _make_generator(self) -> HuggingFaceGenerator:
        with patch("src.generation.huggingface_generator.InferenceClient"):
            gen = HuggingFaceGenerator()
        gen._client = MagicMock()
        return gen

    def test_hf_http_error_wrapped_as_runtime_error(self) -> None:
        from huggingface_hub.errors import HfHubHTTPError

        gen = self._make_generator()
        mock_client = cast(MagicMock, gen._client)
        mock_client.chat_completion.side_effect = HfHubHTTPError(
            "401 Unauthorized", response=MagicMock()
        )
        with pytest.raises(RuntimeError, match="HuggingFace API error"):
            gen.generate(query="Q?", context=[_make_chunk()])

    def test_unexpected_exception_wrapped_as_runtime_error(self) -> None:
        gen = self._make_generator()
        mock_client = cast(MagicMock, gen._client)
        mock_client.chat_completion.side_effect = ConnectionError("network down")
        with pytest.raises(RuntimeError, match="Unexpected error"):
            gen.generate(query="Q?", context=[_make_chunk()])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestHuggingFaceGeneratorFromSettings:
    def test_from_settings_uses_llm_model_and_api_key(self) -> None:
        settings = MagicMock()
        settings.llm_model = "org/my-model"
        settings.llm_api_key = "hf_secret"

        with patch("src.generation.huggingface_generator.InferenceClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            gen = HuggingFaceGenerator.from_settings(settings)

        assert gen.model_name == "org/my-model"
        assert gen._config.api_token == "hf_secret"

    def test_from_settings_empty_api_key_passes_none_to_client(self) -> None:
        settings = MagicMock()
        settings.llm_model = "org/my-model"
        settings.llm_api_key = ""

        with patch("src.generation.huggingface_generator.InferenceClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            HuggingFaceGenerator.from_settings(settings)

        _, init_kwargs = mock_cls.call_args
        assert init_kwargs.get("token") is None
