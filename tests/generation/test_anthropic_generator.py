# tests/generation/test_anthropic_generator.py
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from src.generation.anthropic_generator import (
    AnthropicGenerator,
    AnthropicGeneratorConfig,
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
    content: str, input_tokens: int = 10, output_tokens: int = 5
) -> MagicMock:
    """Build a minimal Messages API response mirroring the Anthropic SDK structure."""
    response = MagicMock()
    response.content = [SimpleNamespace(type="text", text=content)]
    response.usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    response.stop_reason = "end_turn"
    return response


def _make_generator(
    model: str = "claude-haiku-4-5",
) -> tuple[AnthropicGenerator, MagicMock]:
    with patch("src.generation.anthropic_generator.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        gen = AnthropicGenerator(AnthropicGeneratorConfig(model=model))
    gen._client = mock_client
    return gen, mock_client


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestAnthropicGeneratorConfig:
    def test_defaults_are_valid(self) -> None:
        config = AnthropicGeneratorConfig()
        assert config.model == "claude-haiku-4-5"
        assert config.max_tokens == 1024
        assert config.temperature is None
        assert config.api_key == ""

    def test_empty_model_raises(self) -> None:
        with pytest.raises(ValueError, match="model cannot be empty"):
            AnthropicGeneratorConfig(model="   ")

    def test_max_tokens_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            AnthropicGeneratorConfig(max_tokens=0)

    def test_temperature_below_range_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be in"):
            AnthropicGeneratorConfig(temperature=-0.1)

    def test_temperature_above_range_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be in"):
            AnthropicGeneratorConfig(temperature=1.1)

    def test_temperature_none_is_valid(self) -> None:
        config = AnthropicGeneratorConfig(temperature=None)
        assert config.temperature is None

    def test_boundary_temperature_zero_is_valid(self) -> None:
        config = AnthropicGeneratorConfig(temperature=0.0)
        assert config.temperature == pytest.approx(0.0)

    def test_boundary_temperature_one_is_valid(self) -> None:
        config = AnthropicGeneratorConfig(temperature=1.0)
        assert config.temperature == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Provider identity
# ---------------------------------------------------------------------------


class TestAnthropicGeneratorProperties:
    def test_provider_name(self) -> None:
        gen, _ = _make_generator()
        assert gen.provider_name == "anthropic"

    def test_model_name_matches_config(self) -> None:
        gen, _ = _make_generator(model="claude-sonnet-4-6")
        assert gen.model_name == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# generate — happy path
# ---------------------------------------------------------------------------


class TestAnthropicGeneratorGenerate:
    def test_generate_returns_result_with_answer(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.messages.create.return_value = _make_api_response(
            "Paris is the capital."
        )

        result = gen.generate(query="What is Paris?", context=[_make_chunk()])

        assert result.answer == "Paris is the capital."

    def test_generate_sources_equal_chunk_ids(self) -> None:
        gen, mock_client = _make_generator()
        chunks = [_make_chunk("c1"), _make_chunk("c2", "Another fact.")]
        mock_client.messages.create.return_value = _make_api_response("Answer.")

        result = gen.generate(query="Q?", context=chunks)

        assert result.sources == ["c1", "c2"]

    def test_generate_token_count_summed(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.messages.create.return_value = _make_api_response(
            "Answer.", input_tokens=20, output_tokens=8
        )

        result = gen.generate(query="Q?", context=[_make_chunk()])

        assert result.tokens_used == 28

    def test_generate_strips_whitespace_from_answer(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.messages.create.return_value = _make_api_response("  Answer.  \n")

        result = gen.generate(query="Q?", context=[_make_chunk()])

        assert result.answer == "Answer."

    def test_generate_model_stored_in_result(self) -> None:
        gen, mock_client = _make_generator(model="claude-sonnet-4-6")
        mock_client.messages.create.return_value = _make_api_response("Answer.")

        result = gen.generate(query="Q?", context=[_make_chunk()])

        assert result.model == "claude-sonnet-4-6"

    def test_generate_calls_messages_create_with_system_and_user(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.messages.create.return_value = _make_api_response("Answer.")

        gen.generate(query="What is Paris?", context=[_make_chunk("c1", "Paris text.")])

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "system" in call_kwargs
        assert call_kwargs["messages"][0]["role"] == "user"
        assert "What is Paris?" in call_kwargs["messages"][0]["content"]
        assert "Paris text." in call_kwargs["messages"][0]["content"]

    def test_generate_omits_temperature_when_none(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.messages.create.return_value = _make_api_response("Answer.")

        gen.generate(query="Q?", context=[_make_chunk()])

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "temperature" not in call_kwargs

    def test_generate_passes_temperature_when_set(self) -> None:
        config = AnthropicGeneratorConfig(model="claude-haiku-4-5", temperature=0.5)
        with patch(
            "src.generation.anthropic_generator.anthropic.Anthropic"
        ) as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            gen = AnthropicGenerator(config)
        gen._client = mock_client
        mock_client.messages.create.return_value = _make_api_response("Answer.")

        gen.generate(query="Q?", context=[_make_chunk()])

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == pytest.approx(0.5)

    def test_generate_passes_max_tokens(self) -> None:
        config = AnthropicGeneratorConfig(model="claude-haiku-4-5", max_tokens=256)
        with patch(
            "src.generation.anthropic_generator.anthropic.Anthropic"
        ) as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            gen = AnthropicGenerator(config)
        gen._client = mock_client
        mock_client.messages.create.return_value = _make_api_response("Answer.")

        gen.generate(query="Q?", context=[_make_chunk()])

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 256


# ---------------------------------------------------------------------------
# generate — input validation
# ---------------------------------------------------------------------------


class TestAnthropicGeneratorInputValidation:
    def test_empty_query_raises(self) -> None:
        gen, _ = _make_generator()
        with pytest.raises(ValueError, match="query"):
            gen.generate(query="", context=[_make_chunk()])

    def test_whitespace_only_query_raises(self) -> None:
        gen, _ = _make_generator()
        with pytest.raises(ValueError, match="query"):
            gen.generate(query="   ", context=[_make_chunk()])

    def test_empty_context_raises(self) -> None:
        gen, _ = _make_generator()
        with pytest.raises(ValueError, match="context"):
            gen.generate(query="Q?", context=[])

    def test_empty_api_response_raises(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.messages.create.return_value = _make_api_response("   ")
        with pytest.raises(RuntimeError, match="empty response"):
            gen.generate(query="Q?", context=[_make_chunk()])


# ---------------------------------------------------------------------------
# generate — error handling
# ---------------------------------------------------------------------------


class TestAnthropicGeneratorErrorHandling:
    def test_authentication_error_wrapped_as_runtime_error(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.messages.create.side_effect = anthropic.AuthenticationError(
            message="401 Unauthorized",
            response=MagicMock(),
            body={},
        )
        with pytest.raises(RuntimeError, match="authentication failed"):
            gen.generate(query="Q?", context=[_make_chunk()])

    def test_rate_limit_error_wrapped_as_runtime_error(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.messages.create.side_effect = anthropic.RateLimitError(
            message="429 Too Many Requests",
            response=MagicMock(),
            body={},
        )
        with pytest.raises(RuntimeError, match="rate limit"):
            gen.generate(query="Q?", context=[_make_chunk()])

    def test_api_status_error_wrapped_as_runtime_error(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.messages.create.side_effect = anthropic.APIStatusError(
            message="500 Internal Server Error",
            response=MagicMock(),
            body={},
        )
        with pytest.raises(RuntimeError, match="API error"):
            gen.generate(query="Q?", context=[_make_chunk()])

    def test_unexpected_exception_wrapped_as_runtime_error(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.messages.create.side_effect = ConnectionError("network down")
        with pytest.raises(RuntimeError, match="Unexpected error"):
            gen.generate(query="Q?", context=[_make_chunk()])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestAnthropicGeneratorFromSettings:
    def test_from_settings_uses_llm_model_and_api_key(self) -> None:
        settings = MagicMock()
        settings.llm_model = "claude-sonnet-4-6"
        settings.llm_api_key = "sk-ant-secret"
        settings.llm_domain = "general"
        settings.llm_max_context_tokens = 4096

        with patch(
            "src.generation.anthropic_generator.anthropic.Anthropic"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            gen = AnthropicGenerator.from_settings(settings)

        assert gen.model_name == "claude-sonnet-4-6"
        assert gen._config.api_key == "sk-ant-secret"

    def test_from_settings_empty_api_key_passes_none_to_client(self) -> None:
        settings = MagicMock()
        settings.llm_model = "claude-haiku-4-5"
        settings.llm_api_key = ""
        settings.llm_domain = "general"
        settings.llm_max_context_tokens = 4096

        with patch(
            "src.generation.anthropic_generator.anthropic.Anthropic"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            AnthropicGenerator.from_settings(settings)

        _, init_kwargs = mock_cls.call_args
        assert init_kwargs.get("api_key") is None
