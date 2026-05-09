# tests/generation/test_ollama_generator.py
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import ollama
import pytest

from src.generation.ollama_generator import OllamaGenerator, OllamaGeneratorConfig
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
    content: str,
    prompt_eval_count: int | None = 10,
    eval_count: int | None = 5,
) -> MagicMock:
    """Build a minimal chat response mirroring the Ollama SDK structure."""
    response = MagicMock()
    response.message = SimpleNamespace(content=content)
    response.prompt_eval_count = prompt_eval_count
    response.eval_count = eval_count
    return response


def _make_generator(
    model: str = "gemma3:4b",
    base_url: str = "http://localhost:11434",
) -> tuple[OllamaGenerator, MagicMock]:
    with patch("src.generation.ollama_generator.ollama.Client") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        gen = OllamaGenerator(OllamaGeneratorConfig(model=model, base_url=base_url))
    gen._client = mock_client
    return gen, mock_client


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestOllamaGeneratorConfig:
    def test_defaults_are_valid(self) -> None:
        config = OllamaGeneratorConfig()
        assert config.model == "gemma3:4b"
        assert config.base_url == "http://localhost:11434"
        assert config.temperature == pytest.approx(0.1)
        assert config.num_ctx == 4096

    def test_empty_model_raises(self) -> None:
        with pytest.raises(ValueError, match="model cannot be empty"):
            OllamaGeneratorConfig(model="   ")

    def test_empty_base_url_raises(self) -> None:
        with pytest.raises(ValueError, match="base_url cannot be empty"):
            OllamaGeneratorConfig(base_url="   ")

    def test_temperature_below_range_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be in"):
            OllamaGeneratorConfig(temperature=-0.1)

    def test_temperature_above_range_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be in"):
            OllamaGeneratorConfig(temperature=2.1)

    def test_num_ctx_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_ctx must be >= 1"):
            OllamaGeneratorConfig(num_ctx=0)

    def test_boundary_temperature_zero_is_valid(self) -> None:
        config = OllamaGeneratorConfig(temperature=0.0)
        assert config.temperature == pytest.approx(0.0)

    def test_boundary_temperature_two_is_valid(self) -> None:
        config = OllamaGeneratorConfig(temperature=2.0)
        assert config.temperature == pytest.approx(2.0)

    def test_num_ctx_one_is_valid(self) -> None:
        config = OllamaGeneratorConfig(num_ctx=1)
        assert config.num_ctx == 1


# ---------------------------------------------------------------------------
# Provider identity
# ---------------------------------------------------------------------------


class TestOllamaGeneratorProperties:
    def test_provider_name(self) -> None:
        gen, _ = _make_generator()
        assert gen.provider_name == "ollama"

    def test_model_name_matches_config(self) -> None:
        gen, _ = _make_generator(model="llama3:8b")
        assert gen.model_name == "llama3:8b"


# ---------------------------------------------------------------------------
# generate — happy path
# ---------------------------------------------------------------------------


class TestOllamaGeneratorGenerate:
    def test_generate_returns_result_with_answer(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.chat.return_value = _make_api_response("Paris is the capital.")

        result = gen.generate(query="What is Paris?", context=[_make_chunk()])

        assert result.answer == "Paris is the capital."

    def test_generate_sources_equal_chunk_ids(self) -> None:
        gen, mock_client = _make_generator()
        chunks = [_make_chunk("c1"), _make_chunk("c2", "Another fact.")]
        mock_client.chat.return_value = _make_api_response("Answer.")

        result = gen.generate(query="Q?", context=chunks)

        assert result.sources == ["c1", "c2"]

    def test_generate_token_count_summed(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.chat.return_value = _make_api_response(
            "Answer.", prompt_eval_count=20, eval_count=8
        )

        result = gen.generate(query="Q?", context=[_make_chunk()])

        assert result.tokens_used == 28

    def test_generate_tokens_none_when_both_counts_none(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.chat.return_value = _make_api_response(
            "Answer.", prompt_eval_count=None, eval_count=None
        )

        result = gen.generate(query="Q?", context=[_make_chunk()])

        assert result.tokens_used is None

    def test_generate_tokens_none_when_prompt_eval_count_none(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.chat.return_value = _make_api_response(
            "Answer.", prompt_eval_count=None, eval_count=5
        )

        result = gen.generate(query="Q?", context=[_make_chunk()])

        assert result.tokens_used is None

    def test_generate_tokens_none_when_eval_count_none(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.chat.return_value = _make_api_response(
            "Answer.", prompt_eval_count=10, eval_count=None
        )

        result = gen.generate(query="Q?", context=[_make_chunk()])

        assert result.tokens_used is None

    def test_generate_strips_whitespace_from_answer(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.chat.return_value = _make_api_response("  Answer.  \n")

        result = gen.generate(query="Q?", context=[_make_chunk()])

        assert result.answer == "Answer."

    def test_generate_model_stored_in_result(self) -> None:
        gen, mock_client = _make_generator(model="llama3:8b")
        mock_client.chat.return_value = _make_api_response("Answer.")

        result = gen.generate(query="Q?", context=[_make_chunk()])

        assert result.model == "llama3:8b"

    def test_generate_calls_chat_with_system_and_user_messages(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.chat.return_value = _make_api_response("Answer.")

        gen.generate(query="What is Paris?", context=[_make_chunk("c1", "Paris text.")])

        call_kwargs = mock_client.chat.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "What is Paris?" in messages[1]["content"]
        assert "Paris text." in messages[1]["content"]

    def test_generate_passes_temperature_and_num_ctx(self) -> None:
        config = OllamaGeneratorConfig(model="gemma3:4b", temperature=0.7, num_ctx=2048)
        with patch("src.generation.ollama_generator.ollama.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            gen = OllamaGenerator(config)
        gen._client = mock_client
        mock_client.chat.return_value = _make_api_response("Answer.")

        gen.generate(query="Q?", context=[_make_chunk()])

        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs["options"]["temperature"] == pytest.approx(0.7)
        assert call_kwargs["options"]["num_ctx"] == 2048

    def test_generate_passes_model_to_chat(self) -> None:
        gen, mock_client = _make_generator(model="mistral:7b")
        mock_client.chat.return_value = _make_api_response("Answer.")

        gen.generate(query="Q?", context=[_make_chunk()])

        call_kwargs = mock_client.chat.call_args.kwargs
        assert call_kwargs["model"] == "mistral:7b"


# ---------------------------------------------------------------------------
# generate — input validation
# ---------------------------------------------------------------------------


class TestOllamaGeneratorInputValidation:
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
        mock_client.chat.return_value = _make_api_response("   ")
        with pytest.raises(RuntimeError, match="empty response"):
            gen.generate(query="Q?", context=[_make_chunk()])


# ---------------------------------------------------------------------------
# generate — error handling
# ---------------------------------------------------------------------------


class TestOllamaGeneratorErrorHandling:
    def test_response_error_wrapped_as_runtime_error(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.chat.side_effect = ollama.ResponseError("model not found")
        with pytest.raises(RuntimeError, match="ResponseError"):
            gen.generate(query="Q?", context=[_make_chunk()])

    def test_request_error_wrapped_as_runtime_error(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.chat.side_effect = ollama.RequestError("connection refused")
        with pytest.raises(RuntimeError, match="unreachable"):
            gen.generate(query="Q?", context=[_make_chunk()])

    def test_unexpected_exception_wrapped_as_runtime_error(self) -> None:
        gen, mock_client = _make_generator()
        mock_client.chat.side_effect = ConnectionError("network down")
        with pytest.raises(RuntimeError, match="Unexpected error"):
            gen.generate(query="Q?", context=[_make_chunk()])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestOllamaGeneratorFromSettings:
    def test_from_settings_uses_llm_model_and_base_url(self) -> None:
        settings = MagicMock()
        settings.llm_model = "llama3:8b"
        settings.llm_base_url = "http://ollama-server:11434"
        settings.llm_domain = "general"
        settings.llm_max_context_tokens = 4096

        with patch("src.generation.ollama_generator.ollama.Client") as mock_cls:
            mock_cls.return_value = MagicMock()
            gen = OllamaGenerator.from_settings(settings)

        assert gen.model_name == "llama3:8b"
        assert gen._config.base_url == "http://ollama-server:11434"
