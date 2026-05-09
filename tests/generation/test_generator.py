# tests/generation/test_generator.py
from __future__ import annotations

import pytest

from src.generation.generator import GenerationResult, LLMGenerator
from src.shared.models import Chunk, ChunkMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(**overrides: object) -> GenerationResult:
    defaults: dict[str, object] = {
        "answer": "Paris is the capital of France.",
        "sources": ["c1"],
        "detected_language": "fr",
        "model": "gemma3:4b",
        "prompt_version": "v0",
    }
    defaults.update(overrides)
    return GenerationResult(**defaults)  # type: ignore[arg-type]


def _make_chunk(chunk_id: str = "c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        parent_doc_id="doc1",
        text="Some text.",
        chunk_index=0,
        total_chunks=1,
        metadata=ChunkMetadata(),
    )


class _StubGenerator(LLMGenerator):
    """Minimal concrete subclass used to exercise the ABC shared logic."""

    def generate(self, query: str, context: list[Chunk]) -> GenerationResult:
        self._validate_inputs(query, context)
        return _make_result()

    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub-model"


# ---------------------------------------------------------------------------
# GenerationResult — valid construction
# ---------------------------------------------------------------------------


class TestGenerationResult:
    def test_valid_french(self) -> None:
        result = _make_result(detected_language="fr")
        assert result.detected_language == "fr"

    def test_valid_english(self) -> None:
        result = _make_result(detected_language="en")
        assert result.detected_language == "en"

    def test_tokens_used_defaults_to_none(self) -> None:
        result = _make_result()
        assert result.tokens_used is None

    def test_tokens_used_stored_correctly(self) -> None:
        result = _make_result(tokens_used=42)
        assert result.tokens_used == 42

    def test_sources_can_be_empty_list(self) -> None:
        result = _make_result(sources=[])
        assert result.sources == []

    def test_sources_preserves_order(self) -> None:
        result = _make_result(sources=["c3", "c1", "c2"])
        assert result.sources == ["c3", "c1", "c2"]

    def test_answer_stored_correctly(self) -> None:
        result = _make_result(answer="The answer is 42.")
        assert result.answer == "The answer is 42."

    def test_model_stored_correctly(self) -> None:
        result = _make_result(model="claude-haiku-4-5")
        assert result.model == "claude-haiku-4-5"

    def test_prompt_version_stored_correctly(self) -> None:
        result = _make_result(prompt_version="v1")
        assert result.prompt_version == "v1"

    def test_is_frozen(self) -> None:
        result = _make_result()
        with pytest.raises((AttributeError, TypeError)):
            result.answer = "mutated"  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Validation — raises
    # ------------------------------------------------------------------

    def test_empty_answer_raises(self) -> None:
        with pytest.raises(ValueError, match="answer cannot be empty"):
            _make_result(answer="")

    def test_unsupported_language_raises(self) -> None:
        with pytest.raises(ValueError, match="detected_language"):
            _make_result(detected_language="de")

    def test_unknown_language_code_raises(self) -> None:
        with pytest.raises(ValueError, match="detected_language"):
            _make_result(detected_language="")

    def test_empty_model_raises(self) -> None:
        with pytest.raises(ValueError, match="model cannot be empty"):
            _make_result(model="")

    def test_empty_prompt_version_raises(self) -> None:
        with pytest.raises(ValueError, match="prompt_version cannot be empty"):
            _make_result(prompt_version="")


# ---------------------------------------------------------------------------
# LLMGenerator._validate_inputs
# ---------------------------------------------------------------------------


class TestLLMGeneratorValidateInputs:
    def setup_method(self) -> None:
        self.gen = _StubGenerator()

    def test_valid_inputs_do_not_raise(self) -> None:
        self.gen.generate(query="What is Paris?", context=[_make_chunk()])

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValueError, match="query"):
            self.gen.generate(query="", context=[_make_chunk()])

    def test_whitespace_only_query_raises(self) -> None:
        with pytest.raises(ValueError, match="query"):
            self.gen.generate(query="   ", context=[_make_chunk()])

    def test_empty_context_raises(self) -> None:
        with pytest.raises(ValueError, match="context"):
            self.gen.generate(query="Q?", context=[])

    def test_multiple_chunks_accepted(self) -> None:
        chunks = [_make_chunk("c1"), _make_chunk("c2"), _make_chunk("c3")]
        result = self.gen.generate(query="Q?", context=chunks)
        assert result is not None


# ---------------------------------------------------------------------------
# LLMGenerator abstract interface
# ---------------------------------------------------------------------------


class TestLLMGeneratorInterface:
    def test_cannot_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            LLMGenerator()  # type: ignore[abstract]

    def test_stub_provider_name(self) -> None:
        assert _StubGenerator().provider_name == "stub"

    def test_stub_model_name(self) -> None:
        assert _StubGenerator().model_name == "stub-model"
