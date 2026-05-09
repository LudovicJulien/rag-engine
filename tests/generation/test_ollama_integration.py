# tests/generation/test_ollama_integration.py
from __future__ import annotations

import httpx
import ollama
import pytest

from src.generation.generator import GenerationResult
from src.generation.ollama_generator import OllamaGenerator, OllamaGeneratorConfig
from src.shared.models import Chunk

_MODEL = "gemma3:4b"
_BASE_URL = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Module-scoped fixtures — Ollama server + model checked once per session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generator() -> OllamaGenerator:
    """Return a live OllamaGenerator; skip the module if Ollama is unavailable."""
    client = ollama.Client(host=_BASE_URL)
    try:
        response = client.list()
        model_names = [m.model for m in response.models]
    except (ConnectionError, httpx.ConnectError, httpx.HTTPError):
        pytest.skip(f"Ollama server not reachable at {_BASE_URL}")
    if _MODEL not in model_names:
        pytest.skip(f"Model '{_MODEL}' not pulled — run: ollama pull {_MODEL}")
    return OllamaGenerator(OllamaGeneratorConfig(model=_MODEL, base_url=_BASE_URL))


@pytest.fixture(scope="module")
def fr_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="fr-1",
            parent_doc_id="doc-fr",
            text=(
                "Paris est la capitale de la France et abrite "
                "plus de 2 millions d'habitants."
            ),
            chunk_index=0,
            total_chunks=2,
        ),
        Chunk(
            chunk_id="fr-2",
            parent_doc_id="doc-fr",
            text=(
                "La tour Eiffel, construite en 1889, est le monument "
                "le plus visité de Paris."
            ),
            chunk_index=1,
            total_chunks=2,
        ),
    ]


@pytest.fixture(scope="module")
def en_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="en-1",
            parent_doc_id="doc-en",
            text=(
                "Paris is the capital of France and home "
                "to over 2 million inhabitants."
            ),
            chunk_index=0,
            total_chunks=2,
        ),
        Chunk(
            chunk_id="en-2",
            parent_doc_id="doc-en",
            text=(
                "The Eiffel Tower, built in 1889, is the most "
                "visited monument in Paris."
            ),
            chunk_index=1,
            total_chunks=2,
        ),
    ]


@pytest.fixture(scope="module")
def fr_result(generator: OllamaGenerator, fr_chunks: list[Chunk]) -> GenerationResult:
    """Single real LLM call (French) — reused across all FR assertions."""
    return generator.generate(
        query="Quelle est la capitale de la France ?",
        context=fr_chunks,
    )


@pytest.fixture(scope="module")
def en_result(generator: OllamaGenerator, en_chunks: list[Chunk]) -> GenerationResult:
    """Single real LLM call (English) — reused across all EN assertions."""
    return generator.generate(
        query="What is the capital of France?",
        context=en_chunks,
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestOllamaGeneratorIntegration:
    """Integration tests for OllamaGenerator against a live Ollama server.

    Requires Ollama running at http://localhost:11434 with the model pulled:
        ollama pull gemma3:4b

    The entire class is skipped automatically when the server is unreachable
    or the model is absent — no manual skip flag needed.
    """

    # ------------------------------------------------------------------
    # French query — result structure
    # ------------------------------------------------------------------

    def test_fr_result_is_generation_result_instance(
        self, fr_result: GenerationResult
    ) -> None:
        assert isinstance(fr_result, GenerationResult)

    def test_fr_answer_is_non_empty(self, fr_result: GenerationResult) -> None:
        assert fr_result.answer.strip() != ""

    def test_fr_detected_language_is_fr(self, fr_result: GenerationResult) -> None:
        assert fr_result.detected_language == "fr"

    def test_fr_sources_are_subset_of_context_ids(
        self, fr_result: GenerationResult, fr_chunks: list[Chunk]
    ) -> None:
        expected = {c.chunk_id for c in fr_chunks}
        assert set(fr_result.sources).issubset(expected)

    def test_fr_sources_are_non_empty(self, fr_result: GenerationResult) -> None:
        assert len(fr_result.sources) > 0

    def test_fr_tokens_used_is_positive_integer(
        self, fr_result: GenerationResult
    ) -> None:
        assert fr_result.tokens_used is not None
        assert fr_result.tokens_used > 0

    def test_fr_model_name_matches_config(self, fr_result: GenerationResult) -> None:
        assert fr_result.model == _MODEL

    def test_fr_prompt_version_is_non_empty(self, fr_result: GenerationResult) -> None:
        assert fr_result.prompt_version != ""

    # ------------------------------------------------------------------
    # English query — language detection end-to-end
    # ------------------------------------------------------------------

    def test_en_answer_is_non_empty(self, en_result: GenerationResult) -> None:
        assert en_result.answer.strip() != ""

    def test_en_detected_language_is_en(self, en_result: GenerationResult) -> None:
        assert en_result.detected_language == "en"

    def test_en_tokens_used_is_positive_integer(
        self, en_result: GenerationResult
    ) -> None:
        assert en_result.tokens_used is not None
        assert en_result.tokens_used > 0

    # ------------------------------------------------------------------
    # Input validation — no LLM call, immediate ValueError
    # ------------------------------------------------------------------

    def test_empty_query_raises_value_error(
        self, generator: OllamaGenerator, fr_chunks: list[Chunk]
    ) -> None:
        with pytest.raises(ValueError, match="query"):
            generator.generate(query="", context=fr_chunks)

    def test_whitespace_only_query_raises_value_error(
        self, generator: OllamaGenerator, fr_chunks: list[Chunk]
    ) -> None:
        with pytest.raises(ValueError, match="query"):
            generator.generate(query="   ", context=fr_chunks)

    def test_empty_context_raises_value_error(self, generator: OllamaGenerator) -> None:
        with pytest.raises(ValueError, match="context"):
            generator.generate(
                query="Quelle est la capitale de la France ?", context=[]
            )
