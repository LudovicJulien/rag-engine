# src/generation/generator.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.shared.models import Chunk

_SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"fr", "en"})


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """The output of a single LLM generation call.

    Attributes:
        answer: The generated answer text.
        sources: Ordered list of ``chunk_id`` values cited in the answer.
        detected_language: ISO-639-1 code of the incoming query
            (``"fr"`` or ``"en"``).
        model: Exact model identifier used for generation
            (e.g. ``"claude-sonnet-4-6"`` or ``"gemma3:4b"``).
        prompt_version: Version tag of the prompt template that produced
            this result (e.g. ``"v1"``). Stored so evaluation runs can
            be compared across prompt iterations.
        tokens_used: Total tokens consumed (prompt + completion).
            ``None`` when the provider does not expose token counts.

    Raises:
        ValueError: If *answer* is empty, *detected_language* is not a
            supported code, or *model* / *prompt_version* are empty.
    """

    answer: str
    sources: list[str]
    detected_language: str
    model: str
    prompt_version: str
    tokens_used: int | None = field(default=None)

    def __post_init__(self) -> None:
        if not self.answer:
            raise ValueError("GenerationResult.answer cannot be empty")
        if self.detected_language not in _SUPPORTED_LANGUAGES:
            raise ValueError(
                f"detected_language must be one of {sorted(_SUPPORTED_LANGUAGES)}, "
                f"got {self.detected_language!r}"
            )
        if not self.model:
            raise ValueError("GenerationResult.model cannot be empty")
        if not self.prompt_version:
            raise ValueError("GenerationResult.prompt_version cannot be empty")


class LLMGenerator(ABC):
    """Abstract base class for all LLM generation implementations.

    Defines the contract that any generation backend must fulfill to
    integrate with the RAG engine. The engine depends only on this
    interface — it does not care whether generation is performed by
    Ollama, HuggingFace, Anthropic, OpenAI, or any other provider.

    Concrete implementations must call ``_validate_inputs`` at the
    start of their ``generate`` override to enforce the shared
    pre-conditions documented in that method's docstring.

    Example usage::

        class OllamaGenerator(LLMGenerator):
            def generate(self, query: str, context: list[Chunk]) -> GenerationResult:
                self._validate_inputs(query, context)
                ...

            @property
            def provider_name(self) -> str:
                return "ollama"

            @property
            def model_name(self) -> str:
                return self._model
    """

    @abstractmethod
    def generate(self, query: str, context: list[Chunk]) -> GenerationResult:
        """Generate an answer from the query and retrieved context chunks.

        Implementations **must** call ``_validate_inputs`` first.

        Args:
            query: The user's question or request. Must be non-empty.
            context: Ordered list of retrieved :class:`~src.shared.models.Chunk`
                objects (highest relevance first). Must be non-empty.

        Returns:
            A :class:`GenerationResult` containing the answer, cited source
            ``chunk_id`` values, detected language, and generation metadata.

        Raises:
            ValueError: If *query* is empty or *context* is empty.
            RuntimeError: On provider network or API errors.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the identifier of the LLM provider.

        Used for logging and MLflow tracking to identify which backend
        produced the generation.

        Returns:
            Short lowercase string, e.g. ``"ollama"``, ``"anthropic"``,
            ``"huggingface"``.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the exact name of the underlying model.

        Used for logging, MLflow tracking, and stored in
        :attr:`GenerationResult.model` so evaluation runs can be
        compared across model iterations.

        Returns:
            String identifier of the model, e.g. ``"gemma3:4b"`` or
            ``"claude-sonnet-4-6"``.
        """

    @staticmethod
    def _validate_inputs(query: str, context: list[Chunk]) -> None:
        """Validate pre-conditions shared by all ``generate`` implementations.

        Args:
            query: The user query passed to :meth:`generate`.
            context: The context list passed to :meth:`generate`.

        Raises:
            ValueError: If *query* is empty or *context* is empty.
        """
        if not query or not query.strip():
            raise ValueError("'query' must not be empty.")
        if not context:
            raise ValueError("'context' must not be empty.")
