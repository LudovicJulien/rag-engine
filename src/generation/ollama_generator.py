# src/generation/ollama_generator.py
from __future__ import annotations

import logging
from dataclasses import dataclass

import ollama

from src.generation.generator import GenerationResult, LLMGenerator
from src.pipeline.config import Settings
from src.shared.models import Chunk

logger = logging.getLogger(__name__)

# Placeholder until prompt_templates.py is introduced in a later commit.
_PROMPT_VERSION = "v0"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OllamaGeneratorConfig:
    """Immutable configuration for :class:`OllamaGenerator`.

    Attributes:
        model: Ollama model tag to use (e.g. ``"gemma3:4b"``).
        base_url: Ollama server base URL.
        temperature: Sampling temperature in [0.0, 2.0].  Lower values produce
            more deterministic, factual outputs — recommended for RAG.
        num_ctx: Model context window size in tokens.

    Raises:
        ValueError: If *model* or *base_url* are empty, *temperature* is out of
            range, or *num_ctx* < 1.
    """

    model: str = "gemma3:4b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    num_ctx: int = 4096

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("OllamaGeneratorConfig.model cannot be empty")
        if not self.base_url.strip():
            raise ValueError("OllamaGeneratorConfig.base_url cannot be empty")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be in [0.0, 2.0], got {self.temperature}"
            )
        if self.num_ctx < 1:
            raise ValueError(f"num_ctx must be >= 1, got {self.num_ctx}")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class OllamaGenerator(LLMGenerator):
    """Generate answers via a local Ollama server using chat completion.

    Uses the ``ollama`` Python SDK.  The Ollama server must be running and
    the configured model must be pulled before calling :meth:`generate`.

    Instances are stateless beyond their config and client handle, and are
    safe to share across threads.

    Args:
        config: Immutable generator configuration.  Defaults to
            ``OllamaGeneratorConfig()`` (``gemma3:4b`` on localhost).

    Example::

        gen = OllamaGenerator(OllamaGeneratorConfig(model="gemma3:4b"))
        result = gen.generate(query="What is Paris?", context=[chunk])

    Raises:
        RuntimeError: If the Ollama server is unreachable or returns an error
            during :meth:`generate`.
    """

    def __init__(
        self,
        config: OllamaGeneratorConfig = OllamaGeneratorConfig(),
    ) -> None:
        self._config = config
        self._client = ollama.Client(host=config.base_url)
        logger.debug(
            "OllamaGenerator ready | model=%s base_url=%s",
            config.model,
            config.base_url,
        )

    # ------------------------------------------------------------------
    # LLMGenerator interface
    # ------------------------------------------------------------------

    def generate(self, query: str, context: list[Chunk]) -> GenerationResult:
        """Generate an answer using Ollama chat completion.

        Args:
            query: User question. Must be non-empty.
            context: Retrieved chunks ordered by descending relevance.
                Must be non-empty.

        Returns:
            A :class:`~src.generation.generator.GenerationResult` containing
            the answer, all context ``chunk_id`` values as sources, detected
            language, model name, and token counts when available.

        Raises:
            ValueError: If *query* is empty or *context* is empty.
            RuntimeError: If the Ollama server is unreachable, the model is
                not found, or the server returns an empty response.
        """
        self._validate_inputs(query, context)

        logger.debug(
            "Ollama generate | model=%s query_len=%d ctx_chunks=%d",
            self._config.model,
            len(query),
            len(context),
        )

        try:
            response = self._client.chat(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": self._build_system_message()},
                    {
                        "role": "user",
                        "content": self._build_user_message(query, context),
                    },
                ],
                options={
                    "temperature": self._config.temperature,
                    "num_ctx": self._config.num_ctx,
                },
            )
        except ollama.ResponseError as exc:
            raise RuntimeError(
                f"Ollama ResponseError for model '{self._config.model}': {exc}"
            ) from exc
        except ollama.RequestError as exc:
            raise RuntimeError(
                f"Ollama server unreachable at '{self._config.base_url}': {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected error during Ollama generation: {exc}"
            ) from exc

        answer = (response.message.content or "").strip()
        if not answer:
            raise RuntimeError(
                f"Ollama returned an empty response for model '{self._config.model}'."
            )

        tokens_used: int | None = None
        if response.prompt_eval_count is not None and response.eval_count is not None:
            tokens_used = response.prompt_eval_count + response.eval_count

        logger.info(
            "Ollama generate done | model=%s tokens=%s chunks=%d",
            self._config.model,
            tokens_used,
            len(context),
        )

        return GenerationResult(
            answer=answer,
            sources=[chunk.chunk_id for chunk in context],
            detected_language="fr",  # placeholder — detect_language added later
            model=self._config.model,
            prompt_version=_PROMPT_VERSION,
            tokens_used=tokens_used,
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._config.model

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings) -> OllamaGenerator:
        """Build an :class:`OllamaGenerator` from application settings.

        Maps :attr:`~src.pipeline.config.Settings.llm_model` and
        :attr:`~src.pipeline.config.Settings.llm_base_url` to the
        corresponding config fields.  All other config values use defaults.

        Args:
            settings: Populated application settings instance.

        Returns:
            A configured :class:`OllamaGenerator` ready to call.
        """
        return cls(
            OllamaGeneratorConfig(
                model=settings.llm_model,
                base_url=settings.llm_base_url,
            )
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_system_message() -> str:
        return (
            "You are a precise and factual assistant. "
            "Answer the user's question based solely on the provided context. "
            "If the context does not contain enough information to answer, "
            "say so clearly. "
            "When you use information from a source, cite its ID "
            "inline using the format [id: <chunk_id>]."
        )

    @staticmethod
    def _build_user_message(query: str, context: list[Chunk]) -> str:
        formatted_chunks = "\n\n".join(
            f"[{i + 1}] (id: {chunk.chunk_id})\n{chunk.text}"
            for i, chunk in enumerate(context)
        )
        return f"Context:\n{formatted_chunks}\n\nQuestion: {query}"
