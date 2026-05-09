# src/generation/anthropic_generator.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import anthropic

from src.generation.generator import GenerationResult, LLMGenerator
from src.pipeline.config import Settings
from src.shared.models import Chunk

logger = logging.getLogger(__name__)

_PROMPT_VERSION = "v0"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnthropicGeneratorConfig:
    """Immutable configuration for :class:`AnthropicGenerator`.

    Attributes:
        model: Anthropic model ID (e.g. ``"claude-haiku-4-5"``).
        api_key: Anthropic API key.  Empty string falls back to the
            ``ANTHROPIC_API_KEY`` environment variable.
        max_tokens: Maximum number of tokens the model may generate.
            Must be >= 1.
        temperature: Sampling temperature in [0.0, 1.0].  ``None`` omits
            the parameter entirely — required for models such as
            ``claude-opus-4-7`` that do not accept a temperature argument.

    Raises:
        ValueError: If *model* is empty, *max_tokens* < 1, or *temperature*
            is not ``None`` and outside [0.0, 1.0].
    """

    model: str = "claude-haiku-4-5"
    api_key: str = ""
    max_tokens: int = 1024
    temperature: float | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("AnthropicGeneratorConfig.model cannot be empty")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.temperature is not None and not 0.0 <= self.temperature <= 1.0:
            raise ValueError(
                f"temperature must be in [0.0, 1.0] or None, got {self.temperature}"
            )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class AnthropicGenerator(LLMGenerator):
    """Generate answers via the Anthropic Messages API using Claude models.

    Uses the ``anthropic`` Python SDK.  Token counts are always available
    from the API response and are always included in :class:`GenerationResult`.

    Instances are stateless beyond their config and client handle, and are
    safe to share across threads.

    Args:
        config: Immutable generator configuration.  Defaults to
            ``AnthropicGeneratorConfig()`` (``claude-haiku-4-5`` with env-var key).

    Example::

        config = AnthropicGeneratorConfig(
            model="claude-haiku-4-5",
            api_key="sk-ant-...",
        )
        gen = AnthropicGenerator(config)
        result = gen.generate(query="What is Paris?", context=[chunk])

    Raises:
        RuntimeError: If the Anthropic API returns an authentication, rate-limit,
            status, or connection error, or produces an empty response during
            :meth:`generate`.
    """

    def __init__(
        self,
        config: AnthropicGeneratorConfig = AnthropicGeneratorConfig(),
    ) -> None:
        self._config = config
        self._client = anthropic.Anthropic(api_key=config.api_key or None)
        logger.debug(
            "AnthropicGenerator ready | model=%s",
            config.model,
        )

    # ------------------------------------------------------------------
    # LLMGenerator interface
    # ------------------------------------------------------------------

    def generate(self, query: str, context: list[Chunk]) -> GenerationResult:
        """Generate an answer using the Anthropic Messages API.

        Args:
            query: User question. Must be non-empty.
            context: Retrieved chunks ordered by descending relevance.
                Must be non-empty.

        Returns:
            A :class:`~src.generation.generator.GenerationResult` containing
            the answer, all context ``chunk_id`` values as sources, detected
            language, model name, and token counts.

        Raises:
            ValueError: If *query* is empty or *context* is empty.
            RuntimeError: If the Anthropic API returns an error or an
                empty response.
        """
        self._validate_inputs(query, context)

        logger.debug(
            "Anthropic generate | model=%s query_len=%d ctx_chunks=%d",
            self._config.model,
            len(query),
            len(context),
        )

        create_kwargs: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "system": self._build_system_message(),
            "messages": [
                {
                    "role": "user",
                    "content": self._build_user_message(query, context),
                }
            ],
        }
        if self._config.temperature is not None:
            create_kwargs["temperature"] = self._config.temperature

        try:
            response = self._client.messages.create(**create_kwargs)
        except anthropic.AuthenticationError as exc:
            raise RuntimeError(
                f"Anthropic authentication failed for model "
                f"'{self._config.model}': {exc}"
            ) from exc
        except anthropic.RateLimitError as exc:
            raise RuntimeError(
                f"Anthropic rate limit exceeded for model '{self._config.model}': {exc}"
            ) from exc
        except anthropic.APIStatusError as exc:
            raise RuntimeError(
                f"Anthropic API error for model '{self._config.model}': {exc}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise RuntimeError(
                f"Anthropic connection error for model '{self._config.model}': {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected error during Anthropic generation: {exc}"
            ) from exc

        if response.stop_reason == "max_tokens":
            logger.warning(
                "Anthropic generation hit max_tokens | model=%s max_tokens=%d",
                self._config.model,
                self._config.max_tokens,
            )

        answer = next(
            (block.text for block in response.content if block.type == "text"),
            "",
        ).strip()
        if not answer:
            raise RuntimeError(
                f"Anthropic returned an empty response for model "
                f"'{self._config.model}'."
            )

        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        logger.info(
            "Anthropic generate done | model=%s tokens=%d chunks=%d",
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
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._config.model

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings) -> AnthropicGenerator:
        """Build an :class:`AnthropicGenerator` from application settings.

        Maps :attr:`~src.pipeline.config.Settings.llm_model` and
        :attr:`~src.pipeline.config.Settings.llm_api_key` to the
        corresponding config fields.  All other config values use defaults.

        Args:
            settings: Populated application settings instance.

        Returns:
            A configured :class:`AnthropicGenerator` ready to call.
        """
        return cls(
            AnthropicGeneratorConfig(
                model=settings.llm_model,
                api_key=settings.llm_api_key,
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
