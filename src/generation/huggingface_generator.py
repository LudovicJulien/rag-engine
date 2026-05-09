# src/generation/huggingface_generator.py
from __future__ import annotations

import logging
from dataclasses import dataclass

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from src.generation.generator import GenerationResult, LLMGenerator
from src.generation.prompt_templates import get_template
from src.pipeline.config import Settings
from src.shared.models import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HuggingFaceGeneratorConfig:
    """Immutable configuration for :class:`HuggingFaceGenerator`.

    Attributes:
        model: HuggingFace model repository ID
            (e.g. ``"mistralai/Mistral-7B-Instruct-v0.3"``).
        api_token: HuggingFace API token.  Empty string falls back to
            anonymous access (heavily rate-limited — not recommended for
            production use).
        temperature: Sampling temperature in [0.0, 2.0].  Lower values
            produce more deterministic, factual outputs — recommended for RAG.
        max_new_tokens: Maximum number of tokens the model may generate.
            Does not count the prompt tokens.

    Raises:
        ValueError: If *model* is empty, *temperature* is out of range, or
            *max_new_tokens* < 1.
    """

    model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    api_token: str = ""
    temperature: float = 0.1
    max_new_tokens: int = 512
    domain: str = "general"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("HuggingFaceGeneratorConfig.model cannot be empty")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be in [0.0, 2.0], got {self.temperature}"
            )
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class HuggingFaceGenerator(LLMGenerator):
    """Generate answers via the HuggingFace Inference API using chat completion.

    Uses :class:`huggingface_hub.InferenceClient` and targets any
    instruction-tuned model hosted on the HuggingFace Hub that supports the
    ``/v1/chat/completions`` endpoint.

    Instances are stateless beyond their config and client handle, and are
    safe to share across threads.

    Args:
        config: Immutable generator configuration.  Defaults to
            ``HuggingFaceGeneratorConfig()``
            (``mistralai/Mistral-7B-Instruct-v0.3`` with anonymous access).

    Example::

        config = HuggingFaceGeneratorConfig(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            api_token="hf_...",
        )
        gen = HuggingFaceGenerator(config)
        result = gen.generate(query="What is Paris?", context=[chunk])

    Raises:
        RuntimeError: If the HuggingFace Inference API is unreachable,
            returns an HTTP error, or produces an empty response during
            :meth:`generate`.
    """

    def __init__(
        self,
        config: HuggingFaceGeneratorConfig = HuggingFaceGeneratorConfig(),
    ) -> None:
        self._config = config
        self._client = InferenceClient(
            model=config.model,
            token=config.api_token or None,
        )
        logger.debug(
            "HuggingFaceGenerator ready | model=%s",
            config.model,
        )

    # ------------------------------------------------------------------
    # LLMGenerator interface
    # ------------------------------------------------------------------

    def generate(self, query: str, context: list[Chunk]) -> GenerationResult:
        """Generate an answer using the HuggingFace Inference API.

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
            RuntimeError: If the HuggingFace API returns an HTTP error,
                the model is not found, authentication fails, or the
                response is empty.
        """
        self._validate_inputs(query, context)

        logger.debug(
            "HuggingFace generate | model=%s query_len=%d ctx_chunks=%d",
            self._config.model,
            len(query),
            len(context),
        )

        tpl = get_template(domain=self._config.domain)
        try:
            response = self._client.chat_completion(
                messages=[
                    {"role": "system", "content": tpl.system_prompt},
                    {
                        "role": "user",
                        "content": tpl.build_user_message(query, context),
                    },
                ],
                temperature=self._config.temperature,
                max_tokens=self._config.max_new_tokens,
            )
        except HfHubHTTPError as exc:
            raise RuntimeError(
                f"HuggingFace API error for model '{self._config.model}': {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected error during HuggingFace generation: {exc}"
            ) from exc

        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise RuntimeError(
                f"HuggingFace returned an empty response for model "
                f"'{self._config.model}'."
            )

        tokens_used: int | None = None
        if response.usage is not None:
            tokens_used = (
                response.usage.prompt_tokens + response.usage.completion_tokens
            )

        logger.info(
            "HuggingFace generate done | model=%s tokens=%s chunks=%d",
            self._config.model,
            tokens_used,
            len(context),
        )

        return GenerationResult(
            answer=answer,
            sources=[chunk.chunk_id for chunk in context],
            detected_language="fr",  # placeholder — detect_language added later
            model=self._config.model,
            prompt_version=tpl.version,
            tokens_used=tokens_used,
        )

    @property
    def provider_name(self) -> str:
        return "huggingface"

    @property
    def model_name(self) -> str:
        return self._config.model

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings) -> HuggingFaceGenerator:
        """Build a :class:`HuggingFaceGenerator` from application settings.

        Maps :attr:`~src.pipeline.config.Settings.llm_model` and
        :attr:`~src.pipeline.config.Settings.llm_api_key` to the
        corresponding config fields.  All other config values use defaults.

        Args:
            settings: Populated application settings instance.

        Returns:
            A configured :class:`HuggingFaceGenerator` ready to call.
        """
        return cls(
            HuggingFaceGeneratorConfig(
                model=settings.llm_model,
                api_token=settings.llm_api_key,
                domain=settings.llm_domain,
            )
        )
