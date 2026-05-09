# src/generation/factory.py
from __future__ import annotations

import logging
from typing import Callable

from src.generation.anthropic_generator import AnthropicGenerator
from src.generation.generator import LLMGenerator
from src.generation.huggingface_generator import HuggingFaceGenerator
from src.generation.ollama_generator import OllamaGenerator
from src.pipeline.config import Settings

logger = logging.getLogger(__name__)

# Maps provider name → from_settings constructor.
# Add new providers here as they are implemented.
_REGISTRY: dict[str, Callable[[Settings], LLMGenerator]] = {
    "ollama": OllamaGenerator.from_settings,
    "huggingface": HuggingFaceGenerator.from_settings,
    "anthropic": AnthropicGenerator.from_settings,
}

# Providers known to Settings but not yet implemented.
_NOT_IMPLEMENTED: frozenset[str] = frozenset({"openai", "gemini"})


def get_generator(settings: Settings) -> LLMGenerator:
    """Instantiate the appropriate :class:`~src.generation.generator.LLMGenerator`
    for the provider declared in *settings*.

    The provider is read from :attr:`~src.pipeline.config.Settings.llm_provider`
    and must be one of ``"ollama"``, ``"huggingface"``, or ``"anthropic"``.
    Providers ``"openai"`` and ``"gemini"`` are recognised by the settings
    schema but raise :exc:`NotImplementedError` until their generators are
    added to the registry.

    All provider-specific parameters (model, API key, base URL) are forwarded
    via each generator's ``from_settings`` class method.

    Args:
        settings: Populated application settings instance.

    Returns:
        A ready-to-use :class:`~src.generation.generator.LLMGenerator`.

    Raises:
        NotImplementedError: If *settings.llm_provider* is a known but
            not-yet-implemented provider (``"openai"``, ``"gemini"``).
        ValueError: If *settings.llm_provider* is not recognised at all.

    Example::

        from src.pipeline.config import get_settings
        from src.generation.factory import get_generator

        generator = get_generator(get_settings())
        result = generator.generate(query="What is Paris?", context=chunks)
    """
    provider = settings.llm_provider

    if provider in _NOT_IMPLEMENTED:
        raise NotImplementedError(
            f"LLM provider '{provider}' is not yet implemented. "
            f"Available providers: {sorted(_REGISTRY)}"
        )

    factory = _REGISTRY.get(provider)
    if factory is None:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Available providers: {sorted(_REGISTRY)}"
        )

    logger.debug("get_generator | provider=%s model=%s", provider, settings.llm_model)
    return factory(settings)
