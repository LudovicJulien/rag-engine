# src/generation/__init__.py
from src.generation.anthropic_generator import (
    AnthropicGenerator,
    AnthropicGeneratorConfig,
)
from src.generation.factory import get_generator
from src.generation.generator import GenerationResult, LLMGenerator
from src.generation.huggingface_generator import (
    HuggingFaceGenerator,
    HuggingFaceGeneratorConfig,
)
from src.generation.language_detection import detect_language
from src.generation.ollama_generator import OllamaGenerator, OllamaGeneratorConfig
from src.generation.prompt_templates import (
    CURRENT_VERSION,
    DEFAULT_DOMAIN,
    DOMAIN_PROMPTS,
    PromptTemplate,
    get_template,
)

__all__ = [
    "AnthropicGenerator",
    "AnthropicGeneratorConfig",
    "CURRENT_VERSION",
    "DEFAULT_DOMAIN",
    "detect_language",
    "DOMAIN_PROMPTS",
    "GenerationResult",
    "HuggingFaceGenerator",
    "HuggingFaceGeneratorConfig",
    "LLMGenerator",
    "OllamaGenerator",
    "OllamaGeneratorConfig",
    "PromptTemplate",
    "get_generator",
    "get_template",
]
