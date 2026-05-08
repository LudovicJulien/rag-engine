# src/generation/__init__.py
from src.generation.generator import GenerationResult, LLMGenerator
from src.generation.ollama_generator import OllamaGenerator, OllamaGeneratorConfig

__all__ = [
    "GenerationResult",
    "LLMGenerator",
    "OllamaGenerator",
    "OllamaGeneratorConfig",
]
