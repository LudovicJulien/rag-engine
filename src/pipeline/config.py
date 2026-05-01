# src/pipeline/config.py
from __future__ import annotations

import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = os.getenv("ENV_FILE", ".env")


class Settings(BaseSettings):
    """Centralized application configuration.

    Values are loaded from:
    1. Environment variables
    2. .env file
    3. Default values defined below

    Environment variables always take precedence.
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = Field(
        default="localhost",
        description="Qdrant server hostname",
    )
    qdrant_port: int = Field(
        default=6333,
        description="Qdrant server port",
    )
    collection_name: str = Field(
        default="rag-collection",
        description="Qdrant collection name",
    )

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-large",
        description="Embedding model identifier",
    )
    embedding_batch_size: int = Field(
        default=32, description="Embedding inference batch size"
    )

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512, description="Maximum chunk size")
    chunk_overlap: int = Field(default=64, description="Chunk overlap size")

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_provider: str = Field(default="ollama", description="LLM provider identifier")
    llm_base_url: str = Field(
        default="http://localhost:11434", description="Base URL for provider API"
    )
    llm_model: str = Field(
        default="gemma3:4b", description="Model name used by the selected provider"
    )
    llm_api_key: str = Field(default="", description="API key for remote providers")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = Field(
        default=5,
        description="Number of retrieved chunks",
    )
    score_threshold: float = Field(
        default=0.7,
        description="Minimum similarity threshold",
    )

    # ── Logging ───────────────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator(
        "qdrant_host",
        "collection_name",
        "embedding_model",
        "llm_provider",
        "llm_model",
        "log_level",
    )
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be empty")
        return value

    @field_validator("qdrant_port")
    @classmethod
    def validate_qdrant_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"qdrant_port must be between 1 and 65535, got {v}")
        return v

    @field_validator("embedding_batch_size")
    @classmethod
    def validate_embedding_batch_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"embedding_batch_size must be >= 1, got {v}")
        return v

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"chunk_size must be >= 1, got {v}")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {v}")
        return v

    @field_validator("llm_provider")
    @classmethod
    def normalize_llm_provider(cls, value: str) -> str:
        return value.lower().strip()

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = value.upper()
        if normalized not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{value}'")
        return normalized

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"top_k must be >= 1, got {v}")
        return v

    @field_validator("score_threshold")
    @classmethod
    def validate_score_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"score_threshold must be between 0.0 and 1.0, got {v}")
        return v


# Singleton instance
settings = Settings()
