from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


@dataclass(frozen=True, slots=True)
class ServiceHealth:
    ok: bool
    detail: str  # e.g. "connected" | "unreachable at <url>" | "file found"


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """Network connectivity only — Qdrant and LLM."""

    qdrant: ServiceHealth
    llm: ServiceHealth

    @property
    def all_ok(self) -> bool:
        return self.qdrant.ok and self.llm.ok


@dataclass(frozen=True, slots=True)
class ConfigStatus:
    """Local configuration validity — files and collection state."""

    bm25_cache: ServiceHealth
    collection: ServiceHealth

    @property
    def all_ok(self) -> bool:
        return self.bm25_cache.ok and self.collection.ok


@dataclass(frozen=True, slots=True)
class DiagnosticsReport:
    """Full diagnostic report: config + infra + statistics."""

    config: ConfigStatus
    health: HealthStatus
    collection_count: int | None  # None if Qdrant unreachable
    bm25_vocab_size: int | None  # None if BM25 absent or unreadable

    @property
    def all_ok(self) -> bool:
        return self.config.all_ok and self.health.all_ok
