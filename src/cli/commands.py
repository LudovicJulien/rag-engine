from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import httpx
from qdrant_client import QdrantClient

from src.pipeline.config import Settings


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


# ---------------------------------------------------------------------------
# Health probes — each returns ServiceHealth, never raises
# ---------------------------------------------------------------------------


def _probe_qdrant(settings: Settings) -> ServiceHealth:
    addr = f"{settings.qdrant_host}:{settings.qdrant_port}"
    try:
        QdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port, timeout=2
        ).get_collections()
        return ServiceHealth(ok=True, detail=f"connected at {addr}")
    except Exception:
        return ServiceHealth(ok=False, detail=f"unreachable at {addr}")


def _probe_ollama(base_url: str) -> ServiceHealth:
    try:
        httpx.get(f"{base_url}/api/tags", timeout=3.0)
        return ServiceHealth(ok=True, detail=f"ollama reachable at {base_url}")
    except Exception:
        return ServiceHealth(ok=False, detail=f"ollama unreachable at {base_url}")


def _probe_api_key(provider: str, api_key: str) -> ServiceHealth:
    if api_key:
        return ServiceHealth(ok=True, detail=f"{provider} key present")
    return ServiceHealth(ok=False, detail=f"{provider} key missing")


def _probe_llm(settings: Settings) -> ServiceHealth:
    if settings.llm_provider == "ollama":
        return _probe_ollama(settings.llm_base_url)
    return _probe_api_key(settings.llm_provider, settings.llm_api_key)


def check_health(settings: Settings) -> HealthStatus:
    """Network connectivity only — no file I/O.

    Qdrant: calls QdrantClient.get_collections() with a 2 s timeout.
    LLM   : for Ollama, GETs {llm_base_url}/api/tags with a 3 s timeout;
            for key-based providers, checks llm_api_key is non-empty.
    """
    return HealthStatus(
        qdrant=_probe_qdrant(settings),
        llm=_probe_llm(settings),
    )
