from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import typer
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from src.embeddings.bm25_embedder import BM25SparseEmbedder
from src.pipeline.config import Settings, get_settings
from src.pipeline.ingest_pipeline import IngestionPipeline

if TYPE_CHECKING:
    from src.pipeline.rag_pipeline import RAGPipeline


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


# ---------------------------------------------------------------------------
# Config probes — file system and collection state, never raises
# ---------------------------------------------------------------------------


def _probe_bm25_cache(settings: Settings) -> ServiceHealth:
    path = settings.bm25_cache_path
    if path.exists():
        return ServiceHealth(ok=True, detail=f"file found at {path}")
    return ServiceHealth(ok=False, detail=f"file missing at {path}")


def _probe_collection(settings: Settings) -> ServiceHealth:
    name = settings.collection_name
    addr = f"{settings.qdrant_host}:{settings.qdrant_port}"
    try:
        QdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port, timeout=2
        ).get_collection(collection_name=name)
        return ServiceHealth(ok=True, detail=f"{name} exists")
    except UnexpectedResponse:
        # 404 — Qdrant is up but the collection was never created
        return ServiceHealth(ok=False, detail=f"{name} not found")
    except Exception:
        return ServiceHealth(ok=False, detail=f"qdrant unreachable at {addr}")


def check_config(settings: Settings) -> ConfigStatus:
    """Local configuration validity — files and collection state.

    BM25 cache : checks settings.bm25_cache_path exists on disk (no parsing).
    Collection : calls QdrantClient.get_collection() with a 2 s timeout;
                 reports unreachable if Qdrant is down.
    """
    return ConfigStatus(
        bm25_cache=_probe_bm25_cache(settings),
        collection=_probe_collection(settings),
    )


# ---------------------------------------------------------------------------
# Statistics helpers — return None rather than raising
# ---------------------------------------------------------------------------


def _get_collection_count(settings: Settings) -> int | None:
    try:
        result = QdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port, timeout=2
        ).count(collection_name=settings.collection_name, exact=True)
        return int(result.count)
    except Exception:
        return None


def _get_bm25_vocab_size(settings: Settings) -> int | None:
    try:
        return BM25SparseEmbedder.load(settings.bm25_cache_path).embedding_dim
    except Exception:
        return None


def build_diagnostics(settings: Settings) -> DiagnosticsReport:
    """Aggregate check_health + check_config + statistics into one report.

    collection_count : QdrantClient.count() or None if Qdrant unreachable.
    bm25_vocab_size  : BM25SparseEmbedder.embedding_dim or None if absent.
    """
    health = check_health(settings)
    config = check_config(settings)
    return DiagnosticsReport(
        health=health,
        config=config,
        collection_count=(
            _get_collection_count(settings) if health.qdrant.ok else None
        ),
        bm25_vocab_size=_get_bm25_vocab_size(settings),
    )


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


# ---------------------------------------------------------------------------
# Pipeline singleton — one RAGPipeline per process
# ---------------------------------------------------------------------------

_pipeline: RAGPipeline | None = None


def _get_or_build_pipeline(settings: Settings) -> RAGPipeline:
    """Return the cached pipeline, building it on first call."""
    global _pipeline
    if _pipeline is None:
        from src.pipeline.rag_pipeline import RAGPipeline as _RAGPipeline

        _pipeline = _RAGPipeline.build(settings)
    return _pipeline


# ---------------------------------------------------------------------------
# Typer commands
# ---------------------------------------------------------------------------


def cmd_ingest(
    source: Path = typer.Argument(..., help="Chemin vers le fichier JSON pré-chunké"),
    reset: bool = typer.Option(
        False, "--reset", help="Supprime et recrée la collection avant l'upsert"
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", "-o"),
) -> None:
    """Indexe un fichier JSON dans Qdrant et persiste le modèle BM25."""
    import src.cli.display as display  # lazy — display imports from commands

    try:
        with display.spinner("Ingesting documents…"):
            result = IngestionPipeline.build(get_settings()).run(source, reset=reset)
    except FileNotFoundError as e:
        display.error(str(e))
        raise typer.Exit(1)
    except RuntimeError as e:
        display.error(str(e))
        raise typer.Exit(2)
    display.show_ingest_result(result, output)
