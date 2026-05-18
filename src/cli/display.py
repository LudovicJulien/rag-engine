from __future__ import annotations

import dataclasses
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from qdrant_client import QdrantClient
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.cli.commands import (
    DiagnosticsReport,
    HealthStatus,
    OutputFormat,
    ServiceHealth,
)
from src.embeddings.bm25_embedder import BM25SparseEmbedder
from src.pipeline.config import Settings
from src.pipeline.ingest_pipeline import IngestResult
from src.pipeline.rag_pipeline import RAGPipeline, RAGResult

console = Console()

# Field names whose values are replaced with first-four-chars + "****".
_SECRET_FIELDS = frozenset(
    {"llm_api_key", "langfuse_public_key", "langfuse_secret_key"}
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _mask(value: str) -> str:
    """Return first 4 chars + '****', or just '****' for short values."""
    return value[:4] + "****" if len(value) > 4 else "****"


def _icon(ok: bool) -> str:
    return "[green]✓[/green]" if ok else "[red]✗[/red]"


def _svc_line(label: str, svc: ServiceHealth) -> str:
    """Format a single service-health row: padded label + icon + detail."""
    return f"{label:<13}{_icon(svc.ok)}  {svc.detail}"


def _masked_settings(settings: Settings) -> dict[str, str]:
    return {
        name: (
            _mask(str(getattr(settings, name)))
            if name in _SECRET_FIELDS
            else str(getattr(settings, name))
        )
        for name in settings.model_fields
    }


def _health_dict(status: HealthStatus) -> dict[str, Any]:
    return {
        "qdrant": dataclasses.asdict(status.qdrant),
        "llm": dataclasses.asdict(status.llm),
        "all_ok": status.all_ok,
    }


def _diagnostics_dict(report: DiagnosticsReport) -> dict[str, Any]:
    return {
        "health": _health_dict(report.health),
        "config": {
            "bm25_cache": dataclasses.asdict(report.config.bm25_cache),
            "collection": dataclasses.asdict(report.config.collection),
            "all_ok": report.config.all_ok,
        },
        "collection_count": report.collection_count,
        "bm25_vocab_size": report.bm25_vocab_size,
        "all_ok": report.all_ok,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@contextmanager
def spinner(description: str) -> Generator[None, None, None]:
    """Context manager that shows a Rich dots spinner while the block runs."""
    with console.status(description, spinner="dots"):
        yield


def error(msg: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {msg}")


def show_ingest_result(result: IngestResult, fmt: OutputFormat) -> None:
    if fmt == OutputFormat.json:
        print(json.dumps(dataclasses.asdict(result)))
        return
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold", min_width=16)
    table.add_column()
    table.add_row("Source", Path(result.source).name)
    table.add_row("Loaded", f"{result.chunks_loaded} chunks")
    table.add_row("Upserted", str(result.chunks_upserted))
    table.add_row("Failed", str(result.chunks_failed))
    table.add_row("BM25 cache", result.bm25_cache_path)
    console.print(table)


def show_rag_result(result: RAGResult, fmt: OutputFormat) -> None:
    if fmt == OutputFormat.json:
        print(json.dumps(dataclasses.asdict(result)))
        return
    console.print(Panel(result.answer, title="Answer"))
    table = Table(box=None, show_edge=False)
    table.add_column("Sources", min_width=16)
    table.add_column("Model", min_width=16)
    table.add_column("Language", min_width=10)
    table.add_column("Chunks", justify="right")
    table.add_column("Tokens", justify="right")
    sources = result.sources if result.sources else ["—"]
    tokens = str(result.tokens_used) if result.tokens_used is not None else "—"
    for i, source in enumerate(sources):
        if i == 0:
            table.add_row(
                source,
                result.model,
                result.detected_language,
                str(result.chunks_retrieved),
                tokens,
            )
        else:
            table.add_row(source, "", "", "", "")
    console.print(table)


def show_health(status: HealthStatus, fmt: OutputFormat) -> None:
    if fmt == OutputFormat.json:
        print(json.dumps(_health_dict(status)))
        return
    console.print(_svc_line("Qdrant", status.qdrant))
    console.print(_svc_line("LLM", status.llm))


def show_up_result(
    pipeline: RAGPipeline, settings: Settings, fmt: OutputFormat
) -> None:
    """Display pipeline-ready confirmation with component summary."""
    bm25_vocab: int | None = None
    collection_count: int | None = None
    try:
        bm25_vocab = BM25SparseEmbedder.load(settings.bm25_cache_path).embedding_dim
    except Exception:
        pass
    try:
        count_result = QdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port, timeout=2
        ).count(collection_name=settings.collection_name, exact=True)
        collection_count = int(count_result.count)
    except Exception:
        pass

    if fmt == OutputFormat.json:
        data: dict[str, Any] = {
            "status": "ready",
            "dense_model": settings.embedding_model,
            "bm25_cache_path": str(settings.bm25_cache_path),
            "bm25_vocab_size": bm25_vocab,
            "qdrant_addr": (f"{settings.qdrant_host}:{settings.qdrant_port}"),
            "collection": settings.collection_name,
            "collection_count": collection_count,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
        }
        print(json.dumps(data))
        return

    console.print("[bold green]Pipeline ready[/bold green]")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", width=14)
    table.add_column()
    vocab_sfx = f"   vocab={bm25_vocab}" if bm25_vocab is not None else ""
    count_sfx = f"  {collection_count} points" if collection_count is not None else ""
    addr = f"{settings.qdrant_host}:{settings.qdrant_port}"
    table.add_row("Dense model", settings.embedding_model)
    table.add_row("BM25", f"{settings.bm25_cache_path}{vocab_sfx}")
    table.add_row("Qdrant", f"{addr} / {settings.collection_name}{count_sfx}")
    table.add_row("LLM", f"{settings.llm_provider} / {settings.llm_model}")
    console.print(table)


def show_doctor(report: DiagnosticsReport, fmt: OutputFormat) -> None:
    if fmt == OutputFormat.json:
        print(json.dumps(_diagnostics_dict(report)))
        return

    console.rule("Connectivity")
    console.print(_svc_line("Qdrant", report.health.qdrant))
    console.print(_svc_line("LLM", report.health.llm))
    console.print()

    console.rule("Configuration")
    console.print(_svc_line("BM25 cache", report.config.bm25_cache))
    console.print(_svc_line("Collection", report.config.collection))
    console.print()

    console.rule("Statistics")
    count_str = (
        f"{report.collection_count} points"
        if report.collection_count is not None
        else "unavailable"
    )
    vocab_str = (
        f"{report.bm25_vocab_size} terms"
        if report.bm25_vocab_size is not None
        else "unavailable"
    )
    console.print(f"Collection   {count_str}")
    console.print(f"BM25 vocab   {vocab_str}")
    console.print()

    if report.all_ok:
        console.print("[bold green]All checks passed.[/bold green]")
    else:
        console.print("[bold red]Some checks failed — see details above.[/bold red]")


def show_config(settings: Settings, fmt: OutputFormat) -> None:
    masked = _masked_settings(settings)
    if fmt == OutputFormat.json:
        print(json.dumps(masked))
        return
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan", min_width=24)
    table.add_column()
    for key, value in masked.items():
        table.add_row(key, value)
    console.print(table)
