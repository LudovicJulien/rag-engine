from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TraceContext:
    """Opaque handle returned by start_trace and threaded through all span calls."""

    trace_id: str


class PipelineTracer(ABC):
    """Contract for tracing a RAG pipeline request end-to-end.

    Inject an implementation into RAGPipeline and IngestionPipeline.
    Use NoOpTracer when observability is disabled — zero overhead, no imports.
    """

    @abstractmethod
    def start_trace(
        self,
        name: str,
        *,
        query: str,
        metadata: dict[str, Any] | None = None,
    ) -> TraceContext:
        """Open a new trace and return its context."""
        ...

    @abstractmethod
    def start_span(
        self,
        ctx: TraceContext,
        name: str,
        *,
        span_input: dict[str, Any],
    ) -> str:
        """Open a child span inside *ctx*. Returns an opaque span_id."""
        ...

    @abstractmethod
    def end_span(
        self,
        ctx: TraceContext,
        span_id: str,
        *,
        output: dict[str, Any],
        latency_ms: int | None = None,
    ) -> None:
        """Close the span identified by *span_id*."""
        ...

    @abstractmethod
    def record_generation(
        self,
        ctx: TraceContext,
        *,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        latency_ms: int | None = None,
    ) -> None:
        """Record an LLM generation event (tokens, model, latency)."""
        ...

    @abstractmethod
    def score(self, ctx: TraceContext, name: str, value: float) -> None:
        """Attach a named score to an existing trace (e.g. a RAGAS metric)."""
        ...

    @abstractmethod
    def end_trace(
        self,
        ctx: TraceContext,
        *,
        output: str,
        latency_ms: int | None = None,
    ) -> None:
        """Close the trace with the final answer and total latency."""
        ...


class NoOpTracer(PipelineTracer):
    """Null implementation — every method is a no-op with the correct return type.

    Default when observability_enabled=False or Langfuse keys are absent.
    Guarantees zero overhead and zero coupling: callers never need to branch on
    whether tracing is active.
    """

    def start_trace(
        self,
        name: str,
        *,
        query: str,
        metadata: dict[str, Any] | None = None,
    ) -> TraceContext:
        return TraceContext(trace_id="")

    def start_span(
        self,
        ctx: TraceContext,
        name: str,
        *,
        span_input: dict[str, Any],
    ) -> str:
        return ""

    def end_span(
        self,
        ctx: TraceContext,
        span_id: str,
        *,
        output: dict[str, Any],
        latency_ms: int | None = None,
    ) -> None:
        pass

    def record_generation(
        self,
        ctx: TraceContext,
        *,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        latency_ms: int | None = None,
    ) -> None:
        pass

    def score(self, ctx: TraceContext, name: str, value: float) -> None:
        pass

    def end_trace(
        self,
        ctx: TraceContext,
        *,
        output: str,
        latency_ms: int | None = None,
    ) -> None:
        pass
