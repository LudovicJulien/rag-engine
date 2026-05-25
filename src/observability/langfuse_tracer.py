from __future__ import annotations

from typing import Any, cast

from src.observability.tracer import PipelineTracer, TraceContext


class LangfuseTracer(PipelineTracer):
    """Langfuse implementation via the official SDK.

    Maintains internal handles for in-flight traces and spans so callers
    only need to thread an opaque TraceContext — Langfuse objects never leak
    into the domain layer.

    Args:
        public_key: Langfuse project public key.
        secret_key: Langfuse project secret key.
        host: Langfuse server URL.

    Raises:
        ValueError: If public_key or secret_key is empty.
        ImportError: If the langfuse package is not installed.
    """

    def __init__(self, *, public_key: str, secret_key: str, host: str) -> None:
        if not public_key:
            raise ValueError("langfuse public_key must not be empty")
        if not secret_key:
            raise ValueError("langfuse secret_key must not be empty")

        try:
            from langfuse import Langfuse
        except ImportError as exc:
            raise ImportError(
                "Install langfuse to enable tracing: pip install 'langfuse>=2.0'"
            ) from exc

        self._client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        self._handles: dict[str, Any] = {}  # trace_id → langfuse.StatefulTraceClient
        self._spans: dict[str, Any] = {}  # span_id  → langfuse.StatefulSpanClient

    def start_trace(
        self,
        name: str,
        *,
        query: str,
        metadata: dict[str, Any] | None = None,
    ) -> TraceContext:
        trace = self._client.trace(name=name, input=query, metadata=metadata or {})
        ctx = TraceContext(trace_id=trace.id)
        self._handles[trace.id] = trace
        return ctx

    def start_span(
        self,
        ctx: TraceContext,
        name: str,
        *,
        span_input: dict[str, Any],
    ) -> str:
        trace = self._handles[ctx.trace_id]
        span = trace.span(name=name, input=span_input)
        span_id = cast(str, span.id)
        self._spans[span_id] = span
        return span_id

    def end_span(
        self,
        ctx: TraceContext,
        span_id: str,
        *,
        output: dict[str, Any],
        latency_ms: int | None = None,
    ) -> None:
        span = self._spans.pop(span_id, None)
        if span is not None:
            span.end(output=output)

    def record_generation(
        self,
        ctx: TraceContext,
        *,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        latency_ms: int | None = None,
    ) -> None:
        trace = self._handles[ctx.trace_id]
        trace.generation(
            model=model,
            usage={"input": prompt_tokens, "output": completion_tokens},
        )

    def score(self, ctx: TraceContext, name: str, value: float) -> None:
        self._client.score(trace_id=ctx.trace_id, name=name, value=value)

    def end_trace(
        self,
        ctx: TraceContext,
        *,
        output: str,
        latency_ms: int | None = None,
    ) -> None:
        trace = self._handles.pop(ctx.trace_id, None)
        if trace is not None:
            trace.update(output=output)
        self._client.flush()
