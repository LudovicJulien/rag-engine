from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from src.observability.tracer import PipelineTracer, TraceContext

# ---------------------------------------------------------------------------
# Minimal concrete tracer used only in ABC contract tests
# ---------------------------------------------------------------------------


class _StubTracer(PipelineTracer):
    def start_trace(
        self, name: str, *, query: str, metadata: Any = None
    ) -> TraceContext:
        return TraceContext(trace_id="stub")

    def start_span(self, ctx: TraceContext, name: str, *, span_input: Any) -> str:
        return "span-stub"

    def end_span(
        self,
        ctx: TraceContext,
        span_id: str,
        *,
        output: Any,
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
        self, ctx: TraceContext, *, output: str, latency_ms: int | None = None
    ) -> None:
        pass


# ---------------------------------------------------------------------------
# TraceContext
# ---------------------------------------------------------------------------


class TestTraceContext:
    def test_stores_trace_id(self) -> None:
        ctx = TraceContext(trace_id="abc-123")
        assert ctx.trace_id == "abc-123"

    def test_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(TraceContext)

    def test_is_frozen(self) -> None:
        ctx = TraceContext(trace_id="abc-123")
        with pytest.raises(AttributeError):
            ctx.trace_id = "other"  # type: ignore[misc]

    def test_equality_by_value(self) -> None:
        assert TraceContext(trace_id="x") == TraceContext(trace_id="x")

    def test_inequality_on_different_id(self) -> None:
        assert TraceContext(trace_id="x") != TraceContext(trace_id="y")

    def test_empty_string_is_valid(self) -> None:
        # "" is the sentinel that NoOpTracer returns — must not raise
        ctx = TraceContext(trace_id="")
        assert ctx.trace_id == ""

    def test_repr_includes_trace_id(self) -> None:
        ctx = TraceContext(trace_id="t-99")
        assert "t-99" in repr(ctx)


# ---------------------------------------------------------------------------
# PipelineTracer ABC
# ---------------------------------------------------------------------------


class TestPipelineTracerABC:
    def test_cannot_be_instantiated_directly(self) -> None:
        with pytest.raises(TypeError):
            PipelineTracer()  # type: ignore[abstract]

    def test_partial_implementation_raises_typeerror(self) -> None:
        class _Partial(PipelineTracer):
            def start_trace(
                self, name: str, *, query: str, metadata: Any = None
            ) -> TraceContext:
                return TraceContext(trace_id="")

            # all other abstract methods missing

        with pytest.raises(TypeError):
            _Partial()  # type: ignore[abstract]

    def test_full_implementation_can_be_instantiated(self) -> None:
        tracer = _StubTracer()
        assert isinstance(tracer, PipelineTracer)

    def test_start_trace_returns_trace_context(self) -> None:
        ctx = _StubTracer().start_trace("pipeline", query="quel document ?")
        assert isinstance(ctx, TraceContext)

    def test_start_span_returns_string(self) -> None:
        tracer = _StubTracer()
        ctx = tracer.start_trace("t", query="q")
        span_id = tracer.start_span(ctx, "embedding", span_input={"query": "q"})
        assert isinstance(span_id, str)

    def test_end_span_accepts_optional_latency(self) -> None:
        tracer = _StubTracer()
        ctx = tracer.start_trace("t", query="q")
        sid = tracer.start_span(ctx, "s", span_input={})
        tracer.end_span(ctx, sid, output={})  # no latency_ms — must not raise
        tracer.end_span(ctx, sid, output={}, latency_ms=42)

    def test_record_generation_accepts_none_tokens(self) -> None:
        tracer = _StubTracer()
        ctx = tracer.start_trace("t", query="q")
        tracer.record_generation(
            ctx, model="llama3", prompt_tokens=None, completion_tokens=None
        )

    def test_score_accepts_float_value(self) -> None:
        tracer = _StubTracer()
        ctx = tracer.start_trace("t", query="q")
        tracer.score(ctx, "faithfulness", 0.87)

    def test_end_trace_accepts_optional_latency(self) -> None:
        tracer = _StubTracer()
        ctx = tracer.start_trace("t", query="q")
        tracer.end_trace(ctx, output="réponse")
        tracer.end_trace(ctx, output="réponse", latency_ms=200)

    def test_start_trace_passes_metadata(self) -> None:
        class _MetaCapture(PipelineTracer):
            captured_metadata: dict[str, Any] | None = None

            def start_trace(
                self, name: str, *, query: str, metadata: Any = None
            ) -> TraceContext:
                _MetaCapture.captured_metadata = metadata
                return TraceContext(trace_id="")

            def start_span(
                self, ctx: TraceContext, name: str, *, span_input: Any
            ) -> str:
                return ""

            def end_span(
                self,
                ctx: TraceContext,
                span_id: str,
                *,
                output: Any,
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
                self, ctx: TraceContext, *, output: str, latency_ms: int | None = None
            ) -> None:
                pass

        _MetaCapture().start_trace("t", query="q", metadata={"top_k": 5})
        assert _MetaCapture.captured_metadata == {"top_k": 5}
