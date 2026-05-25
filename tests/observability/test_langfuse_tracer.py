from __future__ import annotations

import sys
from typing import cast
from unittest.mock import MagicMock

import pytest

from src.observability.langfuse_tracer import LangfuseTracer
from src.observability.tracer import PipelineTracer, TraceContext

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_HOST = "https://cloud.langfuse.com"
_PK = "pk-test"
_SK = "sk-test"


@pytest.fixture
def mock_langfuse_module(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the langfuse SDK with a MagicMock for the duration of the test."""
    mock_module = MagicMock()
    monkeypatch.setitem(sys.modules, "langfuse", mock_module)
    return mock_module


@pytest.fixture
def mock_client(mock_langfuse_module: MagicMock) -> MagicMock:
    """The Langfuse(...) instance returned by the mocked constructor."""
    return cast(MagicMock, mock_langfuse_module.Langfuse.return_value)


@pytest.fixture
def mock_trace(mock_client: MagicMock) -> MagicMock:
    """Fake langfuse trace handle wired into mock_client.trace(...)."""
    trace = MagicMock()
    trace.id = "trace-abc"
    mock_client.trace.return_value = trace
    return trace


@pytest.fixture
def mock_span(mock_trace: MagicMock) -> MagicMock:
    """Fake langfuse span handle wired into mock_trace.span(...)."""
    span = MagicMock()
    span.id = "span-xyz"
    mock_trace.span.return_value = span
    return span


@pytest.fixture
def tracer(mock_langfuse_module: MagicMock, mock_client: MagicMock) -> LangfuseTracer:
    return LangfuseTracer(public_key=_PK, secret_key=_SK, host=_HOST)


# ---------------------------------------------------------------------------
# TestLangfuseTracerInit
# ---------------------------------------------------------------------------


class TestLangfuseTracerInit:
    def test_is_pipeline_tracer(
        self, mock_langfuse_module: MagicMock, mock_client: MagicMock
    ) -> None:
        assert isinstance(
            LangfuseTracer(public_key=_PK, secret_key=_SK, host=_HOST),
            PipelineTracer,
        )

    def test_constructs_client_with_correct_args(
        self, mock_langfuse_module: MagicMock, mock_client: MagicMock
    ) -> None:
        LangfuseTracer(
            public_key="my-pk", secret_key="my-sk", host="https://my-host.com"
        )
        mock_langfuse_module.Langfuse.assert_called_once_with(
            public_key="my-pk",
            secret_key="my-sk",
            host="https://my-host.com",
        )

    def test_raises_on_empty_public_key(self, mock_langfuse_module: MagicMock) -> None:
        with pytest.raises(ValueError, match="public_key"):
            LangfuseTracer(public_key="", secret_key=_SK, host=_HOST)

    def test_raises_on_empty_secret_key(self, mock_langfuse_module: MagicMock) -> None:
        with pytest.raises(ValueError, match="secret_key"):
            LangfuseTracer(public_key=_PK, secret_key="", host=_HOST)

    def test_raises_if_langfuse_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "langfuse", None)
        with pytest.raises(ImportError, match="langfuse"):
            LangfuseTracer(public_key=_PK, secret_key=_SK, host=_HOST)


# ---------------------------------------------------------------------------
# TestLangfuseTracerStartTrace
# ---------------------------------------------------------------------------


class TestLangfuseTracerStartTrace:
    def test_calls_langfuse_trace_with_name_and_query(
        self,
        tracer: LangfuseTracer,
        mock_client: MagicMock,
        mock_trace: MagicMock,
    ) -> None:
        tracer.start_trace("rag_query", query="quel document ?")
        mock_client.trace.assert_called_once_with(
            name="rag_query",
            input="quel document ?",
            metadata={},
        )

    def test_returns_trace_context(
        self, tracer: LangfuseTracer, mock_client: MagicMock, mock_trace: MagicMock
    ) -> None:
        ctx = tracer.start_trace("t", query="q")
        assert isinstance(ctx, TraceContext)

    def test_trace_id_is_langfuse_trace_id(
        self, tracer: LangfuseTracer, mock_client: MagicMock, mock_trace: MagicMock
    ) -> None:
        ctx = tracer.start_trace("t", query="q")
        assert ctx.trace_id == "trace-abc"

    def test_none_metadata_becomes_empty_dict(
        self, tracer: LangfuseTracer, mock_client: MagicMock, mock_trace: MagicMock
    ) -> None:
        tracer.start_trace("t", query="q", metadata=None)
        assert mock_client.trace.call_args.kwargs["metadata"] == {}

    def test_metadata_is_forwarded(
        self, tracer: LangfuseTracer, mock_client: MagicMock, mock_trace: MagicMock
    ) -> None:
        tracer.start_trace("t", query="q", metadata={"top_k": 5})
        assert mock_client.trace.call_args.kwargs["metadata"] == {"top_k": 5}


# ---------------------------------------------------------------------------
# TestLangfuseTracerStartSpan
# ---------------------------------------------------------------------------


class TestLangfuseTracerStartSpan:
    def test_calls_span_on_trace_handle(
        self,
        tracer: LangfuseTracer,
        mock_client: MagicMock,
        mock_trace: MagicMock,
        mock_span: MagicMock,
    ) -> None:
        ctx = tracer.start_trace("t", query="q")
        tracer.start_span(ctx, "embedding", span_input={"query": "q"})
        mock_trace.span.assert_called_once_with(name="embedding", input={"query": "q"})

    def test_maps_span_input_to_sdk_input_kwarg(
        self,
        tracer: LangfuseTracer,
        mock_client: MagicMock,
        mock_trace: MagicMock,
        mock_span: MagicMock,
    ) -> None:
        ctx = tracer.start_trace("t", query="q")
        tracer.start_span(ctx, "retrieval", span_input={"top_k": 5})
        assert mock_trace.span.call_args.kwargs["input"] == {"top_k": 5}

    def test_returns_span_id(
        self,
        tracer: LangfuseTracer,
        mock_client: MagicMock,
        mock_trace: MagicMock,
        mock_span: MagicMock,
    ) -> None:
        ctx = tracer.start_trace("t", query="q")
        span_id = tracer.start_span(ctx, "s", span_input={})
        assert span_id == "span-xyz"


# ---------------------------------------------------------------------------
# TestLangfuseTracerEndSpan
# ---------------------------------------------------------------------------


class TestLangfuseTracerEndSpan:
    def test_calls_end_on_span_with_output(
        self,
        tracer: LangfuseTracer,
        mock_client: MagicMock,
        mock_trace: MagicMock,
        mock_span: MagicMock,
    ) -> None:
        ctx = tracer.start_trace("t", query="q")
        span_id = tracer.start_span(ctx, "s", span_input={})
        tracer.end_span(ctx, span_id, output={"dim": 1024})
        mock_span.end.assert_called_once_with(output={"dim": 1024})

    def test_span_removed_from_internal_dict(
        self,
        tracer: LangfuseTracer,
        mock_client: MagicMock,
        mock_trace: MagicMock,
        mock_span: MagicMock,
    ) -> None:
        ctx = tracer.start_trace("t", query="q")
        span_id = tracer.start_span(ctx, "s", span_input={})
        tracer.end_span(ctx, span_id, output={})
        assert span_id not in tracer._spans

    def test_unknown_span_id_does_not_raise(
        self,
        tracer: LangfuseTracer,
        mock_client: MagicMock,
        mock_trace: MagicMock,
    ) -> None:
        ctx = tracer.start_trace("t", query="q")
        tracer.end_span(ctx, "nonexistent-span-id", output={})


# ---------------------------------------------------------------------------
# TestLangfuseTracerRecordGeneration
# ---------------------------------------------------------------------------


class TestLangfuseTracerRecordGeneration:
    def test_calls_generation_on_trace(
        self,
        tracer: LangfuseTracer,
        mock_client: MagicMock,
        mock_trace: MagicMock,
    ) -> None:
        ctx = tracer.start_trace("t", query="q")
        tracer.record_generation(
            ctx, model="llama3", prompt_tokens=100, completion_tokens=50
        )
        mock_trace.generation.assert_called_once()

    def test_model_and_usage_are_forwarded(
        self,
        tracer: LangfuseTracer,
        mock_client: MagicMock,
        mock_trace: MagicMock,
    ) -> None:
        ctx = tracer.start_trace("t", query="q")
        tracer.record_generation(
            ctx, model="llama3", prompt_tokens=100, completion_tokens=50
        )
        kwargs = mock_trace.generation.call_args.kwargs
        assert kwargs["model"] == "llama3"
        assert kwargs["usage"] == {"input": 100, "output": 50}

    def test_none_tokens_forwarded_as_none(
        self,
        tracer: LangfuseTracer,
        mock_client: MagicMock,
        mock_trace: MagicMock,
    ) -> None:
        ctx = tracer.start_trace("t", query="q")
        tracer.record_generation(
            ctx, model="llama3", prompt_tokens=None, completion_tokens=None
        )
        kwargs = mock_trace.generation.call_args.kwargs
        assert kwargs["usage"] == {"input": None, "output": None}


# ---------------------------------------------------------------------------
# TestLangfuseTracerScore
# ---------------------------------------------------------------------------


class TestLangfuseTracerScore:
    def test_calls_client_score_with_correct_args(
        self,
        tracer: LangfuseTracer,
        mock_client: MagicMock,
        mock_trace: MagicMock,
    ) -> None:
        ctx = tracer.start_trace("t", query="q")
        tracer.score(ctx, "faithfulness", 0.87)
        mock_client.score.assert_called_once_with(
            trace_id="trace-abc", name="faithfulness", value=0.87
        )

    def test_score_uses_ctx_trace_id(
        self,
        tracer: LangfuseTracer,
        mock_client: MagicMock,
        mock_trace: MagicMock,
    ) -> None:
        ctx = tracer.start_trace("t", query="q")
        tracer.score(ctx, "answer_relevancy", 0.91)
        assert mock_client.score.call_args.kwargs["trace_id"] == ctx.trace_id


# ---------------------------------------------------------------------------
# TestLangfuseTracerEndTrace
# ---------------------------------------------------------------------------


class TestLangfuseTracerEndTrace:
    def test_calls_trace_update_with_output(
        self,
        tracer: LangfuseTracer,
        mock_client: MagicMock,
        mock_trace: MagicMock,
    ) -> None:
        ctx = tracer.start_trace("t", query="q")
        tracer.end_trace(ctx, output="réponse finale")
        mock_trace.update.assert_called_once_with(output="réponse finale")

    def test_calls_flush(
        self,
        tracer: LangfuseTracer,
        mock_client: MagicMock,
        mock_trace: MagicMock,
    ) -> None:
        ctx = tracer.start_trace("t", query="q")
        tracer.end_trace(ctx, output="ans")
        mock_client.flush.assert_called_once()

    def test_trace_removed_from_internal_dict(
        self,
        tracer: LangfuseTracer,
        mock_client: MagicMock,
        mock_trace: MagicMock,
    ) -> None:
        ctx = tracer.start_trace("t", query="q")
        tracer.end_trace(ctx, output="ans")
        assert ctx.trace_id not in tracer._handles

    def test_unknown_trace_id_does_not_raise(
        self,
        tracer: LangfuseTracer,
        mock_client: MagicMock,
    ) -> None:
        ctx = TraceContext(trace_id="unknown-id")
        tracer.end_trace(ctx, output="ans")

    def test_flush_called_even_when_trace_id_unknown(
        self,
        tracer: LangfuseTracer,
        mock_client: MagicMock,
    ) -> None:
        ctx = TraceContext(trace_id="unknown-id")
        tracer.end_trace(ctx, output="ans")
        mock_client.flush.assert_called_once()
