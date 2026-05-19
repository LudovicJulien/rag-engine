from __future__ import annotations

import json
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from src.cli import display
from src.cli.commands import (
    ConfigStatus,
    DiagnosticsReport,
    HealthStatus,
    OutputFormat,
    ServiceHealth,
)
from src.cli.display import (
    _diagnostics_dict,
    _health_dict,
    _mask,
    _masked_settings,
    _svc_line,
    show_config,
    show_doctor,
    show_health,
    show_ingest_result,
    show_rag_result,
    show_up_result,
)
from src.pipeline.config import Settings
from src.pipeline.ingest_pipeline import IngestResult
from src.pipeline.rag_pipeline import RAGResult

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def rich_out(monkeypatch: pytest.MonkeyPatch) -> StringIO:
    """Replace display.console with a no-color, fixed-width StringIO console."""
    buf = StringIO()
    monkeypatch.setattr(
        display,
        "console",
        Console(file=buf, width=120, no_color=True, highlight=False),
    )
    return buf


def _ok(detail: str = "ok") -> ServiceHealth:
    return ServiceHealth(ok=True, detail=detail)


def _fail(detail: str = "fail") -> ServiceHealth:
    return ServiceHealth(ok=False, detail=detail)


def _health(qdrant_ok: bool = True, llm_ok: bool = True) -> HealthStatus:
    return HealthStatus(
        qdrant=_ok("connected") if qdrant_ok else _fail("unreachable"),
        llm=_ok("reachable") if llm_ok else _fail("down"),
    )


def _config(bm25_ok: bool = True, coll_ok: bool = True) -> ConfigStatus:
    return ConfigStatus(
        bm25_cache=_ok("file found") if bm25_ok else _fail("file missing"),
        collection=_ok("exists") if coll_ok else _fail("not found"),
    )


def _report(
    health_ok: bool = True,
    config_ok: bool = True,
    collection_count: int | None = 42,
    bm25_vocab: int | None = 100,
) -> DiagnosticsReport:
    return DiagnosticsReport(
        health=_health(qdrant_ok=health_ok, llm_ok=health_ok),
        config=_config(bm25_ok=config_ok, coll_ok=config_ok),
        collection_count=collection_count,
        bm25_vocab_size=bm25_vocab,
    )


def _rag_result(**kwargs: object) -> RAGResult:
    defaults: dict[str, object] = dict(
        query="What is RAG?",
        answer="RAG stands for Retrieval-Augmented Generation.",
        sources=["chunk-001", "chunk-002"],
        detected_language="en",
        model="gemma3:4b",
        chunks_retrieved=2,
        tokens_used=128,
    )
    defaults.update(kwargs)
    return RAGResult(**defaults)  # type: ignore[arg-type]


def _ingest_result(**kwargs: object) -> IngestResult:
    defaults: dict[str, object] = dict(
        source="/data/chunks.json",
        collection_name="rag-collection",
        chunks_loaded=10,
        chunks_upserted=10,
        chunks_failed=0,
        bm25_cache_path="/tmp/bm25.pkl",
    )
    defaults.update(kwargs)
    return IngestResult(**defaults)  # type: ignore[arg-type]


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = dict(
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="rag-collection",
        llm_provider="ollama",
        llm_base_url="http://localhost:11434",
        llm_api_key="",
        embedding_model="intfloat/multilingual-e5-large",
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestMask
# ---------------------------------------------------------------------------


class TestMask:
    def test_keeps_first_four_chars(self) -> None:
        assert _mask("sk-abcdefgh") == "sk-a****"

    def test_short_value_returns_stars_only(self) -> None:
        assert _mask("abc") == "****"

    def test_exactly_four_chars_returns_stars_only(self) -> None:
        assert _mask("abcd") == "****"

    def test_five_chars_keeps_first_four(self) -> None:
        assert _mask("abcde") == "abcd****"

    def test_empty_string_returns_stars_only(self) -> None:
        assert _mask("") == "****"

    def test_suffix_is_always_four_stars(self) -> None:
        assert _mask("sk-ant-very-long-key").endswith("****")

    def test_prefix_is_exactly_four_chars(self) -> None:
        result = _mask("sk-ant-very-long-key")
        assert result.startswith("sk-a")
        assert len(result) == 8


# ---------------------------------------------------------------------------
# TestSvcLine
# ---------------------------------------------------------------------------


class TestSvcLine:
    def test_contains_label(self) -> None:
        assert "Qdrant" in _svc_line("Qdrant", _ok("connected"))

    def test_contains_detail(self) -> None:
        assert "connected" in _svc_line("Qdrant", _ok("connected"))

    def test_ok_service_contains_checkmark(self) -> None:
        line = _svc_line("LLM", _ok())
        assert "✓" in line

    def test_failed_service_contains_cross(self) -> None:
        line = _svc_line("LLM", _fail())
        assert "✗" in line

    def test_label_is_left_padded_for_alignment(self) -> None:
        short = _svc_line("LLM", _ok())
        long = _svc_line("BM25 cache", _ok())
        # Both have the same start position for the icon
        icon_pos_short = short.index("[")
        icon_pos_long = long.index("[")
        assert icon_pos_short == icon_pos_long


# ---------------------------------------------------------------------------
# TestHealthDict
# ---------------------------------------------------------------------------


class TestHealthDict:
    def test_contains_qdrant_key(self) -> None:
        assert "qdrant" in _health_dict(_health())

    def test_contains_llm_key(self) -> None:
        assert "llm" in _health_dict(_health())

    def test_contains_all_ok(self) -> None:
        assert "all_ok" in _health_dict(_health())

    def test_all_ok_true_when_healthy(self) -> None:
        assert _health_dict(_health())["all_ok"] is True

    def test_all_ok_false_when_qdrant_down(self) -> None:
        assert _health_dict(_health(qdrant_ok=False))["all_ok"] is False

    def test_does_not_contain_bm25_cache(self) -> None:
        assert "bm25_cache" not in _health_dict(_health())


# ---------------------------------------------------------------------------
# TestDiagnosticsDict
# ---------------------------------------------------------------------------


class TestDiagnosticsDict:
    def test_contains_health_health_and_config_keys(self) -> None:
        d = _diagnostics_dict(_report())
        assert "health" in d
        assert "config" in d

    def test_contains_statistics_keys(self) -> None:
        d = _diagnostics_dict(_report())
        assert "collection_count" in d
        assert "bm25_vocab_size" in d

    def test_contains_all_ok(self) -> None:
        assert "all_ok" in _diagnostics_dict(_report())

    def test_nested_health_has_all_ok(self) -> None:
        assert "all_ok" in _diagnostics_dict(_report())["health"]

    def test_nested_config_has_all_ok(self) -> None:
        assert "all_ok" in _diagnostics_dict(_report())["config"]

    def test_none_values_serialise_to_null(self) -> None:
        d = _diagnostics_dict(_report(collection_count=None, bm25_vocab=None))
        assert d["collection_count"] is None
        assert d["bm25_vocab_size"] is None


# ---------------------------------------------------------------------------
# TestShowHealth
# ---------------------------------------------------------------------------


class TestShowHealth:
    def test_text_contains_qdrant(self, rich_out: StringIO) -> None:
        show_health(_health(), OutputFormat.text)
        assert "Qdrant" in rich_out.getvalue()

    def test_text_contains_llm(self, rich_out: StringIO) -> None:
        show_health(_health(), OutputFormat.text)
        assert "LLM" in rich_out.getvalue()

    def test_text_contains_detail(self, rich_out: StringIO) -> None:
        show_health(_health(), OutputFormat.text)
        assert "connected" in rich_out.getvalue()

    def test_json_is_parseable(self, capsys: pytest.CaptureFixture[str]) -> None:
        show_health(_health(), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, dict)

    def test_json_contains_qdrant_and_llm(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        show_health(_health(), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert "qdrant" in data
        assert "llm" in data

    def test_json_contains_all_ok(self, capsys: pytest.CaptureFixture[str]) -> None:
        show_health(_health(), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert "all_ok" in data

    def test_json_does_not_contain_bm25_cache(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        show_health(_health(), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert "bm25_cache" not in data

    def test_json_all_ok_true_when_healthy(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        show_health(_health(), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert data["all_ok"] is True

    def test_json_all_ok_false_when_qdrant_down(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        show_health(_health(qdrant_ok=False), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert data["all_ok"] is False


# ---------------------------------------------------------------------------
# TestShowIngestResult
# ---------------------------------------------------------------------------


class TestShowIngestResult:
    def test_text_contains_filename(self, rich_out: StringIO) -> None:
        show_ingest_result(_ingest_result(), OutputFormat.text)
        assert "chunks.json" in rich_out.getvalue()

    def test_text_contains_loaded_count(self, rich_out: StringIO) -> None:
        show_ingest_result(_ingest_result(chunks_loaded=42), OutputFormat.text)
        assert "42" in rich_out.getvalue()

    def test_text_contains_bm25_path(self, rich_out: StringIO) -> None:
        show_ingest_result(
            _ingest_result(bm25_cache_path="/tmp/bm25.pkl"), OutputFormat.text
        )
        assert "/tmp/bm25.pkl" in rich_out.getvalue()

    def test_json_is_parseable(self, capsys: pytest.CaptureFixture[str]) -> None:
        show_ingest_result(_ingest_result(), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, dict)

    def test_json_contains_chunks_upserted(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        show_ingest_result(_ingest_result(chunks_upserted=10), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert data["chunks_upserted"] == 10

    def test_json_contains_chunks_failed(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        show_ingest_result(_ingest_result(chunks_failed=2), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert data["chunks_failed"] == 2


# ---------------------------------------------------------------------------
# TestShowRagResult
# ---------------------------------------------------------------------------


class TestShowRagResult:
    def test_text_contains_answer(self, rich_out: StringIO) -> None:
        result = _rag_result(answer="RAG is great.")
        show_rag_result(result, OutputFormat.text)
        assert "RAG is great." in rich_out.getvalue()

    def test_text_contains_source(self, rich_out: StringIO) -> None:
        result = _rag_result(sources=["chunk-042"])
        show_rag_result(result, OutputFormat.text)
        assert "chunk-042" in rich_out.getvalue()

    def test_text_contains_model_name(self, rich_out: StringIO) -> None:
        result = _rag_result(model="gemma3:4b")
        show_rag_result(result, OutputFormat.text)
        assert "gemma3:4b" in rich_out.getvalue()

    def test_json_is_parseable(self, capsys: pytest.CaptureFixture[str]) -> None:
        show_rag_result(_rag_result(), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, dict)

    def test_json_contains_answer(self, capsys: pytest.CaptureFixture[str]) -> None:
        show_rag_result(_rag_result(answer="hello"), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert data["answer"] == "hello"

    def test_json_tokens_used_none_serialises_to_null(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        show_rag_result(_rag_result(tokens_used=None), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert data["tokens_used"] is None

    def test_text_empty_sources_shows_placeholder(self, rich_out: StringIO) -> None:
        show_rag_result(_rag_result(sources=[]), OutputFormat.text)
        assert "—" in rich_out.getvalue()


# ---------------------------------------------------------------------------
# TestShowDoctor
# ---------------------------------------------------------------------------


class TestShowDoctor:
    def test_text_contains_connectivity_section(self, rich_out: StringIO) -> None:
        show_doctor(_report(), OutputFormat.text)
        assert "Connectivity" in rich_out.getvalue()

    def test_text_contains_configuration_section(self, rich_out: StringIO) -> None:
        show_doctor(_report(), OutputFormat.text)
        assert "Configuration" in rich_out.getvalue()

    def test_text_contains_statistics_section(self, rich_out: StringIO) -> None:
        show_doctor(_report(), OutputFormat.text)
        assert "Statistics" in rich_out.getvalue()

    def test_text_contains_collection_count(self, rich_out: StringIO) -> None:
        show_doctor(_report(collection_count=142), OutputFormat.text)
        assert "142" in rich_out.getvalue()

    def test_text_contains_bm25_vocab(self, rich_out: StringIO) -> None:
        show_doctor(_report(bm25_vocab=4821), OutputFormat.text)
        assert "4821" in rich_out.getvalue()

    def test_text_all_ok_shows_passed_message(self, rich_out: StringIO) -> None:
        show_doctor(_report(), OutputFormat.text)
        assert "All checks passed" in rich_out.getvalue()

    def test_text_failure_shows_failed_message(self, rich_out: StringIO) -> None:
        show_doctor(_report(health_ok=False), OutputFormat.text)
        assert "failed" in rich_out.getvalue()

    def test_text_none_count_shows_unavailable(self, rich_out: StringIO) -> None:
        show_doctor(_report(collection_count=None), OutputFormat.text)
        assert "unavailable" in rich_out.getvalue()

    def test_json_is_parseable(self, capsys: pytest.CaptureFixture[str]) -> None:
        show_doctor(_report(), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, dict)

    def test_json_contains_all_ok(self, capsys: pytest.CaptureFixture[str]) -> None:
        show_doctor(_report(), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert "all_ok" in data

    def test_json_contains_stats(self, capsys: pytest.CaptureFixture[str]) -> None:
        show_doctor(_report(collection_count=42, bm25_vocab=100), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert data["collection_count"] == 42
        assert data["bm25_vocab_size"] == 100


# ---------------------------------------------------------------------------
# TestShowConfig
# ---------------------------------------------------------------------------


class TestShowConfig:
    def test_text_contains_llm_model(self, rich_out: StringIO) -> None:
        show_config(_settings(llm_model="gemma3:4b"), OutputFormat.text)
        assert "gemma3:4b" in rich_out.getvalue()

    def test_text_masks_llm_api_key(self, rich_out: StringIO) -> None:
        show_config(_settings(llm_api_key="sk-ant-very-secret"), OutputFormat.text)
        output = rich_out.getvalue()
        assert "sk-ant-very-secret" not in output
        assert "****" in output

    def test_text_empty_key_shows_stars_only(self, rich_out: StringIO) -> None:
        show_config(_settings(llm_api_key=""), OutputFormat.text)
        assert "****" in rich_out.getvalue()

    def test_json_is_parseable(self, capsys: pytest.CaptureFixture[str]) -> None:
        show_config(_settings(), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, dict)

    def test_json_masks_llm_api_key(self, capsys: pytest.CaptureFixture[str]) -> None:
        show_config(_settings(llm_api_key="sk-ant-very-secret"), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert data["llm_api_key"].endswith("****")
        assert "sk-ant-very-secret" not in data["llm_api_key"]

    def test_json_non_secret_field_not_masked(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        show_config(_settings(llm_model="gemma3:4b"), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert data["llm_model"] == "gemma3:4b"


# ---------------------------------------------------------------------------
# TestShowUpResult
# ---------------------------------------------------------------------------


class TestShowUpResult:
    def _mock_pipeline(self) -> MagicMock:
        return MagicMock()

    def test_text_contains_pipeline_ready(self, rich_out: StringIO) -> None:
        with (
            patch("src.cli.display.BM25SparseEmbedder.load", side_effect=Exception),
            patch("src.cli.display.QdrantClient", side_effect=Exception),
        ):
            show_up_result(self._mock_pipeline(), _settings(), OutputFormat.text)
        assert "Pipeline ready" in rich_out.getvalue()

    def test_text_contains_embedding_model(self, rich_out: StringIO) -> None:
        with (
            patch("src.cli.display.BM25SparseEmbedder.load", side_effect=Exception),
            patch("src.cli.display.QdrantClient", side_effect=Exception),
        ):
            show_up_result(
                self._mock_pipeline(),
                _settings(embedding_model="intfloat/multilingual-e5-large"),
                OutputFormat.text,
            )
        assert "intfloat/multilingual-e5-large" in rich_out.getvalue()

    def test_text_contains_vocab_size_when_available(self, rich_out: StringIO) -> None:
        mock_bm25 = MagicMock()
        mock_bm25.embedding_dim = 4821
        with (
            patch("src.cli.display.BM25SparseEmbedder.load", return_value=mock_bm25),
            patch("src.cli.display.QdrantClient", side_effect=Exception),
        ):
            show_up_result(self._mock_pipeline(), _settings(), OutputFormat.text)
        assert "4821" in rich_out.getvalue()

    def test_text_shows_without_vocab_when_bm25_unavailable(
        self, rich_out: StringIO
    ) -> None:
        with (
            patch("src.cli.display.BM25SparseEmbedder.load", side_effect=Exception),
            patch("src.cli.display.QdrantClient", side_effect=Exception),
        ):
            show_up_result(self._mock_pipeline(), _settings(), OutputFormat.text)
        assert "Pipeline ready" in rich_out.getvalue()

    def test_json_is_parseable(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch("src.cli.display.BM25SparseEmbedder.load", side_effect=Exception),
            patch("src.cli.display.QdrantClient", side_effect=Exception),
        ):
            show_up_result(self._mock_pipeline(), _settings(), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, dict)

    def test_json_status_is_ready(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch("src.cli.display.BM25SparseEmbedder.load", side_effect=Exception),
            patch("src.cli.display.QdrantClient", side_effect=Exception),
        ):
            show_up_result(self._mock_pipeline(), _settings(), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "ready"

    def test_json_contains_collection_count_when_qdrant_accessible(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with (
            patch("src.cli.display.BM25SparseEmbedder.load", side_effect=Exception),
            patch("src.cli.display.QdrantClient") as mock_cls,
        ):
            mock_cls.return_value.count.return_value = MagicMock(count=99)
            show_up_result(self._mock_pipeline(), _settings(), OutputFormat.json)
        data = json.loads(capsys.readouterr().out)
        assert data["collection_count"] == 99


# ---------------------------------------------------------------------------
# TestMaskedSettings
# ---------------------------------------------------------------------------


class TestMaskedSettings:
    def test_llm_api_key_is_masked(self) -> None:
        settings = _settings(llm_api_key="sk-ant-very-secret")
        masked = _masked_settings(settings)
        assert masked["llm_api_key"].endswith("****")
        assert "sk-ant-very-secret" not in masked["llm_api_key"]

    def test_non_secret_field_is_not_masked(self) -> None:
        settings = _settings(llm_model="gemma3:4b")
        masked = _masked_settings(settings)
        assert masked["llm_model"] == "gemma3:4b"

    def test_empty_api_key_becomes_stars_only(self) -> None:
        settings = _settings(llm_api_key="")
        masked = _masked_settings(settings)
        assert masked["llm_api_key"] == "****"

    def test_all_settings_fields_present(self) -> None:
        settings = _settings()
        masked = _masked_settings(settings)
        assert "qdrant_host" in masked
        assert "llm_model" in masked
        assert "bm25_cache_path" in masked
