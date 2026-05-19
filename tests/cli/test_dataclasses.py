from __future__ import annotations

import dataclasses

import pytest

from src.cli.commands import (
    ConfigStatus,
    DiagnosticsReport,
    HealthStatus,
    OutputFormat,
    ServiceHealth,
)


def _ok(detail: str = "connected") -> ServiceHealth:
    return ServiceHealth(ok=True, detail=detail)


def _fail(detail: str = "unreachable") -> ServiceHealth:
    return ServiceHealth(ok=False, detail=detail)


class TestOutputFormat:
    def test_text_value_is_text_string(self) -> None:
        assert OutputFormat.text == "text"

    def test_json_value_is_json_string(self) -> None:
        assert OutputFormat.json == "json"

    def test_is_str_subclass_for_typer_compatibility(self) -> None:
        assert isinstance(OutputFormat.text, str)

    def test_has_exactly_two_members(self) -> None:
        assert len(OutputFormat) == 2


class TestServiceHealth:
    def test_ok_true_stored(self) -> None:
        s = ServiceHealth(ok=True, detail="connected")
        assert s.ok is True

    def test_ok_false_stored(self) -> None:
        s = ServiceHealth(ok=False, detail="unreachable at localhost:6333")
        assert s.ok is False

    def test_detail_stored_verbatim(self) -> None:
        s = ServiceHealth(ok=True, detail="file found at /tmp/bm25.pkl")
        assert s.detail == "file found at /tmp/bm25.pkl"

    def test_is_frozen(self) -> None:
        s = ServiceHealth(ok=True, detail="connected")
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.ok = False  # type: ignore[misc]


class TestHealthStatus:
    def test_all_ok_true_when_both_services_healthy(self) -> None:
        assert HealthStatus(qdrant=_ok(), llm=_ok()).all_ok is True

    def test_all_ok_false_when_qdrant_down(self) -> None:
        assert HealthStatus(qdrant=_fail(), llm=_ok()).all_ok is False

    def test_all_ok_false_when_llm_down(self) -> None:
        assert HealthStatus(qdrant=_ok(), llm=_fail()).all_ok is False

    def test_all_ok_false_when_both_down(self) -> None:
        assert HealthStatus(qdrant=_fail(), llm=_fail()).all_ok is False

    def test_qdrant_field_stored(self) -> None:
        qdrant = _ok("connected at localhost:6333")
        status = HealthStatus(qdrant=qdrant, llm=_ok())
        assert status.qdrant is qdrant

    def test_llm_field_stored(self) -> None:
        llm = _ok("ollama reachable at localhost:11434")
        status = HealthStatus(qdrant=_ok(), llm=llm)
        assert status.llm is llm

    def test_is_frozen(self) -> None:
        status = HealthStatus(qdrant=_ok(), llm=_ok())
        with pytest.raises(dataclasses.FrozenInstanceError):
            status.qdrant = _fail()  # type: ignore[misc]


class TestConfigStatus:
    def test_all_ok_true_when_both_checks_pass(self) -> None:
        config = ConfigStatus(bm25_cache=_ok("file found"), collection=_ok("exists"))
        assert config.all_ok is True

    def test_all_ok_false_when_bm25_cache_missing(self) -> None:
        config = ConfigStatus(bm25_cache=_fail("file missing"), collection=_ok())
        assert config.all_ok is False

    def test_all_ok_false_when_collection_missing(self) -> None:
        config = ConfigStatus(bm25_cache=_ok(), collection=_fail("not found"))
        assert config.all_ok is False

    def test_all_ok_false_when_both_missing(self) -> None:
        config = ConfigStatus(bm25_cache=_fail(), collection=_fail())
        assert config.all_ok is False

    def test_bm25_cache_field_stored(self) -> None:
        svc = _ok("file found at /tmp/bm25.pkl")
        config = ConfigStatus(bm25_cache=svc, collection=_ok())
        assert config.bm25_cache is svc

    def test_is_frozen(self) -> None:
        config = ConfigStatus(bm25_cache=_ok(), collection=_ok())
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.bm25_cache = _fail()  # type: ignore[misc]


class TestDiagnosticsReport:
    def _make(
        self,
        *,
        config_ok: bool = True,
        health_ok: bool = True,
        collection_count: int | None = 10,
        bm25_vocab_size: int | None = 500,
    ) -> DiagnosticsReport:
        svc = _ok() if health_ok else _fail()
        cfg = _ok() if config_ok else _fail()
        return DiagnosticsReport(
            config=ConfigStatus(bm25_cache=cfg, collection=cfg),
            health=HealthStatus(qdrant=svc, llm=svc),
            collection_count=collection_count,
            bm25_vocab_size=bm25_vocab_size,
        )

    def test_all_ok_true_when_config_and_health_pass(self) -> None:
        assert self._make().all_ok is True

    def test_all_ok_false_when_config_fails(self) -> None:
        assert self._make(config_ok=False).all_ok is False

    def test_all_ok_false_when_health_fails(self) -> None:
        assert self._make(health_ok=False).all_ok is False

    def test_all_ok_false_when_both_fail(self) -> None:
        assert self._make(config_ok=False, health_ok=False).all_ok is False

    def test_collection_count_stored(self) -> None:
        assert self._make(collection_count=142).collection_count == 142

    def test_bm25_vocab_size_stored(self) -> None:
        assert self._make(bm25_vocab_size=4821).bm25_vocab_size == 4821

    def test_collection_count_accepts_none(self) -> None:
        assert self._make(collection_count=None).collection_count is None

    def test_bm25_vocab_size_accepts_none(self) -> None:
        assert self._make(bm25_vocab_size=None).bm25_vocab_size is None

    def test_is_frozen(self) -> None:
        report = self._make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.collection_count = 0  # type: ignore[misc]
