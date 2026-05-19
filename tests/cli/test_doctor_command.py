from __future__ import annotations

import json
from unittest.mock import patch

import typer
from typer.testing import CliRunner

from src.cli.commands import (
    ConfigStatus,
    DiagnosticsReport,
    HealthStatus,
    ServiceHealth,
    cmd_doctor,
)
from src.pipeline.config import Settings

# Local app — __main__.py is wired in a later commit
_app = typer.Typer()
_app.command()(cmd_doctor)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = dict(
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="rag-test",
        llm_provider="ollama",
        llm_base_url="http://localhost:11434",
        llm_api_key="",
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _ok_svc(detail: str = "ok") -> ServiceHealth:
    return ServiceHealth(ok=True, detail=detail)


def _fail_svc(detail: str = "fail") -> ServiceHealth:
    return ServiceHealth(ok=False, detail=detail)


def _full_report(
    collection_count: int = 142,
    bm25_vocab_size: int = 4821,
) -> DiagnosticsReport:
    return DiagnosticsReport(
        health=HealthStatus(qdrant=_ok_svc(), llm=_ok_svc()),
        config=ConfigStatus(bm25_cache=_ok_svc(), collection=_ok_svc()),
        collection_count=collection_count,
        bm25_vocab_size=bm25_vocab_size,
    )


def _partial_failure() -> DiagnosticsReport:
    return DiagnosticsReport(
        health=HealthStatus(qdrant=_fail_svc("unreachable"), llm=_ok_svc()),
        config=ConfigStatus(bm25_cache=_ok_svc(), collection=_fail_svc("not found")),
        collection_count=None,
        bm25_vocab_size=None,
    )


# ---------------------------------------------------------------------------
# cmd_doctor — exit codes
# ---------------------------------------------------------------------------


class TestCmdDoctorExitCodes:
    def test_exits_0_when_all_checks_pass(self) -> None:
        with (
            patch("src.cli.commands.build_diagnostics", return_value=_full_report()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, [])
        assert result.exit_code == 0

    def test_exits_2_when_qdrant_down(self) -> None:
        with (
            patch(
                "src.cli.commands.build_diagnostics", return_value=_partial_failure()
            ),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, [])
        assert result.exit_code == 2

    def test_exits_2_when_bm25_cache_missing(self) -> None:
        report = DiagnosticsReport(
            health=HealthStatus(qdrant=_ok_svc(), llm=_ok_svc()),
            config=ConfigStatus(
                bm25_cache=_fail_svc("file missing"), collection=_ok_svc()
            ),
            collection_count=10,
            bm25_vocab_size=None,
        )
        with (
            patch("src.cli.commands.build_diagnostics", return_value=report),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, [])
        assert result.exit_code == 2

    def test_exits_2_when_llm_down(self) -> None:
        report = DiagnosticsReport(
            health=HealthStatus(qdrant=_ok_svc(), llm=_fail_svc("key missing")),
            config=ConfigStatus(bm25_cache=_ok_svc(), collection=_ok_svc()),
            collection_count=10,
            bm25_vocab_size=500,
        )
        with (
            patch("src.cli.commands.build_diagnostics", return_value=report),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, [])
        assert result.exit_code == 2

    def test_build_diagnostics_called_with_settings(self) -> None:
        settings = _settings(collection_name="prod-col")
        with (
            patch(
                "src.cli.commands.build_diagnostics", return_value=_full_report()
            ) as mock_bd,
            patch("src.cli.commands.get_settings", return_value=settings),
        ):
            runner.invoke(_app, [])
        mock_bd.assert_called_once_with(settings)


# ---------------------------------------------------------------------------
# cmd_doctor — JSON output
# ---------------------------------------------------------------------------


class TestCmdDoctorJsonOutput:
    def test_json_output_is_valid_json(self) -> None:
        with (
            patch("src.cli.commands.build_diagnostics", return_value=_full_report()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert result.exit_code == 0
        assert isinstance(json.loads(result.output), dict)

    def test_json_contains_collection_count(self) -> None:
        with (
            patch("src.cli.commands.build_diagnostics", return_value=_full_report()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert "collection_count" in json.loads(result.output)

    def test_json_collection_count_value_matches_report(self) -> None:
        with (
            patch(
                "src.cli.commands.build_diagnostics",
                return_value=_full_report(collection_count=142),
            ),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert json.loads(result.output)["collection_count"] == 142

    def test_json_contains_bm25_vocab_size(self) -> None:
        with (
            patch("src.cli.commands.build_diagnostics", return_value=_full_report()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert "bm25_vocab_size" in json.loads(result.output)

    def test_json_bm25_vocab_size_value_matches_report(self) -> None:
        with (
            patch(
                "src.cli.commands.build_diagnostics",
                return_value=_full_report(bm25_vocab_size=4821),
            ),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert json.loads(result.output)["bm25_vocab_size"] == 4821

    def test_json_contains_health_section(self) -> None:
        with (
            patch("src.cli.commands.build_diagnostics", return_value=_full_report()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert "health" in json.loads(result.output)

    def test_json_contains_config_section(self) -> None:
        with (
            patch("src.cli.commands.build_diagnostics", return_value=_full_report()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert "config" in json.loads(result.output)

    def test_json_all_ok_true_when_all_pass(self) -> None:
        with (
            patch("src.cli.commands.build_diagnostics", return_value=_full_report()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert json.loads(result.output)["all_ok"] is True

    def test_json_all_ok_false_on_partial_failure(self) -> None:
        with (
            patch(
                "src.cli.commands.build_diagnostics", return_value=_partial_failure()
            ),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert json.loads(result.output)["all_ok"] is False

    def test_json_collection_count_none_when_qdrant_unreachable(self) -> None:
        with (
            patch(
                "src.cli.commands.build_diagnostics", return_value=_partial_failure()
            ),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert json.loads(result.output)["collection_count"] is None

    def test_json_exit_code_2_with_json_output_on_failure(self) -> None:
        with (
            patch(
                "src.cli.commands.build_diagnostics", return_value=_partial_failure()
            ),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert result.exit_code == 2
        assert json.loads(result.output)["all_ok"] is False
