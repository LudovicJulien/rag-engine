from __future__ import annotations

import json
from unittest.mock import patch

import typer
from typer.testing import CliRunner

from src.cli.commands import HealthStatus, ServiceHealth, cmd_health
from src.pipeline.config import Settings

# Local app — __main__.py is wired in a later commit
_app = typer.Typer()
_app.command()(cmd_health)

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


def _ok() -> ServiceHealth:
    return ServiceHealth(ok=True, detail="ok")


def _fail(detail: str = "fail") -> ServiceHealth:
    return ServiceHealth(ok=False, detail=detail)


def _all_ok() -> HealthStatus:
    return HealthStatus(qdrant=_ok(), llm=_ok())


def _qdrant_down() -> HealthStatus:
    return HealthStatus(qdrant=_fail("unreachable at localhost:6333"), llm=_ok())


def _llm_down() -> HealthStatus:
    return HealthStatus(qdrant=_ok(), llm=_fail("ollama unreachable"))


# ---------------------------------------------------------------------------
# cmd_health — exit codes
# ---------------------------------------------------------------------------


class TestCmdHealthExitCodes:
    def test_exits_0_when_all_services_ok(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=_all_ok()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, [])
        assert result.exit_code == 0

    def test_exits_2_when_qdrant_down(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=_qdrant_down()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, [])
        assert result.exit_code == 2

    def test_exits_2_when_llm_down(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=_llm_down()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, [])
        assert result.exit_code == 2

    def test_exits_2_when_both_services_down(self) -> None:
        status = HealthStatus(qdrant=_fail(), llm=_fail())
        with (
            patch("src.cli.commands.check_health", return_value=status),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, [])
        assert result.exit_code == 2

    def test_check_health_called_with_settings(self) -> None:
        settings = _settings(qdrant_host="myhost")
        with (
            patch("src.cli.commands.check_health", return_value=_all_ok()) as mock_ch,
            patch("src.cli.commands.get_settings", return_value=settings),
        ):
            runner.invoke(_app, [])
        mock_ch.assert_called_once_with(settings)


# ---------------------------------------------------------------------------
# cmd_health — JSON output
# ---------------------------------------------------------------------------


class TestCmdHealthJsonOutput:
    def test_json_output_is_valid_json(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=_all_ok()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)

    def test_json_contains_qdrant_key(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=_all_ok()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert "qdrant" in json.loads(result.output)

    def test_json_contains_llm_key(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=_all_ok()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert "llm" in json.loads(result.output)

    def test_json_contains_all_ok_key(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=_all_ok()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert "all_ok" in json.loads(result.output)

    def test_json_all_ok_true_when_all_services_healthy(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=_all_ok()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert json.loads(result.output)["all_ok"] is True

    def test_json_all_ok_false_when_qdrant_down(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=_qdrant_down()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert json.loads(result.output)["all_ok"] is False

    def test_json_does_not_contain_bm25_cache(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=_all_ok()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert "bm25_cache" not in json.loads(result.output)

    def test_json_does_not_contain_collection(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=_all_ok()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert "collection" not in json.loads(result.output)

    def test_json_qdrant_entry_has_ok_and_detail(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=_all_ok()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        qdrant = json.loads(result.output)["qdrant"]
        assert "ok" in qdrant
        assert "detail" in qdrant

    def test_json_exit_code_is_2_with_json_output_when_service_down(self) -> None:
        with (
            patch("src.cli.commands.check_health", return_value=_qdrant_down()),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert result.exit_code == 2
        assert json.loads(result.output)["all_ok"] is False
