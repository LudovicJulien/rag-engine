from __future__ import annotations

import pytest
from typer.testing import CliRunner

from src.cli.__main__ import app

runner = CliRunner()

_ALL_COMMANDS = ["ingest", "query", "up", "health", "doctor", "config"]


# ---------------------------------------------------------------------------
# App-level configuration
# ---------------------------------------------------------------------------


class TestAppConfiguration:
    def test_no_args_shows_help_content(self) -> None:
        # no_args_is_help=True — help text is printed even when no subcommand given
        result = runner.invoke(app, [])
        assert any(cmd in result.output for cmd in _ALL_COMMANDS)

    def test_help_flag_exits_0(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_app_name_in_help_output(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "rag-engine" in result.output

    def test_help_output_is_non_empty(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert len(result.output.strip()) > 0

    def test_no_install_completion_in_help(self) -> None:
        # add_completion=False — completion commands must not appear
        result = runner.invoke(app, ["--help"])
        assert "--install-completion" not in result.output


# ---------------------------------------------------------------------------
# Command registration — all 6 commands visible in top-level help
# ---------------------------------------------------------------------------


class TestCommandRegistration:
    @pytest.mark.parametrize("command", _ALL_COMMANDS)
    def test_command_appears_in_help(self, command: str) -> None:
        result = runner.invoke(app, ["--help"])
        assert command in result.output

    def test_all_six_commands_registered(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert all(cmd in result.output for cmd in _ALL_COMMANDS)


# ---------------------------------------------------------------------------
# Subcommand help — every wired command accepts --help without raising
# ---------------------------------------------------------------------------


class TestSubcommandHelp:
    @pytest.mark.parametrize("command", _ALL_COMMANDS)
    def test_subcommand_help_exits_0(self, command: str) -> None:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0

    @pytest.mark.parametrize("command", _ALL_COMMANDS)
    def test_subcommand_help_output_non_empty(self, command: str) -> None:
        result = runner.invoke(app, [command, "--help"])
        assert len(result.output.strip()) > 0


# ---------------------------------------------------------------------------
# Command options — verify key flags are wired through from the functions
# ---------------------------------------------------------------------------


class TestCommandOptions:
    def test_ingest_exposes_reset_flag(self) -> None:
        result = runner.invoke(app, ["ingest", "--help"])
        assert "--reset" in result.output

    def test_ingest_exposes_output_flag(self) -> None:
        result = runner.invoke(app, ["ingest", "--help"])
        assert "--output" in result.output

    def test_query_exposes_top_k_flag(self) -> None:
        result = runner.invoke(app, ["query", "--help"])
        assert "--top-k" in result.output

    def test_query_exposes_score_threshold_flag(self) -> None:
        result = runner.invoke(app, ["query", "--help"])
        assert "--score-threshold" in result.output

    def test_health_exposes_output_flag(self) -> None:
        result = runner.invoke(app, ["health", "--help"])
        assert "--output" in result.output

    def test_doctor_exposes_output_flag(self) -> None:
        result = runner.invoke(app, ["doctor", "--help"])
        assert "--output" in result.output

    def test_config_exposes_output_flag(self) -> None:
        result = runner.invoke(app, ["config", "--help"])
        assert "--output" in result.output
