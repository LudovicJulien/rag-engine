from __future__ import annotations

import json
from unittest.mock import patch

import typer
from typer.testing import CliRunner

from src.cli.commands import cmd_config
from src.pipeline.config import Settings

# Local app — __main__.py is wired in a later commit
_app = typer.Typer()
_app.command()(cmd_config)

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


def _settings_with_key(key: str = "sk-abcdefghijklmnop") -> Settings:
    return _settings(llm_api_key=key)


# ---------------------------------------------------------------------------
# cmd_config — exit codes
# ---------------------------------------------------------------------------


class TestCmdConfigExitCodes:
    def test_exits_0_always(self) -> None:
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            result = runner.invoke(_app, [])
        assert result.exit_code == 0

    def test_exits_0_with_api_key_set(self) -> None:
        with patch(
            "src.cli.commands.get_settings",
            return_value=_settings_with_key(),
        ):
            result = runner.invoke(_app, [])
        assert result.exit_code == 0

    def test_exits_0_with_json_output(self) -> None:
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            result = runner.invoke(_app, ["--output", "json"])
        assert result.exit_code == 0

    def test_get_settings_called_once(self) -> None:
        with patch(
            "src.cli.commands.get_settings", return_value=_settings()
        ) as mock_gs:
            runner.invoke(_app, [])
        mock_gs.assert_called_once()


# ---------------------------------------------------------------------------
# cmd_config — JSON output structure
# ---------------------------------------------------------------------------


class TestCmdConfigJsonOutput:
    def test_json_output_is_valid_json(self) -> None:
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            result = runner.invoke(_app, ["--output", "json"])
        assert result.exit_code == 0
        assert isinstance(json.loads(result.output), dict)

    def test_json_contains_qdrant_host(self) -> None:
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            result = runner.invoke(_app, ["--output", "json"])
        assert "qdrant_host" in json.loads(result.output)

    def test_json_contains_collection_name(self) -> None:
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            result = runner.invoke(_app, ["--output", "json"])
        assert "collection_name" in json.loads(result.output)

    def test_json_contains_llm_provider(self) -> None:
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            result = runner.invoke(_app, ["--output", "json"])
        assert "llm_provider" in json.loads(result.output)

    def test_json_contains_llm_api_key(self) -> None:
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            result = runner.invoke(_app, ["--output", "json"])
        assert "llm_api_key" in json.loads(result.output)

    def test_json_qdrant_host_value_matches_settings(self) -> None:
        with patch(
            "src.cli.commands.get_settings",
            return_value=_settings(qdrant_host="myhost"),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert json.loads(result.output)["qdrant_host"] == "myhost"

    def test_json_collection_name_value_matches_settings(self) -> None:
        with patch(
            "src.cli.commands.get_settings",
            return_value=_settings(collection_name="prod-col"),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert json.loads(result.output)["collection_name"] == "prod-col"

    def test_json_non_secret_fields_are_not_masked(self) -> None:
        with patch(
            "src.cli.commands.get_settings",
            return_value=_settings(qdrant_host="myhost"),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert json.loads(result.output)["qdrant_host"] == "myhost"


# ---------------------------------------------------------------------------
# cmd_config — secret masking
# ---------------------------------------------------------------------------


class TestCmdConfigSecretMasking:
    def test_json_llm_api_key_masked_when_long_key(self) -> None:
        with patch(
            "src.cli.commands.get_settings",
            return_value=_settings_with_key("sk-abcdefghijklmnop"),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert json.loads(result.output)["llm_api_key"].endswith("****")

    def test_json_llm_api_key_exposes_first_four_chars(self) -> None:
        with patch(
            "src.cli.commands.get_settings",
            return_value=_settings_with_key("sk-abcdefghijklmnop"),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert json.loads(result.output)["llm_api_key"] == "sk-a****"

    def test_json_llm_api_key_fully_masked_when_short(self) -> None:
        with patch(
            "src.cli.commands.get_settings",
            return_value=_settings_with_key("abc"),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert json.loads(result.output)["llm_api_key"] == "****"

    def test_json_llm_api_key_fully_masked_when_empty(self) -> None:
        with patch(
            "src.cli.commands.get_settings",
            return_value=_settings(llm_api_key=""),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert json.loads(result.output)["llm_api_key"] == "****"

    def test_json_raw_api_key_not_in_output(self) -> None:
        raw_key = "sk-abcdefghijklmnop"
        with patch(
            "src.cli.commands.get_settings",
            return_value=_settings_with_key(raw_key),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert raw_key not in result.output

    def test_json_llm_api_key_exactly_four_chars_is_fully_masked(self) -> None:
        with patch(
            "src.cli.commands.get_settings",
            return_value=_settings_with_key("abcd"),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert json.loads(result.output)["llm_api_key"] == "****"

    def test_json_llm_api_key_five_chars_exposes_first_four(self) -> None:
        with patch(
            "src.cli.commands.get_settings",
            return_value=_settings_with_key("abcde"),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert json.loads(result.output)["llm_api_key"] == "abcd****"
