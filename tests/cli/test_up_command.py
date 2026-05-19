from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

import src.cli.commands as commands
from src.cli.commands import cmd_up
from src.pipeline.config import Settings

# Local app — __main__.py is wired in a later commit
_app = typer.Typer()
_app.command()(cmd_up)

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


def _mock_pipeline() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# cmd_up — exit codes
# ---------------------------------------------------------------------------


class TestCmdUpExitCodes:
    def test_exits_0_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(commands, "_pipeline", _mock_pipeline())
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            result = runner.invoke(_app, [])
        assert result.exit_code == 0

    def test_exits_1_when_bm25_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(commands, "_pipeline", None)
        with (
            patch(
                "src.pipeline.rag_pipeline.RAGPipeline.build",
                side_effect=FileNotFoundError("bm25.pkl not found"),
            ),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, [])
        assert result.exit_code == 1

    def test_exits_2_when_qdrant_down(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(commands, "_pipeline", None)
        with (
            patch(
                "src.pipeline.rag_pipeline.RAGPipeline.build",
                side_effect=ConnectionError("Connection refused"),
            ),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, [])
        assert result.exit_code == 2

    def test_exits_2_on_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(commands, "_pipeline", None)
        with (
            patch(
                "src.pipeline.rag_pipeline.RAGPipeline.build",
                side_effect=RuntimeError("unexpected failure"),
            ),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, [])
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# cmd_up — error messages
# ---------------------------------------------------------------------------


class TestCmdUpErrorMessages:
    def test_file_not_found_error_includes_ingest_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(commands, "_pipeline", None)
        captured: list[str] = []
        with (
            patch(
                "src.pipeline.rag_pipeline.RAGPipeline.build",
                side_effect=FileNotFoundError("bm25.pkl not found"),
            ),
            patch("src.cli.commands.get_settings", return_value=_settings()),
            patch(
                "src.cli.display.error", side_effect=lambda msg: captured.append(msg)
            ),
        ):
            runner.invoke(_app, [])
        assert len(captured) == 1
        assert "ingest" in captured[0]

    def test_connection_error_message_does_not_include_ingest_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(commands, "_pipeline", None)
        captured: list[str] = []
        with (
            patch(
                "src.pipeline.rag_pipeline.RAGPipeline.build",
                side_effect=ConnectionError("refused"),
            ),
            patch("src.cli.commands.get_settings", return_value=_settings()),
            patch(
                "src.cli.display.error", side_effect=lambda msg: captured.append(msg)
            ),
        ):
            runner.invoke(_app, [])
        assert len(captured) == 1
        assert "ingest" not in captured[0]


# ---------------------------------------------------------------------------
# cmd_up — JSON output
# ---------------------------------------------------------------------------


class TestCmdUpJsonOutput:
    def test_json_output_is_valid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(commands, "_pipeline", _mock_pipeline())
        with (
            patch("src.cli.commands.get_settings", return_value=_settings()),
            patch("src.cli.display.BM25SparseEmbedder.load", side_effect=Exception),
            patch("src.cli.display.QdrantClient", side_effect=Exception),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)

    def test_json_output_status_is_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(commands, "_pipeline", _mock_pipeline())
        with (
            patch("src.cli.commands.get_settings", return_value=_settings()),
            patch("src.cli.display.BM25SparseEmbedder.load", side_effect=Exception),
            patch("src.cli.display.QdrantClient", side_effect=Exception),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        data = json.loads(result.output)
        assert data["status"] == "ready"

    def test_json_output_contains_dense_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(commands, "_pipeline", _mock_pipeline())
        with (
            patch("src.cli.commands.get_settings", return_value=_settings()),
            patch("src.cli.display.BM25SparseEmbedder.load", side_effect=Exception),
            patch("src.cli.display.QdrantClient", side_effect=Exception),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        data = json.loads(result.output)
        assert "dense_model" in data

    def test_json_output_contains_collection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(commands, "_pipeline", _mock_pipeline())
        with (
            patch("src.cli.commands.get_settings", return_value=_settings()),
            patch("src.cli.display.BM25SparseEmbedder.load", side_effect=Exception),
            patch("src.cli.display.QdrantClient", side_effect=Exception),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        data = json.loads(result.output)
        assert data["collection"] == "rag-test"

    def test_json_output_contains_llm_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(commands, "_pipeline", _mock_pipeline())
        with (
            patch("src.cli.commands.get_settings", return_value=_settings()),
            patch("src.cli.display.BM25SparseEmbedder.load", side_effect=Exception),
            patch("src.cli.display.QdrantClient", side_effect=Exception),
        ):
            result = runner.invoke(_app, ["--output", "json"])
        data = json.loads(result.output)
        assert data["llm_provider"] == "ollama"


# ---------------------------------------------------------------------------
# cmd_up — singleton
# ---------------------------------------------------------------------------


class TestCmdUpSingleton:
    def test_uses_existing_pipeline_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = _mock_pipeline()
        monkeypatch.setattr(commands, "_pipeline", mock)
        with (
            patch("src.cli.commands.get_settings", return_value=_settings()),
            patch("src.pipeline.rag_pipeline.RAGPipeline.build") as mock_build,
        ):
            runner.invoke(_app, [])
        mock_build.assert_not_called()

    def test_builds_pipeline_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(commands, "_pipeline", None)
        mock = _mock_pipeline()
        with (
            patch(
                "src.pipeline.rag_pipeline.RAGPipeline.build", return_value=mock
            ) as mock_build,
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            runner.invoke(_app, [])
        mock_build.assert_called_once()

    def test_singleton_stored_after_build(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(commands, "_pipeline", None)
        mock = _mock_pipeline()
        with (
            patch("src.pipeline.rag_pipeline.RAGPipeline.build", return_value=mock),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            runner.invoke(_app, [])
        assert commands._pipeline is mock
