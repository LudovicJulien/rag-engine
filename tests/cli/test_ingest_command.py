from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

import src.cli.commands as commands
from src.cli.commands import cmd_ingest
from src.pipeline.config import Settings
from src.pipeline.ingest_pipeline import IngestResult

# Local app — __main__.py is wired in a later commit
_app = typer.Typer()
_app.command()(cmd_ingest)

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


def _ok_result(source: str = "chunks.json") -> IngestResult:
    return IngestResult(
        source=source,
        collection_name="rag-test",
        chunks_loaded=10,
        chunks_upserted=10,
        chunks_failed=0,
        bm25_cache_path="/tmp/bm25/chunks.pkl",
    )


# ---------------------------------------------------------------------------
# _get_or_build_pipeline
# ---------------------------------------------------------------------------


class TestGetOrBuildPipeline:
    def test_builds_pipeline_when_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(commands, "_pipeline", None)
        mock_pipeline = MagicMock()
        with patch(
            "src.pipeline.rag_pipeline.RAGPipeline.build",
            return_value=mock_pipeline,
        ):
            result = commands._get_or_build_pipeline(_settings())
        assert result is mock_pipeline

    def test_returns_cached_pipeline_on_second_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(commands, "_pipeline", None)
        mock_pipeline = MagicMock()
        with patch(
            "src.pipeline.rag_pipeline.RAGPipeline.build",
            return_value=mock_pipeline,
        ) as mock_build:
            commands._get_or_build_pipeline(_settings())
            commands._get_or_build_pipeline(_settings())
        mock_build.assert_called_once()

    def test_skips_build_when_pipeline_already_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        existing = MagicMock()
        monkeypatch.setattr(commands, "_pipeline", existing)
        with patch("src.pipeline.rag_pipeline.RAGPipeline.build") as mock_build:
            result = commands._get_or_build_pipeline(_settings())
        mock_build.assert_not_called()
        assert result is existing

    def test_stores_built_pipeline_as_singleton(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(commands, "_pipeline", None)
        mock_pipeline = MagicMock()
        with patch(
            "src.pipeline.rag_pipeline.RAGPipeline.build",
            return_value=mock_pipeline,
        ):
            commands._get_or_build_pipeline(_settings())
        assert commands._pipeline is mock_pipeline


# ---------------------------------------------------------------------------
# cmd_ingest — exit codes
# ---------------------------------------------------------------------------


class TestCmdIngestExitCodes:
    def test_exits_0_on_success(self) -> None:
        with (
            patch("src.cli.commands.IngestionPipeline.build") as mock_build,
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            mock_build.return_value.run.return_value = _ok_result()
            result = runner.invoke(_app, ["chunks.json"])
        assert result.exit_code == 0

    def test_exits_1_on_file_not_found(self) -> None:
        with (
            patch("src.cli.commands.IngestionPipeline.build") as mock_build,
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            mock_build.return_value.run.side_effect = FileNotFoundError(
                "chunks.json not found"
            )
            result = runner.invoke(_app, ["chunks.json"])
        assert result.exit_code == 1

    def test_exits_2_on_runtime_error(self) -> None:
        with (
            patch("src.cli.commands.IngestionPipeline.build") as mock_build,
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            mock_build.return_value.run.side_effect = RuntimeError("Qdrant down")
            result = runner.invoke(_app, ["chunks.json"])
        assert result.exit_code == 2

    def test_exits_1_when_build_raises_file_not_found(self) -> None:
        with (
            patch(
                "src.cli.commands.IngestionPipeline.build",
                side_effect=FileNotFoundError("bm25 missing"),
            ),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["chunks.json"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# cmd_ingest — JSON output
# ---------------------------------------------------------------------------


class TestCmdIngestJsonOutput:
    def test_json_output_is_valid_json(self) -> None:
        with (
            patch("src.cli.commands.IngestionPipeline.build") as mock_build,
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            mock_build.return_value.run.return_value = _ok_result()
            result = runner.invoke(_app, ["chunks.json", "--output", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)

    def test_json_output_contains_chunks_loaded(self) -> None:
        with (
            patch("src.cli.commands.IngestionPipeline.build") as mock_build,
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            mock_build.return_value.run.return_value = _ok_result()
            result = runner.invoke(_app, ["chunks.json", "--output", "json"])
        data = json.loads(result.output)
        assert "chunks_loaded" in data

    def test_json_output_contains_source(self) -> None:
        with (
            patch("src.cli.commands.IngestionPipeline.build") as mock_build,
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            mock_build.return_value.run.return_value = _ok_result()
            result = runner.invoke(_app, ["chunks.json", "--output", "json"])
        data = json.loads(result.output)
        assert "source" in data

    def test_json_output_chunks_loaded_value_matches_result(self) -> None:
        with (
            patch("src.cli.commands.IngestionPipeline.build") as mock_build,
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            mock_build.return_value.run.return_value = _ok_result()
            result = runner.invoke(_app, ["chunks.json", "--output", "json"])
        data = json.loads(result.output)
        assert data["chunks_loaded"] == 10

    def test_json_output_zero_failed_on_clean_run(self) -> None:
        with (
            patch("src.cli.commands.IngestionPipeline.build") as mock_build,
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            mock_build.return_value.run.return_value = _ok_result()
            result = runner.invoke(_app, ["chunks.json", "--output", "json"])
        data = json.loads(result.output)
        assert data["chunks_failed"] == 0


# ---------------------------------------------------------------------------
# cmd_ingest — --reset flag forwarding
# ---------------------------------------------------------------------------


class TestCmdIngestResetFlag:
    def test_reset_false_by_default(self) -> None:
        with (
            patch("src.cli.commands.IngestionPipeline.build") as mock_build,
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            mock_build.return_value.run.return_value = _ok_result()
            runner.invoke(_app, ["chunks.json"])
        mock_build.return_value.run.assert_called_once_with(
            pytest.approx(mock_build.return_value.run.call_args.args[0]),
            reset=False,
        )

    def test_reset_true_when_flag_passed(self) -> None:
        with (
            patch("src.cli.commands.IngestionPipeline.build") as mock_build,
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            mock_build.return_value.run.return_value = _ok_result()
            runner.invoke(_app, ["chunks.json", "--reset"])
        _, call_kwargs = mock_build.return_value.run.call_args
        assert call_kwargs["reset"] is True

    def test_run_called_once(self) -> None:
        with (
            patch("src.cli.commands.IngestionPipeline.build") as mock_build,
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            mock_build.return_value.run.return_value = _ok_result()
            runner.invoke(_app, ["chunks.json"])
        mock_build.return_value.run.assert_called_once()
