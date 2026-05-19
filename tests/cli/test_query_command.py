from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

import src.cli.commands as commands
from src.cli.commands import cmd_query
from src.pipeline.config import Settings
from src.pipeline.rag_pipeline import RAGResult

# Local app — __main__.py is wired in a later commit
_app = typer.Typer()
_app.command()(cmd_query)

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


def _ok_result() -> RAGResult:
    return RAGResult(
        query="What is Paris?",
        answer="Paris est la capitale de la France.",
        sources=["chunk-001", "chunk-002"],
        detected_language="fr",
        model="gemma3:4b",
        chunks_retrieved=2,
        tokens_used=312,
    )


def _mock_pipeline(result: RAGResult | None = None) -> MagicMock:
    mock = MagicMock()
    mock.query.return_value = result or _ok_result()
    return mock


# ---------------------------------------------------------------------------
# cmd_query — singleton via monkeypatch
# ---------------------------------------------------------------------------


class TestCmdQuerySingleton:
    def test_uses_existing_pipeline_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = _mock_pipeline()
        monkeypatch.setattr(commands, "_pipeline", mock)
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            runner.invoke(_app, ["What is Paris?"])
        mock.query.assert_called_once()

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
            runner.invoke(_app, ["What is Paris?"])
        mock_build.assert_called_once()

    def test_does_not_rebuild_pipeline_on_second_call(
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
            runner.invoke(_app, ["First question"])
            runner.invoke(_app, ["Second question"])
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
            runner.invoke(_app, ["What is Paris?"])
        assert commands._pipeline is mock


# ---------------------------------------------------------------------------
# cmd_query — exit codes
# ---------------------------------------------------------------------------


class TestCmdQueryExitCodes:
    def test_exits_0_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(commands, "_pipeline", _mock_pipeline())
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            result = runner.invoke(_app, ["What is Paris?"])
        assert result.exit_code == 0

    def test_exits_1_on_file_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(commands, "_pipeline", None)
        with (
            patch(
                "src.pipeline.rag_pipeline.RAGPipeline.build",
                side_effect=FileNotFoundError("bm25.pkl not found"),
            ),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["What is Paris?"])
        assert result.exit_code == 1

    def test_exits_2_on_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(commands, "_pipeline", None)
        with (
            patch(
                "src.pipeline.rag_pipeline.RAGPipeline.build",
                side_effect=RuntimeError("Qdrant down"),
            ),
            patch("src.cli.commands.get_settings", return_value=_settings()),
        ):
            result = runner.invoke(_app, ["What is Paris?"])
        assert result.exit_code == 2

    def test_exits_2_when_query_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = _mock_pipeline()
        mock.query.side_effect = RuntimeError("LLM timeout")
        monkeypatch.setattr(commands, "_pipeline", mock)
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            result = runner.invoke(_app, ["What is Paris?"])
        assert result.exit_code == 2

    def test_exits_1_when_query_raises_file_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = _mock_pipeline()
        mock.query.side_effect = FileNotFoundError("missing resource")
        monkeypatch.setattr(commands, "_pipeline", mock)
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            result = runner.invoke(_app, ["What is Paris?"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# cmd_query — JSON output
# ---------------------------------------------------------------------------


class TestCmdQueryJsonOutput:
    def test_json_output_is_valid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(commands, "_pipeline", _mock_pipeline())
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            result = runner.invoke(_app, ["What is Paris?", "--output", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)

    def test_json_output_contains_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(commands, "_pipeline", _mock_pipeline())
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            result = runner.invoke(_app, ["What is Paris?", "--output", "json"])
        data = json.loads(result.output)
        assert "answer" in data

    def test_json_output_answer_matches_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(commands, "_pipeline", _mock_pipeline())
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            result = runner.invoke(_app, ["What is Paris?", "--output", "json"])
        data = json.loads(result.output)
        assert data["answer"] == "Paris est la capitale de la France."

    def test_json_output_contains_sources(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(commands, "_pipeline", _mock_pipeline())
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            result = runner.invoke(_app, ["What is Paris?", "--output", "json"])
        data = json.loads(result.output)
        assert "sources" in data

    def test_json_output_contains_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(commands, "_pipeline", _mock_pipeline())
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            result = runner.invoke(_app, ["What is Paris?", "--output", "json"])
        data = json.loads(result.output)
        assert "model" in data

    def test_json_output_contains_chunks_retrieved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(commands, "_pipeline", _mock_pipeline())
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            result = runner.invoke(_app, ["What is Paris?", "--output", "json"])
        data = json.loads(result.output)
        assert "chunks_retrieved" in data


# ---------------------------------------------------------------------------
# cmd_query — option forwarding (top_k, score_threshold)
# ---------------------------------------------------------------------------


class TestCmdQueryOptionForwarding:
    def test_top_k_none_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock = _mock_pipeline()
        monkeypatch.setattr(commands, "_pipeline", mock)
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            runner.invoke(_app, ["What is Paris?"])
        _, kwargs = mock.query.call_args
        assert kwargs["top_k"] is None

    def test_top_k_forwarded_when_provided(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = _mock_pipeline()
        monkeypatch.setattr(commands, "_pipeline", mock)
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            runner.invoke(_app, ["What is Paris?", "--top-k", "5"])
        _, kwargs = mock.query.call_args
        assert kwargs["top_k"] == 5

    def test_score_threshold_none_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = _mock_pipeline()
        monkeypatch.setattr(commands, "_pipeline", mock)
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            runner.invoke(_app, ["What is Paris?"])
        _, kwargs = mock.query.call_args
        assert kwargs["score_threshold"] is None

    def test_score_threshold_forwarded_when_provided(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = _mock_pipeline()
        monkeypatch.setattr(commands, "_pipeline", mock)
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            runner.invoke(_app, ["What is Paris?", "--score-threshold", "0.75"])
        _, kwargs = mock.query.call_args
        assert kwargs["score_threshold"] == pytest.approx(0.75)

    def test_query_called_with_question(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock = _mock_pipeline()
        monkeypatch.setattr(commands, "_pipeline", mock)
        with patch("src.cli.commands.get_settings", return_value=_settings()):
            runner.invoke(_app, ["What is the capital of France?"])
        args, _ = mock.query.call_args
        assert args[0] == "What is the capital of France?"
