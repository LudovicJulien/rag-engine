from __future__ import annotations

import pytest

from src.chunking import get_chunker
from src.chunking.chunker import TextChunker
from src.chunking.recursive import RecursiveTextChunker
from src.pipeline.config import Settings


def _settings(**overrides: object) -> Settings:
    """Build a Settings instance overriding only the given chunker fields."""
    defaults = {
        "chunker_chunk_size": 256,
        "chunker_chunk_overlap": 32,
        "chunker_min_chunk_size": 16,
    }
    defaults.update(overrides)  # type: ignore[arg-type]
    return Settings(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestGetChunkerReturnType:
    def test_returns_text_chunker(self) -> None:
        assert isinstance(get_chunker(_settings()), TextChunker)

    def test_returns_recursive_text_chunker(self) -> None:
        assert isinstance(get_chunker(_settings()), RecursiveTextChunker)

    def test_returns_new_instance_on_each_call(self) -> None:
        s = _settings()
        assert get_chunker(s) is not get_chunker(s)


# ---------------------------------------------------------------------------
# Settings wiring — each field reaches ChunkConfig
# ---------------------------------------------------------------------------


class TestGetChunkerWiring:
    def test_chunk_size_wired_from_settings(self) -> None:
        chunker = get_chunker(_settings(chunker_chunk_size=128))
        assert chunker.config.chunk_size == 128

    def test_chunk_overlap_wired_from_settings(self) -> None:
        chunker = get_chunker(_settings(chunker_chunk_overlap=8))
        assert chunker.config.chunk_overlap == 8

    def test_min_chunk_size_wired_from_settings(self) -> None:
        chunker = get_chunker(_settings(chunker_min_chunk_size=10))
        assert chunker.config.min_chunk_size == 10

    def test_all_three_fields_wired_simultaneously(self) -> None:
        chunker = get_chunker(
            _settings(
                chunker_chunk_size=300,
                chunker_chunk_overlap=20,
                chunker_min_chunk_size=5,
            )
        )
        assert chunker.config.chunk_size == 300
        assert chunker.config.chunk_overlap == 20
        assert chunker.config.min_chunk_size == 5


# ---------------------------------------------------------------------------
# Default settings fallback (None → get_settings())
# ---------------------------------------------------------------------------


class TestGetChunkerDefaultSettings:
    def test_none_uses_module_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.pipeline.config as cfg

        custom = Settings(chunker_chunk_size=777, chunker_chunk_overlap=0)
        monkeypatch.setattr(cfg, "settings", custom)
        chunker = get_chunker(None)
        assert chunker.config.chunk_size == 777

    def test_explicit_settings_takes_priority_over_singleton(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CHUNKER_CHUNK_SIZE", "999")
        chunker = get_chunker(_settings(chunker_chunk_size=42))
        assert chunker.config.chunk_size == 42


# ---------------------------------------------------------------------------
# Validation propagation — Settings → ChunkConfig cross-field constraint
# ---------------------------------------------------------------------------


class TestGetChunkerValidation:
    def test_overlap_equal_to_chunk_size_raises(self) -> None:
        # Settings validates fields individually; ChunkConfig enforces overlap < size.
        s = Settings(
            chunker_chunk_size=100,
            chunker_chunk_overlap=100,
            chunker_min_chunk_size=0,
        )
        with pytest.raises(ValueError, match="chunk_overlap"):
            get_chunker(s)

    def test_overlap_greater_than_chunk_size_raises(self) -> None:
        s = Settings(
            chunker_chunk_size=50,
            chunker_chunk_overlap=60,
            chunker_min_chunk_size=0,
        )
        with pytest.raises(ValueError, match="chunk_overlap"):
            get_chunker(s)


# ---------------------------------------------------------------------------
# End-to-end: returned chunker can actually chunk a document
# ---------------------------------------------------------------------------


class TestGetChunkerEndToEnd:
    def test_returned_chunker_can_chunk_a_document(self) -> None:
        from src.shared.models import Document

        chunker = get_chunker(_settings(chunker_chunk_size=50, chunker_chunk_overlap=0))
        doc = Document(doc_id="e2e", text="word " * 40)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        assert all(c.parent_doc_id == "e2e" for c in chunks)

    def test_default_chunk_size_matches_settings_default(self) -> None:
        # Settings default is 512; verify it flows through to the config.
        chunker = get_chunker(Settings())
        assert chunker.config.chunk_size == Settings().chunker_chunk_size
