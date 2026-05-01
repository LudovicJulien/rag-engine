# tests/ingestion/test_models.py
from __future__ import annotations

from datetime import datetime

import pytest

from src.ingestion.models import Document


class TestDocumentCreation:
    """Tests for valid Document instantiation."""

    def test_minimal_valid_document(self) -> None:
        """Document creates successfully with required fields only."""
        doc = Document(id="doc-001", content="Hello world", source="test")
        assert doc.id == "doc-001"
        assert doc.content == "Hello world"
        assert doc.source == "test"

    def test_default_metadata_is_empty_dict(self) -> None:
        """Metadata defaults to empty dict, not shared mutable object."""
        doc1 = Document(id="doc-001", content="Hello", source="test")
        doc2 = Document(id="doc-002", content="World", source="test")
        doc1.metadata["key"] = "value"
        assert doc2.metadata == {}

    def test_default_ingested_at_is_datetime(self) -> None:
        """ingested_at defaults to a valid datetime."""
        doc = Document(id="doc-001", content="Hello", source="test")
        assert isinstance(doc.ingested_at, datetime)

    def test_custom_metadata(self) -> None:
        """Document accepts arbitrary metadata."""
        metadata = {"type": "restaurant", "quartier": "Plateau"}
        doc = Document(id="doc-001", content="Hello", source="test", metadata=metadata)
        assert doc.metadata["type"] == "restaurant"
        assert doc.metadata["quartier"] == "Plateau"

    def test_custom_ingested_at(self) -> None:
        """Document accepts a custom ingested_at timestamp."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        doc = Document(id="doc-001", content="Hello", source="test", ingested_at=ts)
        assert doc.ingested_at == ts


class TestDocumentValidation:
    """Tests for Document validation in __post_init__."""

    def test_empty_id_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Document id cannot be empty"):
            Document(id="", content="Hello", source="test")

    def test_empty_content_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Document content cannot be empty"):
            Document(id="doc-001", content="", source="test")

    def test_empty_source_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Document source cannot be empty"):
            Document(id="doc-001", content="Hello", source="")


class TestDocumentProperties:
    """Tests for Document computed properties."""

    def test_word_count_single_word(self) -> None:
        doc = Document(id="doc-001", content="Hello", source="test")
        assert doc.word_count == 1

    def test_word_count_multiple_words(self) -> None:
        doc = Document(id="doc-001", content="Hello world foo bar", source="test")
        assert doc.word_count == 4

    def test_char_count(self) -> None:
        doc = Document(id="doc-001", content="Hello", source="test")
        assert doc.char_count == 5

    def test_word_count_and_char_count_consistency(self) -> None:
        """char_count should always be >= word_count for non-empty content."""
        doc = Document(id="doc-001", content="Hello world", source="test")
        assert doc.char_count >= doc.word_count
