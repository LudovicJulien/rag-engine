# tests/shared/test_enums.py
from __future__ import annotations

import pytest

from src.shared.enums import Language


class TestLanguageEnum:
    """Tests for Language enum values and behavior."""

    def test_fr_value(self) -> None:
        """FR maps to ISO 639-1 code 'fr'."""
        assert Language.FR.value == "fr"

    def test_en_value(self) -> None:
        """EN maps to ISO 639-1 code 'en'."""
        assert Language.EN.value == "en"

    def test_language_is_string(self) -> None:
        """Language inherits from str — direct string comparison works."""
        assert Language.FR == "fr"
        assert Language.EN == "en"

    def test_language_in_string_context(self) -> None:
        """Language can be used directly where a string is expected."""
        assert f"{Language.FR}" == "Language.FR"
        assert Language.FR.value in ["fr", "en"]

    def test_all_values_are_lowercase(self) -> None:
        """All language codes must be lowercase ISO 639-1."""
        for lang in Language:
            assert lang.value == lang.value.lower()

    def test_all_values_are_two_chars(self) -> None:
        """All language codes must be exactly 2 characters (ISO 639-1)."""
        for lang in Language:
            assert len(lang.value) == 2


class TestLanguageFromCode:
    """Tests for Language.from_code() factory method."""

    def test_from_code_lowercase(self) -> None:
        assert Language.from_code("fr") == Language.FR
        assert Language.from_code("en") == Language.EN

    def test_from_code_uppercase(self) -> None:
        """from_code() is case-insensitive."""
        assert Language.from_code("FR") == Language.FR
        assert Language.from_code("EN") == Language.EN

    def test_from_code_mixed_case(self) -> None:
        assert Language.from_code("Fr") == Language.FR
        assert Language.from_code("En") == Language.EN

    def test_from_code_invalid_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported language code"):
            Language.from_code("de")

    def test_from_code_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported language code"):
            Language.from_code("")

    def test_from_code_error_message_contains_supported_codes(self) -> None:
        """Error message must list the supported language codes."""
        with pytest.raises(ValueError, match="fr"):
            Language.from_code("xx")

    def test_from_code_roundtrip(self) -> None:
        """from_code(lang.value) must return the same language."""
        for lang in Language:
            assert Language.from_code(lang.value) == lang