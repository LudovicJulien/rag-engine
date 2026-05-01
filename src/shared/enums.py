# src/shared/enums.py
from __future__ import annotations

from enum import Enum


class Language(str, Enum):
    """Supported languages for text processing and generation.

    Inherits from str to allow direct comparison with string values
    and seamless JSON serialization.
    """

    FR = "fr"
    EN = "en"

    @classmethod
    def from_code(cls, code: str) -> Language:
        """Create a Language from a ISO 639-1 code, case-insensitive."""
        try:
            return cls(code.lower())
        except ValueError:
            raise ValueError(
                f"Unsupported language code: '{code}'. "
                f"Supported: {[l.value for l in cls]}"
            )
