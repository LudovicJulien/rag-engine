from __future__ import annotations

import pytest

from src.chunking.chunker import ChunkConfig
from src.chunking.recursive import RecursiveTextChunker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunker(chunk_size: int, **kwargs: object) -> RecursiveTextChunker:
    """Build a RecursiveTextChunker with a fixed chunk_size and no overlap."""
    config = ChunkConfig(
        chunk_size=chunk_size, chunk_overlap=0, **kwargs  # type: ignore[arg-type]
    )
    return RecursiveTextChunker(config)


def _seps(*args: str) -> list[str]:
    return list(args)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestRecursiveTextChunkerInit:
    def test_default_config_applied_when_none_given(self) -> None:
        chunker = RecursiveTextChunker()
        assert chunker.config == ChunkConfig()

    def test_custom_config_stored(self) -> None:
        config = ChunkConfig(chunk_size=256, chunk_overlap=0)
        chunker = RecursiveTextChunker(config)
        assert chunker.config is config

    def test_config_property_returns_same_object(self) -> None:
        chunker = RecursiveTextChunker()
        assert chunker.config is chunker.config


# ---------------------------------------------------------------------------
# Early exit — text already within budget
# ---------------------------------------------------------------------------


class TestSplitRecursiveEarlyExit:
    def test_text_shorter_than_chunk_size_returned_as_single_item(self) -> None:
        chunker = _chunker(chunk_size=100)
        result = chunker._split_recursive("short text", _seps("\n\n", " ", ""))
        assert result == ["short text"]

    def test_text_exactly_chunk_size_returned_as_single_item(self) -> None:
        text = "x" * 50
        chunker = _chunker(chunk_size=50)
        result = chunker._split_recursive(text, _seps(" ", ""))
        assert result == [text]

    def test_early_exit_does_not_recurse_into_separators(self) -> None:
        # Text with paragraph breaks but short enough — must not be split.
        text = "a\n\nb"
        chunker = _chunker(chunk_size=100)
        result = chunker._split_recursive(text, _seps("\n\n", " ", ""))
        assert result == [text]


# ---------------------------------------------------------------------------
# Empty / missing separators
# ---------------------------------------------------------------------------


class TestSplitRecursiveNoSeparators:
    def test_empty_separator_list_returns_text_unchanged(self) -> None:
        chunker = _chunker(chunk_size=5)
        result = chunker._split_recursive("hello world", [])
        assert result == ["hello world"]

    def test_separator_absent_from_text_falls_through_to_next(self) -> None:
        # "\n\n" not present → should fall through to " " and split on words.
        text = "hello world foo bar"  # 18 chars
        chunker = _chunker(chunk_size=10)
        result = chunker._split_recursive(text, _seps("\n\n", " ", ""))
        assert all(len(f) <= 10 for f in result)
        assert len(result) > 1

    def test_no_usable_separator_returns_text_unchanged(self) -> None:
        # No separator matches; only separator list is empty after fallthrough.
        chunker = _chunker(chunk_size=3)
        result = chunker._split_recursive("abcdef", [])
        assert result == ["abcdef"]


# ---------------------------------------------------------------------------
# Single separator — paragraph / word level
# ---------------------------------------------------------------------------


class TestSplitRecursiveSingleSeparator:
    def test_paragraph_separator_splits_text(self) -> None:
        text = "first paragraph\n\nsecond paragraph"
        chunker = _chunker(chunk_size=20)
        result = chunker._split_recursive(text, _seps("\n\n"))
        assert "first paragraph" in result
        assert "second paragraph" in result

    def test_word_separator_splits_long_line(self) -> None:
        # 5 words of 6 chars each = 34 chars total; chunk_size = 10.
        text = "banana cherry mango lemon grape"
        chunker = _chunker(chunk_size=10)
        result = chunker._split_recursive(text, _seps(" "))
        assert len(result) > 1
        assert all(len(f) <= 10 for f in result)

    def test_small_fragments_accumulated_into_larger_chunk(self) -> None:
        # Each word is 1 char; chunk_size = 5 → words should be merged.
        text = "a b c d e f g h"
        chunker = _chunker(chunk_size=5)
        result = chunker._split_recursive(text, _seps(" "))
        # No fragment should be a bare single char when neighbours fit.
        assert all(len(f) > 1 for f in result)

    def test_separator_rejoined_between_accumulated_parts(self) -> None:
        # "a\n\nb" and "c\n\nd" should be merged to "a\n\nb\n\nc\n\nd"
        # if it fits within chunk_size.
        text = "a\n\nb\n\nc\n\nd"  # 10 chars
        chunker = _chunker(chunk_size=50)
        result = chunker._split_recursive(text, _seps("\n\n"))
        assert result == [text]

    def test_no_empty_strings_in_result(self) -> None:
        text = "a\n\n\n\nb"  # consecutive \n\n produces empty split
        chunker = _chunker(chunk_size=5)
        result = chunker._split_recursive(text, _seps("\n\n", ""))
        assert all(len(f) > 0 for f in result)

    def test_result_is_non_empty_for_non_empty_text(self) -> None:
        chunker = _chunker(chunk_size=5)
        result = chunker._split_recursive("hello world", _seps(" "))
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Hierarchical descent — recursive fallback
# ---------------------------------------------------------------------------


class TestSplitRecursiveHierarchicalDescent:
    def test_oversized_paragraph_recursed_into_words(self) -> None:
        # One long paragraph that needs word-level splitting.
        long_para = "one two three four five six seven eight nine ten"  # 48 chars
        text = f"{long_para}\n\nshort"
        chunker = _chunker(chunk_size=20)
        result = chunker._split_recursive(text, _seps("\n\n", " ", ""))
        assert all(len(f) <= 20 for f in result)
        assert "short" in result

    def test_oversized_word_recursed_into_characters(self) -> None:
        # A single very long "word" with no internal spaces.
        long_word = "a" * 20
        text = f"prefix {long_word} suffix"
        chunker = _chunker(chunk_size=10)
        result = chunker._split_recursive(text, _seps(" ", ""))
        assert all(len(f) <= 10 for f in result)

    def test_three_level_descent_paragraphs_sentences_words(self) -> None:
        sentence = "word " * 10  # 50 chars
        text = f"{sentence}\n\n{sentence}"
        chunker = _chunker(chunk_size=15)
        result = chunker._split_recursive(text, _seps("\n\n", ". ", " ", ""))
        assert all(len(f) <= 15 for f in result)

    def test_small_fragments_after_recursion_not_lost(self) -> None:
        # "short" should appear somewhere in the result, even after descent.
        text = "short\n\n" + "x" * 30
        chunker = _chunker(chunk_size=10)
        result = chunker._split_recursive(text, _seps("\n\n", " ", ""))
        concatenated = "".join(result)
        assert "short" in concatenated


# ---------------------------------------------------------------------------
# Character-level fallback (empty string separator)
# ---------------------------------------------------------------------------


class TestSplitRecursiveCharacterFallback:
    def test_empty_separator_splits_into_character_groups(self) -> None:
        text = "abcdefghij"  # 10 chars
        chunker = _chunker(chunk_size=3)
        result = chunker._split_recursive(text, _seps(""))
        assert result == ["abc", "def", "ghi", "j"]

    def test_empty_separator_respects_chunk_size_boundary(self) -> None:
        text = "x" * 17
        chunker = _chunker(chunk_size=5)
        result = chunker._split_recursive(text, _seps(""))
        assert all(len(f) <= 5 for f in result)
        assert "".join(result) == text

    def test_empty_separator_single_char_chunk_size(self) -> None:
        chunker = _chunker(chunk_size=1)
        result = chunker._split_recursive("abc", _seps(""))
        assert result == ["a", "b", "c"]

    def test_empty_separator_preserves_all_characters(self) -> None:
        text = "Hello, World!"
        chunker = _chunker(chunk_size=4)
        result = chunker._split_recursive(text, _seps(""))
        assert "".join(result) == text


# ---------------------------------------------------------------------------
# Invariant: every fragment <= chunk_size
# ---------------------------------------------------------------------------


class TestSplitRecursiveFragmentSizeInvariant:
    """The core correctness guarantee: every returned fragment fits the budget."""

    @pytest.mark.parametrize(
        "text, chunk_size",
        [
            ("word " * 50, 20),
            ("a\n\n" * 30 + "last", 15),
            ("sentence. " * 20, 25),
            ("x" * 200, 7),
            ("ab cd\n\nef gh\n\nij kl", 5),
        ],
    )
    def test_all_fragments_within_chunk_size(self, text: str, chunk_size: int) -> None:
        chunker = _chunker(chunk_size=chunk_size)
        result = chunker._split_recursive(text, list(chunker.config.separators))
        violations = [f for f in result if len(f) > chunk_size]
        assert (
            violations == []
        ), f"Fragments exceed chunk_size={chunk_size}: " + ", ".join(
            repr(v) for v in violations
        )

    def test_result_contains_no_empty_fragments(self) -> None:
        text = "a\n\n\n\nb  c\n\nd"
        chunker = _chunker(chunk_size=5)
        result = chunker._split_recursive(text, list(chunker.config.separators))
        assert all(f for f in result), f"Empty fragment found in {result}"
