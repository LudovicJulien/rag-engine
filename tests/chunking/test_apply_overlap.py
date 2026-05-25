from __future__ import annotations

import pytest

from src.chunking.chunker import ChunkConfig
from src.chunking.recursive import RecursiveTextChunker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunker(overlap: int, chunk_size: int = 512) -> RecursiveTextChunker:
    return RecursiveTextChunker(
        ChunkConfig(chunk_size=chunk_size, chunk_overlap=overlap)
    )


# ---------------------------------------------------------------------------
# No-op paths — no allocation expected
# ---------------------------------------------------------------------------


class TestApplyOverlapNoOp:
    def test_empty_list_returns_empty_list(self) -> None:
        result = _chunker(overlap=10)._apply_overlap([])
        assert result == []

    def test_single_element_returns_same_object(self) -> None:
        texts = ["only chunk"]
        result = _chunker(overlap=3)._apply_overlap(texts)
        assert result is texts

    def test_overlap_zero_returns_same_object(self) -> None:
        texts = ["first", "second", "third"]
        result = _chunker(overlap=0)._apply_overlap(texts)
        assert result is texts

    def test_overlap_zero_list_content_unchanged(self) -> None:
        texts = ["first", "second"]
        result = _chunker(overlap=0)._apply_overlap(texts)
        assert result == ["first", "second"]


# ---------------------------------------------------------------------------
# Two-chunk overlap — exact character verification
# ---------------------------------------------------------------------------


class TestApplyOverlapTwoChunks:
    def test_first_chunk_is_unchanged(self) -> None:
        result = _chunker(overlap=3)._apply_overlap(["ABCDE", "FGHIJ"])
        assert result[0] == "ABCDE"

    def test_second_chunk_prefixed_with_tail_of_first(self) -> None:
        # overlap=3: last 3 of "ABCDE" = "CDE"
        result = _chunker(overlap=3)._apply_overlap(["ABCDE", "FGHIJ"])
        assert result[1] == "CDEFGHIJ"

    def test_overlap_of_one_char(self) -> None:
        # overlap=1: last char of "ABCDE" = "E"
        result = _chunker(overlap=1)._apply_overlap(["ABCDE", "FGHIJ"])
        assert result[1] == "EFGHIJ"

    def test_result_length_equals_input_length(self) -> None:
        texts = ["ABCDE", "FGHIJ"]
        result = _chunker(overlap=3)._apply_overlap(texts)
        assert len(result) == len(texts)

    def test_original_chunk_text_present_as_suffix(self) -> None:
        # The original chunk text must still appear at the end of the result.
        texts = ["ABCDE", "FGHIJ"]
        result = _chunker(overlap=3)._apply_overlap(texts)
        assert result[1].endswith("FGHIJ")

    def test_previous_chunk_shorter_than_overlap_uses_full_previous(self) -> None:
        # "AB" has only 2 chars; overlap=5 → Python slicing gives "AB" intact.
        result = _chunker(overlap=5)._apply_overlap(["AB", "CDEFGH"])
        assert result[1] == "ABCDEFGH"


# ---------------------------------------------------------------------------
# Three-chunk overlap — chain and cascading behaviour
# ---------------------------------------------------------------------------


class TestApplyOverlapThreeChunks:
    def test_each_chunk_prefixed_with_tail_of_previous_result(self) -> None:
        # overlap=2; separators are arbitrary letters for clarity.
        # result[0] = "ABCDE"
        # result[1] = "DE" + "FGHIJ"  = "DEFGHIJ"
        # result[2] = "IJ" + "KLMNO"  = "IJKLMNO"  ← last 2 of "DEFGHIJ"
        result = _chunker(overlap=2)._apply_overlap(["ABCDE", "FGHIJ", "KLMNO"])
        assert result[0] == "ABCDE"
        assert result[1] == "DEFGHIJ"
        assert result[2] == "IJKLMNO"

    def test_cascade_uses_already_overlapped_previous(self) -> None:
        # overlap=3; middle chunk "XY" is shorter than overlap.
        # result[0] = "ABCDE"
        # result[1] = "CDE" + "XY"    = "CDEXY"
        # result[2] = "EXY" + "FGHIJ" = "EXYFGHIJ"  ← last 3 of "CDEXY"
        # (NOT "XY" + "FGHIJ" which would be what texts[i-1] gives)
        result = _chunker(overlap=3)._apply_overlap(["ABCDE", "XY", "FGHIJ"])
        assert result[1] == "CDEXY"
        assert result[2] == "EXYFGHIJ"

    def test_all_original_texts_present_as_suffix(self) -> None:
        texts = ["AAA", "BBB", "CCC"]
        result = _chunker(overlap=2)._apply_overlap(texts)
        for i, original in enumerate(texts):
            assert result[i].endswith(original)

    def test_result_length_preserved_for_long_list(self) -> None:
        texts = [f"chunk{i:02d}" for i in range(10)]
        result = _chunker(overlap=3)._apply_overlap(texts)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestApplyOverlapEdgeCases:
    def test_all_chunks_single_char_shorter_than_overlap(self) -> None:
        # Every fragment is 1 char, overlap=5 → each chunk gets full previous.
        texts = ["A", "B", "C"]
        result = _chunker(overlap=5)._apply_overlap(texts)
        assert result[0] == "A"
        # result[1]: "A"[-5:] = "A" → "AB"
        assert result[1] == "AB"
        # result[2]: "AB"[-5:] = "AB" → "ABC"
        assert result[2] == "ABC"

    def test_overlap_exact_length_of_previous_chunk(self) -> None:
        # "ABCDE" has 5 chars; overlap=5 → entire previous chunk is prefixed.
        result = _chunker(overlap=5)._apply_overlap(["ABCDE", "FGHIJ"])
        assert result[1] == "ABCDEFGHIJ"

    def test_two_identical_chunks_overlap_is_idempotent(self) -> None:
        result = _chunker(overlap=3)._apply_overlap(["HELLO", "HELLO"])
        assert result[0] == "HELLO"
        assert result[1] == "LLOHELLO"

    def test_unicode_characters_counted_by_python_len(self) -> None:
        # "héllo" has 5 chars in Python (é = 1 code point).
        result = _chunker(overlap=2)._apply_overlap(["héllo", "wörld"])
        assert result[1] == "lowörld"


# ---------------------------------------------------------------------------
# Invariants over the full result list
# ---------------------------------------------------------------------------


class TestApplyOverlapInvariants:
    @pytest.mark.parametrize(
        "texts, overlap",
        [
            (["ab", "cd", "ef", "gh"], 1),
            (["ABCDE", "FGHIJ", "KLMNO"], 3),
            (["x" * 20, "y" * 20, "z" * 20], 5),
            (["short", "a", "longer text here"], 4),
        ],
    )
    def test_each_chunk_starts_with_tail_of_previous_result(
        self, texts: list[str], overlap: int
    ) -> None:
        result = _chunker(overlap=overlap)._apply_overlap(texts)
        for i in range(1, len(result)):
            expected_prefix = result[i - 1][-overlap:]
            assert result[i].startswith(expected_prefix), (
                f"result[{i}]={result[i]!r} does not start with "
                f"result[{i-1}][-{overlap}:]={expected_prefix!r}"
            )

    @pytest.mark.parametrize("overlap", [1, 2, 5, 10])
    def test_first_chunk_never_modified(self, overlap: int) -> None:
        texts = ["FIRST_CHUNK", "second", "third"]
        result = _chunker(overlap=overlap)._apply_overlap(texts)
        assert result[0] == "FIRST_CHUNK"

    @pytest.mark.parametrize("overlap", [1, 3, 5])
    def test_result_length_always_equals_input_length(self, overlap: int) -> None:
        texts = ["a", "b", "c", "d", "e"]
        result = _chunker(overlap=overlap)._apply_overlap(texts)
        assert len(result) == len(texts)

    def test_each_result_at_least_as_long_as_original_text(self) -> None:
        texts = ["ABCDE", "FGHIJ", "KLMNO"]
        result = _chunker(overlap=3)._apply_overlap(texts)
        for i, original in enumerate(texts):
            assert len(result[i]) >= len(original)
