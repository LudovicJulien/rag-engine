# tests/generation/test_context_window.py
from __future__ import annotations

import logging

import pytest

from src.generation.context_window import guard_context_window
from src.shared.models import Chunk

# The context window guard estimates token usage from:
# - the query
# - the system prompt
# - a fixed internal message overhead
#
# The tests below validate the observable truncation behavior rather than
# relying on internal implementation details.


def _make_chunk(chunk_id: str, char_count: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        parent_doc_id=f"doc-{chunk_id}",
        text="x" * char_count,
        chunk_index=0,
        total_chunks=1,
    )


# ---------------------------------------------------------------------------
# Guard disabled
# ---------------------------------------------------------------------------


class TestGuardDisabled:
    def test_zero_max_tokens_returns_all_chunks(self) -> None:
        chunks = [_make_chunk("c1", 400), _make_chunk("c2", 400)]
        assert guard_context_window(chunks, "query", "system", 0) == chunks

    def test_negative_max_tokens_returns_all_chunks(self) -> None:
        chunks = [_make_chunk("c1", 400)]
        assert guard_context_window(chunks, "query", "system", -1) == chunks

    def test_zero_max_tokens_returns_same_list_object(self) -> None:
        chunks = [_make_chunk("c1", 40)]
        assert guard_context_window(chunks, "", "", 0) == chunks


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestGuardEmpty:
    def test_empty_chunks_with_active_guard_returns_empty(self) -> None:
        assert guard_context_window([], "query", "system", 1000) == []

    def test_empty_chunks_with_disabled_guard_returns_empty(self) -> None:
        assert guard_context_window([], "query", "system", 0) == []


# ---------------------------------------------------------------------------
# All chunks fit
# ---------------------------------------------------------------------------


class TestGuardAllFit:
    def test_all_chunks_fit_returns_all(self) -> None:
        # 3 chunks × 40 chars = 10 tokens each; base=100; budget=9900
        chunks = [_make_chunk(f"c{i}", 40) for i in range(3)]
        assert guard_context_window(chunks, "", "", 10000) == chunks

    def test_exact_budget_returns_all(self) -> None:
        # 2 chunks × 400 chars = 100 tokens each
        # base=100; max_tokens=300; budget=200
        # chunk1(100) ≤ 200 → selected, budget=100
        # chunk2(100) ≤ 100 → selected, budget=0  → exact fit
        chunks = [_make_chunk("c1", 400), _make_chunk("c2", 400)]
        assert guard_context_window(chunks, "", "", 300) == chunks

    def test_single_chunk_fits(self) -> None:
        chunk = _make_chunk("solo", 40)
        assert guard_context_window([chunk], "", "", 500) == [chunk]

    def test_no_warning_when_all_fit(self, caplog: pytest.LogCaptureFixture) -> None:
        chunks = [_make_chunk("c1", 40)]
        with caplog.at_level(logging.WARNING, logger="src.generation.context_window"):
            guard_context_window(chunks, "", "", 10000)
        assert caplog.text == ""


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestGuardTruncation:
    def test_truncates_to_one_chunk(self) -> None:
        # base=100; max_tokens=201; budget=101
        # chunk1(100) ≤ 101 → selected, budget=1
        # chunk2(100) > 1   → break
        chunks = [_make_chunk(f"c{i}", 400) for i in range(3)]
        result = guard_context_window(chunks, "", "", 201)
        assert result == [chunks[0]]

    def test_truncates_to_two_chunks(self) -> None:
        # base=100; max_tokens=300; budget=200
        # chunk1(100) ≤ 200 → selected, budget=100
        # chunk2(100) ≤ 100 → selected, budget=0
        # chunk3(100) > 0   → break
        chunks = [_make_chunk(f"c{i}", 400) for i in range(3)]
        result = guard_context_window(chunks, "", "", 300)
        assert result == [chunks[0], chunks[1]]

    def test_preserves_relevance_order(self) -> None:
        chunks = [
            _make_chunk("top", 40),  # 10 tokens
            _make_chunk("bottom", 400),  # 100 tokens
        ]
        # base=100; max_tokens=200; budget=100
        # top(10) → selected, budget=90; bottom(100) > 90 → break
        result = guard_context_window(chunks, "", "", 200)
        assert result == [chunks[0]]
        assert result[0].chunk_id == "top"

    def test_truncation_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        chunks = [_make_chunk(f"c{i}", 400) for i in range(3)]
        with caplog.at_level(logging.WARNING, logger="src.generation.context_window"):
            guard_context_window(chunks, "", "", 201)
        assert "truncated" in caplog.text.lower()

    def test_warning_contains_original_and_final_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        chunks = [_make_chunk(f"c{i}", 400) for i in range(3)]
        with caplog.at_level(logging.WARNING, logger="src.generation.context_window"):
            guard_context_window(chunks, "", "", 201)
        assert "3" in caplog.text  # original count
        assert "1" in caplog.text  # final count


# ---------------------------------------------------------------------------
# Minimum one chunk guarantee
# ---------------------------------------------------------------------------


class TestGuardMinimumOne:
    def test_keeps_first_chunk_when_budget_tiny(self) -> None:
        # max_tokens=1: base=100 > 1, budget=-99, but always keep ≥1
        chunks = [_make_chunk("must-keep", 400), _make_chunk("dropped", 400)]
        result = guard_context_window(chunks, "", "", 1)
        assert len(result) >= 1
        assert result[0].chunk_id == "must-keep"

    def test_keeps_first_chunk_when_system_prompt_huge(self) -> None:
        # system=1000 chars=250 tokens; base=350; max_tokens=10 → budget=-340
        chunks = [_make_chunk("c1", 400), _make_chunk("c2", 400)]
        result = guard_context_window(chunks, "", "s" * 1000, 10)
        assert result == [chunks[0]]

    def test_single_large_chunk_always_returned(self) -> None:
        chunk = _make_chunk("big", 100_000)  # 25 000 tokens
        result = guard_context_window([chunk], "", "", 100)
        assert result == [chunk]

    def test_only_first_chunk_returned_when_budget_exhausted_after_it(self) -> None:
        # budget=100; chunk1(100) fills it exactly → budget=0; chunk2 dropped
        chunks = [_make_chunk("c1", 400), _make_chunk("c2", 400)]
        result = guard_context_window(chunks, "", "", 200)
        assert result == [chunks[0]]


# ---------------------------------------------------------------------------
# Base token accounting
# ---------------------------------------------------------------------------


class TestGuardBaseTokens:
    def test_query_tokens_reduce_available_budget(self) -> None:
        # query=400 chars=100 tokens; base=100+0+100=200
        # max_tokens=300; budget=100
        # chunk1(100) ≤ 100 → selected; chunk2(100) > 0 → dropped
        chunks = [_make_chunk("c1", 400), _make_chunk("c2", 400)]
        result = guard_context_window(chunks, "a" * 400, "", 300)
        assert result == [chunks[0]]

    def test_system_prompt_tokens_reduce_available_budget(self) -> None:
        # system=400 chars=100 tokens; base=0+100+100=200
        # max_tokens=300; budget=100; chunk1 fits, chunk2 dropped
        chunks = [_make_chunk("c1", 400), _make_chunk("c2", 400)]
        result = guard_context_window(chunks, "", "s" * 400, 300)
        assert result == [chunks[0]]

    def test_overhead_is_accounted_for_in_budget(self) -> None:
        # base with empty strings = 0+0+100=100; max_tokens=200; budget=100
        # chunk of 100 tokens fits exactly
        chunk = _make_chunk("c1", 400)
        assert guard_context_window([chunk], "", "", 200) == [chunk]
