from __future__ import annotations

import pytest

from src.chunking.chunker import _DEFAULT_SEPARATORS, ChunkConfig


class TestChunkConfigDefaults:
    """Default values are valid and cover the expected use-case."""

    def test_default_chunk_size(self) -> None:
        assert ChunkConfig().chunk_size == 512

    def test_default_chunk_overlap(self) -> None:
        assert ChunkConfig().chunk_overlap == 64

    def test_default_min_chunk_size(self) -> None:
        assert ChunkConfig().min_chunk_size == 32

    def test_default_separators_match_module_constant(self) -> None:
        assert ChunkConfig().separators == _DEFAULT_SEPARATORS

    def test_default_overlap_is_less_than_default_chunk_size(self) -> None:
        config = ChunkConfig()
        assert config.chunk_overlap < config.chunk_size

    def test_default_separators_end_with_empty_string_fallback(self) -> None:
        """The last separator must be "" to guarantee termination on any input."""
        assert ChunkConfig().separators[-1] == ""

    def test_default_separators_start_with_broadest_boundary(self) -> None:
        assert ChunkConfig().separators[0] == "\n\n\n"


class TestChunkConfigValidation:
    """__post_init__ rejects every invalid combination."""

    def test_raises_if_chunk_size_is_zero(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            ChunkConfig(chunk_size=0)

    def test_raises_if_chunk_size_is_negative(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            ChunkConfig(chunk_size=-1)

    def test_raises_if_chunk_overlap_is_negative(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkConfig(chunk_overlap=-1)

    def test_raises_if_chunk_overlap_equals_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkConfig(chunk_size=100, chunk_overlap=100)

    def test_raises_if_chunk_overlap_exceeds_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkConfig(chunk_size=100, chunk_overlap=101)

    def test_raises_if_min_chunk_size_is_negative(self) -> None:
        with pytest.raises(ValueError, match="min_chunk_size"):
            ChunkConfig(min_chunk_size=-1)

    def test_raises_if_separators_is_empty_tuple(self) -> None:
        with pytest.raises(ValueError, match="separators"):
            ChunkConfig(separators=())


class TestChunkConfigValidBoundaries:
    """Edge values that must be accepted without error."""

    def test_chunk_size_of_one_is_valid(self) -> None:
        config = ChunkConfig(chunk_size=1, chunk_overlap=0)
        assert config.chunk_size == 1

    def test_chunk_overlap_of_zero_is_valid(self) -> None:
        config = ChunkConfig(chunk_overlap=0)
        assert config.chunk_overlap == 0

    def test_min_chunk_size_of_zero_is_valid(self) -> None:
        config = ChunkConfig(min_chunk_size=0)
        assert config.min_chunk_size == 0

    def test_overlap_one_less_than_chunk_size_is_valid(self) -> None:
        config = ChunkConfig(chunk_size=10, chunk_overlap=9)
        assert config.chunk_overlap == 9

    def test_single_separator_is_valid(self) -> None:
        config = ChunkConfig(separators=("",))
        assert config.separators == ("",)

    def test_custom_separators_are_stored(self) -> None:
        seps = ("\n\n", "\n", " ", "")
        config = ChunkConfig(separators=seps)
        assert config.separators == seps


class TestChunkConfigImmutability:
    """ChunkConfig is frozen — mutations must raise."""

    def test_cannot_mutate_chunk_size(self) -> None:
        config = ChunkConfig()
        with pytest.raises(AttributeError):
            config.chunk_size = 1024  # type: ignore[misc]

    def test_cannot_mutate_separators(self) -> None:
        config = ChunkConfig()
        with pytest.raises(AttributeError):
            config.separators = ("\n",)  # type: ignore[misc]

    def test_equality_on_identical_configs(self) -> None:
        assert ChunkConfig(chunk_size=256) == ChunkConfig(chunk_size=256)

    def test_inequality_on_different_chunk_size(self) -> None:
        assert ChunkConfig(chunk_size=256) != ChunkConfig(chunk_size=512)

    def test_hashable(self) -> None:
        """Frozen dataclasses must be usable as dict keys or set members."""
        config = ChunkConfig()
        assert hash(config) == hash(config)
        seen = {config}
        assert config in seen
