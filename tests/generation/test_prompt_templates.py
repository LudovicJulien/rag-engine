# tests/generation/test_prompt_templates.py
from __future__ import annotations

import pytest

from src.generation.prompt_templates import (
    CURRENT_VERSION,
    DEFAULT_DOMAIN,
    DOMAIN_PROMPTS,
    PromptTemplate,
    get_template,
)
from src.shared.models import Chunk, ChunkMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str = "c1", text: str = "Paris is the capital of France."
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        parent_doc_id="doc1",
        text=text,
        chunk_index=0,
        total_chunks=1,
        metadata=ChunkMetadata(),
    )


# ---------------------------------------------------------------------------
# CURRENT_VERSION and registry
# ---------------------------------------------------------------------------


class TestCurrentVersion:
    def test_current_version_is_v1(self) -> None:
        assert CURRENT_VERSION == "v1"

    def test_default_template_matches_current_version(self) -> None:
        assert get_template().version == CURRENT_VERSION

    def test_unknown_version_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="v99"):
            get_template("v99")

    def test_error_lists_available_versions(self) -> None:
        with pytest.raises(ValueError, match="v0"):
            get_template("unknown")


# ---------------------------------------------------------------------------
# get_template — version selection
# ---------------------------------------------------------------------------


class TestGetTemplate:
    def test_explicit_v0_returns_v0(self) -> None:
        assert get_template("v0").version == "v0"

    def test_explicit_v1_returns_v1(self) -> None:
        assert get_template("v1").version == "v1"

    def test_no_arg_returns_current(self) -> None:
        assert get_template() == get_template(CURRENT_VERSION)

    def test_returns_prompt_template_instance(self) -> None:
        assert isinstance(get_template(), PromptTemplate)

    def test_template_is_frozen(self) -> None:
        tpl = get_template()
        with pytest.raises((AttributeError, TypeError)):
            setattr(tpl, "system_prompt", "mutated")


# ---------------------------------------------------------------------------
# v0 system prompt — English-only baseline
# ---------------------------------------------------------------------------


class TestV0SystemPrompt:
    def setup_method(self) -> None:
        self.tpl = get_template("v0")

    def test_contains_context_instruction(self) -> None:
        assert "context" in self.tpl.system_prompt.lower()

    def test_contains_citation_format(self) -> None:
        assert "[id: <chunk_id>]" in self.tpl.system_prompt

    def test_contains_factual_assistant_instruction(self) -> None:
        assert "factual" in self.tpl.system_prompt.lower()

    def test_is_english_only(self) -> None:
        assert "contexte" not in self.tpl.system_prompt.lower()
        assert "vous" not in self.tpl.system_prompt.lower()


# ---------------------------------------------------------------------------
# v1 system prompt — bilingual FR / EN
# ---------------------------------------------------------------------------


class TestV1SystemPrompt:
    def setup_method(self) -> None:
        self.tpl = get_template("v1")

    def test_contains_english_context_instruction(self) -> None:
        assert "context" in self.tpl.system_prompt

    def test_contains_french_context_instruction(self) -> None:
        assert "contexte" in self.tpl.system_prompt.lower()

    def test_contains_language_detection_instruction_en(self) -> None:
        assert "language" in self.tpl.system_prompt.lower()

    def test_contains_language_detection_instruction_fr(self) -> None:
        assert "langue" in self.tpl.system_prompt.lower()

    def test_contains_citation_format_in_english_section(self) -> None:
        en_section = self.tpl.system_prompt.split("\n\n")[0]
        assert "[id: <chunk_id>]" in en_section

    def test_contains_citation_format_in_french_section(self) -> None:
        fr_section = self.tpl.system_prompt.split("\n\n")[1]
        assert "[id: <chunk_id>]" in fr_section

    def test_explicitly_mentions_french_and_english(self) -> None:
        prompt = self.tpl.system_prompt.lower()
        assert "french" in prompt or "français" in prompt
        assert "english" in prompt or "anglais" in prompt

    def test_v1_system_prompt_differs_from_v0(self) -> None:
        assert get_template("v1").system_prompt != get_template("v0").system_prompt

    def test_v1_is_longer_than_v0(self) -> None:
        assert len(get_template("v1").system_prompt) > len(
            get_template("v0").system_prompt
        )


# ---------------------------------------------------------------------------
# build_user_message
# ---------------------------------------------------------------------------


class TestBuildUserMessage:
    def test_contains_query(self) -> None:
        tpl = get_template()
        msg = tpl.build_user_message("What is Paris?", [_make_chunk()])
        assert "What is Paris?" in msg

    def test_contains_chunk_text(self) -> None:
        tpl = get_template()
        msg = tpl.build_user_message("Q?", [_make_chunk("c1", "Paris text.")])
        assert "Paris text." in msg

    def test_contains_chunk_id(self) -> None:
        tpl = get_template()
        msg = tpl.build_user_message("Q?", [_make_chunk("my-chunk-id")])
        assert "my-chunk-id" in msg

    def test_single_chunk_numbered_one(self) -> None:
        tpl = get_template()
        msg = tpl.build_user_message("Q?", [_make_chunk()])
        assert "[1]" in msg

    def test_multiple_chunks_numbered_sequentially(self) -> None:
        tpl = get_template()
        chunks = [
            _make_chunk("c1", "T1"),
            _make_chunk("c2", "T2"),
            _make_chunk("c3", "T3"),
        ]
        msg = tpl.build_user_message("Q?", chunks)
        assert "[1]" in msg
        assert "[2]" in msg
        assert "[3]" in msg

    def test_chunk_order_preserved(self) -> None:
        tpl = get_template()
        chunks = [_make_chunk("first"), _make_chunk("second")]
        msg = tpl.build_user_message("Q?", chunks)
        assert msg.index("first") < msg.index("second")

    def test_consistent_across_v0_and_v1(self) -> None:
        chunk = _make_chunk()
        v0_msg = get_template("v0").build_user_message("Q?", [chunk])
        v1_msg = get_template("v1").build_user_message("Q?", [chunk])
        assert v0_msg == v1_msg

    def test_question_appears_after_context(self) -> None:
        tpl = get_template()
        msg = tpl.build_user_message("My question here", [_make_chunk()])
        context_pos = msg.index("Context:")
        question_pos = msg.index("My question here")
        assert context_pos < question_pos


# ---------------------------------------------------------------------------
# Domain prompts
# ---------------------------------------------------------------------------


class TestDomainPrompts:
    def test_default_domain_is_general(self) -> None:
        assert DEFAULT_DOMAIN == "general"

    def test_general_domain_in_registry(self) -> None:
        assert "general" in DOMAIN_PROMPTS

    def test_tourism_domain_in_registry(self) -> None:
        assert "tourism" in DOMAIN_PROMPTS

    def test_general_domain_is_empty_string(self) -> None:
        assert DOMAIN_PROMPTS["general"] == ""

    def test_tourism_domain_is_non_empty(self) -> None:
        assert DOMAIN_PROMPTS["tourism"]

    def test_unknown_domain_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown domain"):
            get_template(domain="medical")

    def test_error_lists_available_domains(self) -> None:
        with pytest.raises(ValueError, match="general"):
            get_template(domain="unknown")


class TestGeneralDomain:
    def test_general_returns_base_template_unchanged(self) -> None:
        assert get_template("v1", "general") == get_template("v1")

    def test_general_system_prompt_equals_base(self) -> None:
        base = get_template("v1")
        with_domain = get_template("v1", "general")
        assert with_domain.system_prompt == base.system_prompt

    def test_general_version_preserved(self) -> None:
        assert get_template("v1", "general").version == "v1"


class TestTourismDomain:
    def setup_method(self) -> None:
        self.tpl = get_template("v1", "tourism")

    def test_version_is_v1(self) -> None:
        assert self.tpl.version == "v1"

    def test_system_prompt_starts_with_domain_persona(self) -> None:
        assert self.tpl.system_prompt.startswith(DOMAIN_PROMPTS["tourism"])

    def test_system_prompt_differs_from_base(self) -> None:
        assert self.tpl.system_prompt != get_template("v1").system_prompt

    def test_contains_tourism_persona(self) -> None:
        assert "tourist guide" in self.tpl.system_prompt.lower()

    def test_rag_rules_present_in_english(self) -> None:
        assert "provided context" in self.tpl.system_prompt

    def test_rag_rules_present_in_french(self) -> None:
        assert "contexte fourni" in self.tpl.system_prompt

    def test_citation_format_preserved(self) -> None:
        assert "[id: <chunk_id>]" in self.tpl.system_prompt

    def test_language_detection_rule_preserved(self) -> None:
        assert "language" in self.tpl.system_prompt.lower()
        assert "langue" in self.tpl.system_prompt.lower()

    def test_tourism_template_not_same_object_as_base(self) -> None:
        assert self.tpl is not get_template("v1")

    def test_build_user_message_unchanged_by_domain(self) -> None:
        chunk = _make_chunk()
        tourism_msg = self.tpl.build_user_message("Q?", [chunk])
        base_msg = get_template("v1").build_user_message("Q?", [chunk])
        assert tourism_msg == base_msg
