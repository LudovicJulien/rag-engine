# tests/generation/test_language_detection.py
from __future__ import annotations

from src.generation.language_detection import detect_language

# ---------------------------------------------------------------------------
# French queries
# ---------------------------------------------------------------------------


class TestFrenchDetection:
    def test_typical_french_question(self) -> None:
        assert detect_language("Quelle est la capitale de la France ?") == "fr"

    def test_french_with_accents(self) -> None:
        assert detect_language("Où se trouve le musée du Louvre ?") == "fr"

    def test_french_subject_pronouns(self) -> None:
        assert detect_language("Je voudrais savoir comment il est possible") == "fr"

    def test_french_plural_article(self) -> None:
        assert (
            detect_language("Quels sont les meilleurs restaurants du quartier ?")
            == "fr"
        )

    def test_french_possessives(self) -> None:
        assert (
            detect_language("Donnez-moi les détails sur son histoire et ses monuments")
            == "fr"
        )

    def test_french_prepositions(self) -> None:
        assert (
            detect_language("Les activités disponibles dans ce parc avec des enfants")
            == "fr"
        )

    def test_french_why_question(self) -> None:
        assert detect_language("Pourquoi ce monument est-il célèbre ?") == "fr"

    def test_french_single_stopword(self) -> None:
        assert detect_language("le") == "fr"

    def test_french_question_word_comment(self) -> None:
        assert detect_language("Comment visiter ce site ?") == "fr"


# ---------------------------------------------------------------------------
# English queries
# ---------------------------------------------------------------------------


class TestEnglishDetection:
    def test_typical_english_question(self) -> None:
        assert detect_language("What is the capital of France?") == "en"

    def test_english_with_auxiliary(self) -> None:
        assert detect_language("What are the best restaurants in this area?") == "en"

    def test_english_subject_pronouns(self) -> None:
        assert detect_language("Can you tell me how it works?") == "en"

    def test_english_determiners(self) -> None:
        assert detect_language("Is this the right place to visit?") == "en"

    def test_english_possessives(self) -> None:
        assert (
            detect_language("What is its history and what are its main attractions?")
            == "en"
        )

    def test_english_why_question(self) -> None:
        assert detect_language("Why is this monument so famous?") == "en"

    def test_english_single_stopword_the(self) -> None:
        assert detect_language("the") == "en"

    def test_english_past_tense(self) -> None:
        assert (
            detect_language("When was this building constructed and who designed it?")
            == "en"
        )

    def test_english_modal_verb(self) -> None:
        assert detect_language("Could you recommend the best route to take?") == "en"


# ---------------------------------------------------------------------------
# Edge cases and defaults
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_string_defaults_to_fr(self) -> None:
        assert detect_language("") == "fr"

    def test_no_stopwords_defaults_to_fr(self) -> None:
        assert detect_language("Paris Lyon Marseille Bordeaux") == "fr"

    def test_numbers_only_defaults_to_fr(self) -> None:
        assert detect_language("42 1789 2024") == "fr"

    def test_tie_defaults_to_fr(self) -> None:
        # "le" (fr=1) vs "the" (en=1) → tie → "fr"
        assert detect_language("le the") == "fr"

    def test_case_insensitive_french(self) -> None:
        assert detect_language("Quelle EST LA Capitale") == "fr"

    def test_case_insensitive_english(self) -> None:
        assert detect_language("WHAT IS THE Capital") == "en"

    def test_mixed_punctuation_ignored(self) -> None:
        assert detect_language("Quelle est la meilleure route ? (en voiture)") == "fr"

    def test_returns_fr_or_en_only(self) -> None:
        for query in ["hello", "bonjour", "test", ""]:
            result = detect_language(query)
            assert result in {"fr", "en"}

    def test_single_word_no_stopword_defaults_to_fr(self) -> None:
        assert detect_language("Eiffel") == "fr"
