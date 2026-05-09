# src/generation/prompt_templates.py
from __future__ import annotations

from dataclasses import dataclass

from src.shared.models import Chunk

# Bump this constant whenever a new prompt version becomes the default.
CURRENT_VERSION = "v1"

_V0_SYSTEM = (
    "You are a precise and factual assistant. "
    "Answer the user's question based solely on the provided context. "
    "If the context does not contain enough information to answer, "
    "say so clearly. "
    "When you use information from a source, cite its ID "
    "inline using the format [id: <chunk_id>]."
)

# v1 — bilingual (EN + FR): instructs the model to detect the query language
# and reply in the same language.  Both instruction sets are included so that
# monolingual models trained primarily in one language still understand the
# citation and answering rules.
_V1_SYSTEM = (
    "You are a precise and factual assistant. "
    "Answer the user's question based solely on the provided context. "
    "If the context does not contain enough information to answer, say so clearly. "
    "Detect the language of the user's question and always respond in that same "
    "language (French or English). "
    "When you use information from a source, cite its ID inline using the format "
    "[id: <chunk_id>].\n\n"
    "Vous êtes un assistant précis et factuel. "
    "Répondez à la question de l'utilisateur en vous basant uniquement sur le "
    "contexte fourni. "
    "Si le contexte ne contient pas suffisamment d'informations pour répondre, "
    "dites-le clairement. "
    "Détectez la langue de la question de l'utilisateur et répondez toujours dans "
    "cette même langue (français ou anglais). "
    "Lorsque vous utilisez une information d'une source, citez son identifiant "
    "avec le format [id: <chunk_id>]."
)

DOMAIN_PROMPTS: dict[str, str] = {
    "general": "",
    "tourism": (
        "You are a professional tourist guide assistant. "
        "Your role is to help users discover places, attractions, activities, "
        "restaurants, and travel recommendations. "
        "Be practical, helpful, and location-aware. "
        "Prioritize useful suggestions, itineraries, and clear explanations. "
        "If multiple options exist, propose structured recommendations."
    ),
}

DEFAULT_DOMAIN = "general"

# Bilingual RAG rules (no persona line) appended to domain-specific prompts so
# that context-only answering, language detection, and citation rules are
# enforced regardless of which domain persona is active.
_V1_DOMAIN_RULES = (
    "Answer the user's question based solely on the provided context. "
    "If the context does not contain enough information to answer, say so clearly. "
    "Detect the language of the user's question and always respond in that same "
    "language (French or English). "
    "When you use information from a source, cite its ID inline using the format "
    "[id: <chunk_id>].\n\n"
    "Répondez à la question de l'utilisateur en vous basant uniquement sur le "
    "contexte fourni. "
    "Si le contexte ne contient pas suffisamment d'informations, dites-le clairement. "
    "Détectez la langue de la question et répondez toujours dans cette même langue "
    "(français ou anglais). "
    "Lorsque vous utilisez une source, citez son identifiant avec le format "
    "[id: <chunk_id>]."
)


@dataclass(frozen=True)
class PromptTemplate:
    """Versioned prompt template used by all LLM generators.

    Attributes:
        version: Identifier of this template version (e.g. ``"v1"``).
            Stored in :attr:`~src.generation.generator.GenerationResult.prompt_version`
            so evaluation runs can be compared across prompt iterations.
        system_prompt: The system-role message sent to the model before the
            user turn.
    """

    version: str
    system_prompt: str

    def build_user_message(self, query: str, context: list[Chunk]) -> str:
        """Format the numbered context chunks and the user query into a single
        user-role message string.

        Args:
            query: The user's question.
            context: Retrieved chunks ordered by descending relevance.

        Returns:
            A formatted string with all chunks numbered, each labelled with
            its ``chunk_id``, followed by the question.
        """
        formatted_chunks = "\n\n".join(
            f"[{i + 1}] (id: {chunk.chunk_id})\n{chunk.text}"
            for i, chunk in enumerate(context)
        )
        return f"Context:\n{formatted_chunks}\n\nQuestion: {query}"


_REGISTRY: dict[str, PromptTemplate] = {
    "v0": PromptTemplate(version="v0", system_prompt=_V0_SYSTEM),
    "v1": PromptTemplate(version="v1", system_prompt=_V1_SYSTEM),
}


def get_template(
    version: str = CURRENT_VERSION,
    domain: str = DEFAULT_DOMAIN,
) -> PromptTemplate:
    """Return the :class:`PromptTemplate` for the requested *version* and *domain*.

    When *domain* is ``"general"`` the base versioned template is returned
    unchanged.  For any other domain the domain persona is prepended to the
    bilingual RAG rules block, producing a specialised system prompt that
    still enforces context-only answering, language detection, and citation.

    Args:
        version: Template version string (e.g. ``"v0"``, ``"v1"``).
            Defaults to :data:`CURRENT_VERSION`.
        domain: Domain specialisation key registered in :data:`DOMAIN_PROMPTS`
            (e.g. ``"general"``, ``"tourism"``).
            Defaults to :data:`DEFAULT_DOMAIN`.

    Returns:
        The matching :class:`PromptTemplate`, built on the fly for non-general
        domains.

    Raises:
        ValueError: If *version* is not registered or *domain* is unknown.
    """
    base = _REGISTRY.get(version)
    if base is None:
        raise ValueError(
            f"Unknown prompt version '{version}'. "
            f"Available versions: {sorted(_REGISTRY)}"
        )

    if domain not in DOMAIN_PROMPTS:
        raise ValueError(
            f"Unknown domain '{domain}'. "
            f"Available domains: {sorted(DOMAIN_PROMPTS)}"
        )

    domain_persona = DOMAIN_PROMPTS[domain]
    if not domain_persona:
        return base

    system_prompt = domain_persona + "\n\n" + _V1_DOMAIN_RULES
    return PromptTemplate(version=version, system_prompt=system_prompt)
