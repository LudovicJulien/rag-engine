# src/pipeline/rag_pipeline.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RAGResult:
    query: str
    answer: str
    sources: list[str]
    detected_language: str
    model: str
    chunks_retrieved: int
    tokens_used: int | None
