# src/pipeline/ingest_pipeline.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IngestResult:
    source: str
    collection_name: str
    chunks_loaded: int
    chunks_upserted: int
    chunks_failed: int
    bm25_cache_path: str
