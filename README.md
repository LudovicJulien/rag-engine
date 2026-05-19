# RAG Engine

> **A production-first RAG system with native observability, modular architecture, and agent-ready design.**

RAG Engine is a production-oriented backend system for building, debugging, and evaluating
Retrieval-Augmented Generation pipelines with full observability and modular components.

![Version](https://img.shields.io/badge/version-0.6.0-brightgreen)
![CI](https://img.shields.io/github/actions/workflow/status/LudovicJulien/rag-engine/ci.yml?branch=main&label=CI)
![Tests](https://img.shields.io/badge/tests-1026-blue)
![Coverage](https://img.shields.io/badge/status-active%20development-orange)
![Python](https://img.shields.io/badge/python-3.11+-green)
![License](https://img.shields.io/badge/license-GPL--3.0-blue)

---

## Why This Exists

Most RAG frameworks (LangChain, LlamaIndex) are powerful but:

- Hard to debug when retrieval quality degrades
- Too abstract for production troubleshooting
- Lack native observability — you don't know *why* an answer is wrong
- Require external tooling to measure anything

RAG Engine focuses on:

- **Full transparency** of every retrieval and generation step
- **Production debugging** — CLI-first, scriptable, exit-code driven
- **Simple and predictable architecture** — no magic, no hidden abstractions
- **Developer-first experience** — Rich terminal output, JSON mode for CI, structured diagnostics

---

## Highlights

- **Hybrid retrieval** — BM25 sparse + SentenceTransformer dense vectors, fused at query time via Reciprocal Rank Fusion (RRF), stored as named vectors in Qdrant
- **Multi-provider LLM** — swap between Ollama (local), Anthropic, and HuggingFace via a single env variable; OpenAI and Gemini stubs ready
- **Embedding cache** — MD5-keyed file cache per model config, transparent to callers
- **CLI-first developer experience** — 6 production-grade commands with Rich output, JSON mode, and actionable exit codes
- **Full pipeline observability** — Langfuse tracing (embedding span, retrieval span, generation span, RAGAS score attachment) — in roadmap
- **Plug-and-play data sources** — abstract ingestion interface; pre-chunked JSON adapter ships out of the box
- **1026 unit and integration tests** — every layer tested in isolation with real domain models

---

## Architecture

```
    Data Source
         │
         ▼
┌─────────────────┐
│ Ingestion Layer │  Pre-chunked JSON → Chunk + ChunkMetadata
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │  Dense (multilingual-e5-large) + Sparse (BM25) + MD5 cache
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │  Qdrant — named vectors, HNSW index, metadata filtering
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Retrieval     │  Hybrid RRF fusion (dense + sparse) — score threshold + top-k
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Re-ranking    │  CrossEncoder (ms-marco-MiniLM) — NoOp default, plug-in ready
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Generation │  Ollama · Anthropic · HuggingFace — language-aware prompt templates
└────────┬────────┘
         │
         ▼
  Answer + Sources + trace_id
         │
         ▼
┌─────────────────┐
│  Observability  │  Langfuse traces — latency, tokens, RAGAS scores attached per trace
└─────────────────┘
```

**Key design patterns:**

- Every layer implements an abstract base class (`EmbeddingModel`, `LLMGenerator`, `Retriever`, …). Swap implementations without touching the pipeline.
- `Chunk` and `ChunkMetadata` are the single currency across all layers — infrastructure never leaks into domain models.
- `NoOp` implementations ship alongside every pluggable layer: `NoOpReranker`, `NoOpTracer` — zero overhead when a feature is off.
- A single pydantic-settings `Settings` class reads from `.env`. One config, all layers.

---

## CLI — Developer Interface

The CLI is the primary interface for operating the pipeline locally.
Every command supports `--output json` for scripting and CI integration.

```bash
# Index a document set
python -m src.cli ingest data/chunks.json

# Re-index from scratch
python -m src.cli ingest data/chunks.json --reset

# Ask a question (Rich panel output)
python -m src.cli query "What is hybrid retrieval?"

# Ask — pipe-friendly JSON output
python -m src.cli query "What is BM25?" --output json

# Override retrieval parameters
python -m src.cli query "How does RRF work?" --top-k 10 --score-threshold 0.5

# Pre-warm the pipeline (validates all components)
python -m src.cli up

# Check infrastructure connectivity (Qdrant + LLM)
python -m src.cli health
python -m src.cli health --output json   # for CI health checks

# Full diagnostics — config + infra + collection stats
python -m src.cli doctor

# Inspect active configuration (API keys masked)
python -m src.cli config
```

**Exit codes:** `0` success · `1` user/config error · `2` infrastructure error.
Scripts can distinguish a missing BM25 cache from a downed Qdrant instance.

**Sample output:**

```
╭─ Answer ─────────────────────────────────────────────────────────────────────╮
│ Hybrid retrieval combines dense vector search with BM25 sparse retrieval,    │
│ fusing results at query time via Reciprocal Rank Fusion (RRF)...             │
╰──────────────────────────────────────────────────────────────────────────────╯
Sources          Model                            Language   Chunks   Tokens
chunk-042        intfloat/multilingual-e5-large   en         3        312
chunk-017
```

```
── Connectivity ────────────────────────────────────────────
Qdrant       ✓  connected at localhost:6333
LLM          ✓  ollama reachable at localhost:11434

── Configuration ───────────────────────────────────────────
BM25 cache   ✓  file found at ~/.cache/rag/bm25.pkl
Collection   ✓  rag-collection exists

── Statistics ──────────────────────────────────────────────
Collection   142 points
BM25 vocab   4821 terms

All checks passed.
```

---

## What's Built

### Phase 1 — Ingestion
- [x] Abstract `DataIngestionPipeline` interface
- [x] `JSONChunkIngestionPipeline` — pre-chunked JSON adapter
- [x] `Chunk` + `ChunkMetadata` domain models

### Phase 2 — Embeddings
- [x] Abstract `EmbeddingModel` interface
- [x] Dense embeddings — `intfloat/multilingual-e5-large` via SentenceTransformers
- [x] Sparse embeddings — BM25 (`rank_bm25`), fit on corpus, persistent `.pkl` cache
- [x] `HybridEmbedder` — fuses dense + sparse into named vector payloads
- [x] MD5-keyed embedding cache — scoped per model config, transparent to callers

### Phase 3 — Vector Store
- [x] Abstract `VectorStore` interface
- [x] `QdrantVectorStore` — named vectors, HNSW index, metadata filtering, upsert + search

### Phase 4 — Retrieval
- [x] `DenseRetriever` — top-k + score threshold, delegates RRF fusion to Qdrant
- [x] Hybrid search (BM25 + dense) fused via Reciprocal Rank Fusion at query time

### Phase 5 — Generation
- [x] Abstract `LLMGenerator` interface + factory pattern
- [x] `OllamaGenerator` — local models (gemma3:4b, llama3, …)
- [x] `AnthropicGenerator` — Claude models via Anthropic API
- [x] `HuggingFaceGenerator` — inference API
- [x] Language detection — auto-selects prompt template (FR/EN/…)
- [x] Context window management — token budget enforcement
- [x] Structured `RAGResult` with answer, sources, model, tokens, language

### Phase 6 — Pipeline Orchestration
- [x] `IngestionPipeline` — fit BM25 → embed batch → upsert → persist
- [x] `RAGPipeline` — embed query → retrieve → generate → `RAGResult`
- [x] Pipeline singleton — dense model loaded once per process
- [x] `RAGPipeline.build(settings)` classmethod

### Phase 7 — CLI
- [x] `ingest` — index documents, `--reset` flag, text/JSON output
- [x] `query` — ask questions, `--top-k`, `--score-threshold`, text/JSON output
- [x] `up` — pre-warm pipeline, reports all components
- [x] `health` — infra connectivity check, CI-ready JSON output
- [x] `doctor` — full diagnostics (config + infra + collection stats)
- [x] `config` — display active settings with masked secrets
- [x] Rich terminal output (panels, tables, spinners)
- [x] Structured exit codes (0 / 1 / 2)

---

## Roadmap

### Phase 8 — Re-ranking *(next)*
- [ ] `Reranker` abstract interface + `NoOpReranker` default
- [ ] `CrossEncoderReranker` — `ms-marco-MiniLM-L-6-v2`
- [ ] Decoupled `top_k` (retriever pool) from `reranker_top_n` (final context)
- [ ] Wire into `RAGPipeline` between retrieval and generation

### Phase 9 — Evaluation (RAGAS + MLflow)
- [ ] `EvaluationSample`, `SampleScore`, `EvaluationResult` dataclasses
- [ ] `RAGASEvaluator` — faithfulness, answer relevancy, context precision, context recall
- [ ] `MLflowTracker` — log aggregate metrics + per-sample artifact
- [ ] `EvalRunner` — orchestrates dataset → RAG → evaluation → MLflow
- [ ] `evaluate` CLI subcommand
- [ ] Separate LLM judge from generation LLM (Claude Haiku by default)

### Phase 10 — Observability (Langfuse)
- [ ] `PipelineTracer` abstract interface + `NoOpTracer` (zero overhead default)
- [ ] `LangfuseTracer` — trace per query: embedding span, retrieval span, generation span
- [ ] RAGAS scores attached to traces a posteriori from `EvalRunner`
- [ ] `trace_id` propagated through `RAGResult`
- [ ] Inject tracer into `RAGPipeline` and `IngestionPipeline`

### Phase 11 — FastAPI REST API
- [ ] `POST /ingest` — trigger ingestion from uploaded file
- [ ] `POST /query` — query endpoint returning `RAGResult` as JSON
- [ ] `GET /health` — maps to `check_health()` output
- [ ] OpenAPI documentation auto-generated

### Phase 12 — Infrastructure
- [ ] Docker Compose — Qdrant + API + optional Langfuse self-hosted
- [ ] `Dockerfile` for the API service
- [ ] GitHub Actions CI already in place (`lint → test`)

---

## Tech Stack

| Layer         | Technology                                       | Status      |
|---------------|--------------------------------------------------|-------------|
| Language      | Python 3.11+                                     | Active      |
| Dense Embed   | SentenceTransformers `multilingual-e5-large`     | Built       |
| Sparse Embed  | BM25 (`rank_bm25`)                               | Built       |
| Vector Store  | Qdrant (named vectors, HNSW, RRF)                | Built       |
| Generation    | Ollama · Anthropic · HuggingFace                 | Built       |
| CLI           | Typer + Rich                                     | Built       |
| Re-ranking    | CrossEncoder (`ms-marco-MiniLM`)                 | Roadmap     |
| Evaluation    | RAGAS + MLflow                                   | Roadmap     |
| Observability | Langfuse                                         | Roadmap     |
| API           | FastAPI                                          | Roadmap     |
| Config        | pydantic-settings + `.env`                       | Built       |
| CI/CD         | GitHub Actions                                   | Built       |
| Packaging     | `pyproject.toml`                                 | Built       |

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Qdrant](https://qdrant.tech) running locally (or via Docker)
- One of the supported LLM providers (see table below)
- **Windows only**: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) — select "Desktop development with C++"

### LLM Providers

| Provider           | `LLM_PROVIDER`  | `LLM_BASE_URL`                              | `LLM_API_KEY`  |
|--------------------|-----------------|---------------------------------------------|----------------|
| **Ollama** (local) | `ollama`        | `http://localhost:11434`                    | *(empty)*      |
| **Anthropic**      | `anthropic`     | `https://api.anthropic.com`                 | Anthropic key  |
| **HuggingFace**    | `huggingface`   | `https://api-inference.huggingface.co`      | HF token       |
| **OpenAI**         | `openai`        | `https://api.openai.com/v1`                 | *(roadmap)*    |
| **Gemini**         | `gemini`        | `https://generativelanguage.googleapis.com` | *(roadmap)*    |

### Installation

```bash
git clone https://github.com/LudovicJulien/rag-engine.git
cd rag-engine
make install
```

### Environment

```bash
cp .env.example .env
# Edit .env — set LLM_PROVIDER, LLM_BASE_URL, and optionally LLM_API_KEY
```

### Quick start

```bash
# 1. Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 2. Index your documents
python -m src.cli ingest data/your_chunks.json

# 3. Query
python -m src.cli query "Your question here"
```

---

## Testing

```bash
make test        # full suite (1026 tests)
make lint        # black + isort + flake8 + mypy
```

Run a specific file or class:

```bash
pytest tests/embeddings/test_hybrid_embedder.py
pytest tests/cli/test_query_command.py::TestQueryCommandExitCodes
```

Tests mirror `src/` under `tests/`. See [docs/testing.md](docs/testing.md) for conventions.

- Unit tests mock all external dependencies (Qdrant, LLM, filesystem)
- Integration tests (suffixed `Integration`) hit real models and services — skipped by default in CI unless services are available
- CLI tests use `typer.testing.CliRunner` — no network calls

---

## License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.
