# rag-engine

> Modular RAG engine — plug in any data source, get semantic search out of the box.

![Status](https://img.shields.io/badge/status-in%20development-orange)
![License](https://img.shields.io/badge/license-GPL--3.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)

---

## Overview

**rag-engine** is a generic, modular Retrieval-Augmented Generation engine built for reusability.
Modular by design, plug in your data source, configure your models, and get 
semantic search with LLM-powered answers out of the box.


---

## Architecture

```
    Data Source
         │
         ▼
┌─────────────────┐
│ Ingestion Layer │  Pre-chunked JSON 
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │  Dense (SentenceTransformers) + Sparse (BM25) → Hybrid
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │  Qdrant with HNSW index + metadata filtering
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Retrieval     │  Dense / Sparse / Hybrid (RRF fusion) + MMR
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Re-ranking    │  CrossEncoder (ms-marco-MiniLM) + NoOp for A/B
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Generation │   Ollama (Gemma3:4b)
└────────┬────────┘
         │
         ▼
  Answer + Sources
```

---

## Planned Features

- [ ] Ingestion pipeline (pre-chunked JSON)
- [ ] Hybrid embeddings (dense + sparse BM25)
- [ ] Vector store with metadata filtering (Qdrant)
- [ ] Hybrid retrieval with RRF fusion + MMR
- [ ] Cross-encoder re-ranking
- [ ] LLM generation (Ollama gemma3:4b  )
- [ ] Evaluation framework (RAGAS metrics)
- [ ] FastAPI REST API
- [ ] Docker + CI/CD
- [ ] Observability (Langfuse, MLflow)

---

## Tech Stack

| Layer         | Technology                 |
|---------------|----------------------------|
| Language      | Python 3.11+               |
| Embeddings    | SentenceTransformers, BM25 |
| Vector Store  | Qdrant                     |
| LLM           | gemma3:4b                  |
| API           | FastAPI                    |
| Evaluation    | RAGAS, MLflow              |
| Observability | Langfuse                   |
| CI/CD         | GitHub Actions             |
| Packaging     | pyproject.toml, Docker     |

---

## Project Structure *(planned)*

```
rag-engine/
├── src/
│   ├── embeddings/       # Embedding models
│   ├── evaluation/       # RAGAS metrics and runner
│   ├── generation/       # LLM generators
│   ├── ingestion/        # Data ingestion adapters
│   ├── pipeline/         # RAGPipeline and RAGQuery
│   ├── reranking/        # Re-ranking models
│   ├── retrieval/        # Retrieval strategies
│   ├── shared/           # Shared enums, types, and utilities
│   └── vector_store/     # Qdrant client wrapper 
├── tests/
├── demo_data/            # Sample documents for local dev
├── golden_dataset/       # Q/A pairs for evaluation
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EVALUATION.md
│   ├── DEPLOYMENT.md
│   └── adr/              # Architecture Decision Records
├── .github/
│   └── workflows/
│       └── ci.ym
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── pyproject.toml
└── docker-compose.yml
```

---

## Getting Started

>Installation, configuration, and usage instructions will be added as 
development progresses.

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) with `gemma3:4b` pulled locally

### Installation

```bash
git clone https://github.com/ton-username/rag-engine.git
cd rag-engine
make install
```

### Environment

```bash
cp .env.example .env
# Edit .env with your values
```

---

## Design Decisions *(coming soon)*

A dedicated section covering key architectural choices — embedding strategy, chunking approach, retrieval fusion — will be added once the implementation is complete, backed by benchmark results.

---

## License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.