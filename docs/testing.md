# Testing Guide

Tests are written to document behavior, not implementation. Each test is a statement of fact about the system — readable without context, reproducible without side effects.

---

## Structure

Tests live in `tests/`, mirroring the `src/` hierarchy:

```
tests/
├── embeddings/   → src/embeddings/
├── ingestion/    → src/ingestion/
├── pipeline/     → src/pipeline/
└── shared/       → src/shared/
```

---

## Test classes

Each file groups tests into classes by logical concern. The class name encodes both the subject and the scenario:

```
TestBM25SparseEmbedderFit                  # class + method
TestSettingsFromEnv                        # class + scenario
TestJSONChunkIngestionPipelineIntegration  # integration scope
```

One class = one concern. Each test within the class covers exactly one behavior.

---

## Naming

```python
def test_fit_empty_corpus_raises(self) -> None: ...
def test_embed_text_rare_term_higher_score_than_common_term(self) -> None: ...
def test_log_level_lowercase_normalized_to_uppercase(self) -> None: ...
```

Pattern: `test_<subject>_<expected_outcome>`. No verbs like "should" or "must" — the name states the fact directly.

---

## Docstrings

Only written when the *why* is non-obvious — a hidden constraint, a subtle invariant, or model-specific behavior:

```python
def test_embed_query_adds_e5_prefix(self) -> None:
    """e5 models require 'query: ' prefix for queries."""

def test_embed_batch_preserves_order(self) -> None:
    """Order of returned vectors must match order of input texts."""
```

Obvious tests have no docstring.

---

## Fixtures and factories

Fixtures are defined at module level, never inside a class. For models with many required fields, a module-level factory with sensible defaults removes repetition:

```python
def _make_chunk(**kwargs: Any) -> Chunk:
    defaults = {"chunk_id": "chunk-001", "text": "Hello world", ...}
    defaults.update(kwargs)
    return Chunk(**defaults)
```

Only override what the test actually cares about — the rest stays default.

---

## Mocking

External dependencies (ML models, HTTP clients) are mocked with `unittest.mock.patch`. Internal domain models are used directly — no mocking at domain boundaries:

```python
with patch("src.embeddings.sentence_transformer_embedder.SentenceTransformer") as mock_cls:
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 1024
    mock_cls.return_value = mock_model
```

The rule: mock what you don't own, use what you do.

---

## Assertions

| What to assert | Convention |
|---|---|
| Exception type + message | `pytest.raises(ValueError, match="...")` — always include `match=` to pin the error message |
| Float equality | `pytest.approx(...)` |
| Environment overrides | `monkeypatch.setenv(...)` — never mutate `os.environ` directly |
| Filesystem | `tmp_path` built-in fixture |
| Conditional skip | `pytest.skip("reason")` when a prerequisite is absent at runtime |

---

## Unit vs integration

Integration tests live in a dedicated `TestXxxIntegration` class and run against real files from `demo_data/`. They are kept separate so they can be scoped out in CI if needed.

Unit tests mock all I/O and external dependencies. Integration tests use nothing fake.
