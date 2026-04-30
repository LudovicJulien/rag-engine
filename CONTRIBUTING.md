# Contributing

## Commit Convention

This project follows [Conventional Commits](https://www.conventionalcommits.org/),
with additional types inspired by the [Angular commit convention](https://github.com/angular/angular/blob/main/contributing-docs/commit-message-guidelines.md).

| Type       | When to use                          |
|------------|--------------------------------------|
| `feat`     | New feature                          |
| `fix`      | Bug fix                              |
| `docs`     | Documentation only                   |
| `chore`    | Maintenance (config, dependencies)   |
| `test`     | Adding or updating tests             |
| `refactor` | Code change without behavior change  |
| `ci`       | GitHub Actions, pipelines            |

Examples:
```
feat: add reranker score logging for observability
fix: handle empty document list in chunker
chore: configure pyproject.toml with project metadata and dependencies
```

## Branch Strategy

- `main` → always stable
- `feature/xxx` → one branch per feature, merged via PR

> Direct commits to `main` are allowed for repository 
> setup and documentation only.

Examples:
```
feature/json-ingestion
feature/semantic-chunking
feature/qdrant-vector-store
```

## Development Setup

```bash
make install
```

## Running Tests

```bash
make test
```
