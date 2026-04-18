# Contributing to coralbricks-context-prep

Thanks for considering a contribution!

## Quick start

```bash
git clone https://github.com/Coral-Bricks-AI/coral-ai.git
cd coral-ai/context_prep
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Scope

Before opening a PR, please read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
The library is intentionally small. PRs that broaden scope (file
loaders, an orchestrator, a vector-store client, a query layer) will
generally be politely declined — those concerns belong in adjacent
tools.

PRs that are gladly accepted:

- New chunker strategies.
- New embedder backends (must follow `BaseEmbedder` and lazy-import
  the heavy dependency).
- New regex extractors with broad applicability.
- New triple extractors / graph aggregations.
- Bug fixes, type-hint fixes, doc improvements.
- New examples in `examples/`.

## Development

- **Style:** `ruff format` + `ruff check`.
- **Types:** `mypy coralbricks/prep`.
- **Tests:** `pytest`. New code needs tests. Optional-dep tests should
  use `pytest.importorskip(...)`.
- **Imports:** every heavy or optional dependency must be imported
  lazily inside the function/method that needs it, not at module top
  level.

## Commit messages

Conventional commits preferred (`feat:`, `fix:`, `docs:`, `chore:`,
`refactor:`, `test:`, `perf:`).

## Releases

Maintainers cut releases by:

1. Bumping `version` in `pyproject.toml` and `coralbricks/context_prep/__init__.py`.
2. Adding a `CHANGELOG.md` section.
3. Tagging `prep-vX.Y.Z` on `main`.
4. CI publishes to PyPI.
