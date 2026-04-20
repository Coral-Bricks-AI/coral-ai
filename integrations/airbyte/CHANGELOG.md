# Changelog

All notable changes to `coralbricks-airbyte` are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-20

Initial open-source release.

### Added
- `read_airbyte_output()` — reads JSONL files from Airbyte's Local JSON
  and S3 destinations, yields `list[dict]` records in
  `coralbricks.context_prep`'s shape (`id`, `text`, `source`,
  `metadata`).
- Dual-format support: older Local JSON
  (`_airbyte_ab_id` / `_airbyte_emitted_at`) and newer destinations-v2
  (`_airbyte_raw_id` / `_airbyte_extracted_at` / `_airbyte_meta` /
  `_airbyte_generation_id`). Plus a flat-column fallback if a
  destination ever emits columns at the top level.
- `text_field` accepts a column name, a sequence of column names
  (concatenated with spaces, skipping empty values), or a
  `callable(dict) -> str`.
- `id_field` accepts a column name or a `callable(dict) -> str`;
  defaults to the Airbyte envelope's raw id.
- Directory input walks `.jsonl` files recursively; `stream` filters
  by filename substring and also sets each record's `source`.
- Zero runtime dependencies (stdlib `json` only).
- Unit tests and checked-in HackerNews-style JSONL fixtures
  exercising both envelope formats.

### Layout

- PyPI distribution: `coralbricks-airbyte`. Importable as
  `coralbricks.airbyte`, part of the shared `coralbricks.*` PEP 420
  namespace used by every Coral Bricks library and the closed-source
  `coralbricks-platform` SDK.
