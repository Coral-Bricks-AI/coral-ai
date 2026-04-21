# Changelog

All notable changes to `coralbricks-connectors` are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-21

### Changed

- **Package renamed** from `coralbricks-airbyte` to
  `coralbricks-connectors`. The Airbyte reader moves under the
  `coralbricks.connectors.airbyte` subpackage; the umbrella package is
  now the home for all Coral Bricks data connectors (Airbyte today,
  PyAirbyte + direct source readers next).

### Migration

```diff
- pip install coralbricks-airbyte
+ pip install coralbricks-connectors

- from coralbricks.airbyte import read_airbyte_output
+ from coralbricks.connectors.airbyte import read_airbyte_output
```

No behaviour changes. The reader API, dict shape, and fixture format
are unchanged from 0.1.0.

## [0.1.0] - 2026-04-20

Initial open-source release (as `coralbricks-airbyte`).

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
