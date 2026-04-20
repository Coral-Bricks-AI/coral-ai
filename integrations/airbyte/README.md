# coralbricks-airbyte

> Read Airbyte destination output into plain `list[dict]` records ready
> for [`coralbricks-context-prep`](../../context_prep).

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

`coralbricks-airbyte` is the bridge between Airbyte — the 600+ connector
catalog — and the Coral Bricks OSS data-prep libraries. It reads the
JSONL files that Airbyte's Local JSON and S3 destinations write, and
hands you dict records that plug straight into `chunk → embed → enrich
→ hydrate`.

It is deliberately tiny: one function, zero hard dependencies, no state.

## Install

```bash
pip install coralbricks-airbyte
```

## 30-second quickstart

Point [Airbyte OSS](https://docs.airbyte.com/using-airbyte/getting-started/oss-quickstart/)
at any source — HackerNews, Slack, Postgres, Google Drive — with a
**Local JSON** or **S3 JSONL** destination. Then:

```python
from coralbricks.airbyte import read_airbyte_output

records = read_airbyte_output(
    "/tmp/airbyte_local/hackernews/",  # dir (recursively walked) or single .jsonl
    stream="stories",                  # optional filename substring filter
    text_field="title",                # which source column becomes `text`
    id_field="id",                     # which source column becomes `id`
)
# [{"id": "39678900", "text": "...", "source": "stories", "metadata": {...}}, ...]
```

Feed those records straight into `context_prep`:

```python
from coralbricks.context_prep import chunk, embed

chunks  = chunk(records, strategy="sliding_token", target_tokens=512)
vectors = embed(chunks, model="openai:text-embedding-3-large")
```

See [`context_prep/examples/airbyte_rag.py`](../../context_prep/examples/airbyte_rag.py)
for the full Airbyte → chunk → embed → enrich → hydrate → DuckDB vss +
duckpgq retrieval demo.

## API

```python
read_airbyte_output(
    path,                     # str | Path — file or directory (recursively walked)
    *,
    stream=None,              # optional filename substring filter
    text_field="text",        # str | list[str] | callable(dict) -> str
    id_field=None,            # str | callable(dict) -> str | None
) -> list[dict]
```

Each returned dict has:

| Key        | Type   | Source                                                                                   |
| ---------- | ------ | ---------------------------------------------------------------------------------------- |
| `id`       | `str`  | From `id_field`, or Airbyte's `_airbyte_raw_id` / `_airbyte_ab_id`                       |
| `text`     | `str`  | From `text_field` (column name, list of column names concatenated, or callable)          |
| `source`   | `str`  | `stream` arg, or the filename stem with trailing digit-segments stripped                 |
| `metadata` | `dict` | The full source row + a nested `_airbyte` block with the envelope's metadata             |

## Supported Airbyte formats

Airbyte's Local JSON destination wraps each source row like:

```json
{"_airbyte_ab_id": "...", "_airbyte_emitted_at": 1, "_airbyte_data": { ...row... }}
```

Newer destinations (S3 JSONL, destinations-v2) add `_airbyte_raw_id`,
`_airbyte_extracted_at`, `_airbyte_generation_id`, and `_airbyte_meta`
but still nest the source data under `_airbyte_data`. Both formats are
handled transparently.

If a destination ever emits flat top-level columns without
`_airbyte_data`, the reader falls back to treating all non-metadata
keys as the source row.

## What this does NOT do

- **Run Airbyte for you.** Airbyte is a separate process (Docker
  Compose, `abctl`, or Airbyte Cloud) that you configure independently.
- **Read directly from S3.** Sync to local first
  (`aws s3 sync s3://bucket/prefix /tmp/ab/`) and point the reader at
  the local directory. Native `s3://` support is a candidate for 0.2.0.
- **Parse Airbyte's Parquet or CSV destinations.** JSONL only — the
  lowest-common-denominator format every Airbyte destination supports.

## License

Apache 2.0 © Coral Bricks AI.
