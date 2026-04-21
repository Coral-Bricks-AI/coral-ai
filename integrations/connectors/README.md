# coralbricks-connectors

> Data connectors for Coral Bricks — readers and ingestion helpers that
> turn external-source exports into `list[dict]` records ready for
> [`coralbricks-context-prep`](../../context_prep).

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

`coralbricks-connectors` is the umbrella package for pulling data into
the Coral Bricks OSS pipeline. Every connector normalises its source's
wire format to the same record shape
(`{id, text, source, metadata}`) so the downstream
`chunk → embed → enrich → hydrate` steps don't care where the data
came from.

## Install

```bash
pip install coralbricks-connectors
```

## Available connectors

| Submodule                                | Source                                                                                     | Docs                           |
| ---------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------ |
| `coralbricks.connectors.airbyte`         | [Airbyte](https://airbyte.com) destination output (Local JSON / S3 JSONL)                  | [README](airbyte/)             |

## 30-second quickstart (Airbyte)

Point [Airbyte OSS](https://docs.airbyte.com/using-airbyte/getting-started/oss-quickstart/)
at any source with a **Local JSON** or **S3 JSONL** destination, then:

```python
from coralbricks.connectors.airbyte import read_airbyte_output

records = read_airbyte_output(
    "/tmp/airbyte_local/hackernews/",
    stream="stories",
    text_field="title",
    id_field="id",
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

## License

Apache 2.0 © Coral Bricks AI.
