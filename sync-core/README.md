# coralbricks-sync-core

Shared Airbyte sync core for [Coral Bricks](https://coralbricks.ai) — a Docker
runner that speaks the Airbyte Protocol and a gzipped-JSONL S3 writer that
emits the Coral Bricks per-record envelope.

This package is the source of truth for two surfaces that must produce
byte-identical S3 output:

- **`coralbricks` CLI** (`coralbricks sync`) — runs Docker on a user laptop.
- **Coral Bricks ECS supervisor** (closed-source) — runs the same protocol
  parser against an Airbyte connector container's CloudWatch log stream.

Both pip-install `coralbricks-sync-core` and call into the same
`runner` / `s3` modules, so downstream consumers only ever see one S3 layout
and one record envelope.

## Install

```bash
pip install coralbricks-sync-core
```

Runtime deps: `docker>=7.0`, `boto3>=1.34`. Python 3.10+.

## What's in the box

```python
from coralbricks.sync_core import runner, s3

# Spawn an Airbyte source container, parse RECORD/STATE/LOG/TRACE messages.
runner.run_read(docker_client, image, config, configured_catalog, handlers)

# Buffer records per stream, gzip in memory, upload one part per stream.
writer = s3.ScopedS3Writer(bucket=..., key_prefix=..., sts=...)
writer.write_record(stream, data, emitted_at)
writer.close()  # → upload summary
```

The S3 envelope written per line is the inner Airbyte Protocol RECORD
payload verbatim — `{stream, data, emitted_at}` — no Coral envelope layer.

## License

Apache-2.0.
