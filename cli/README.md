# coralbricks

> Pip-installable CLI for Coral Bricks AI — connect 600+ data sources and
> run syncs on your own machine, straight into managed storage.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

```bash
pip install coralbricks
```

## 30-second quickstart

```bash
coralbricks login             # paste an API key from coralbricks.ai/settings/api-keys
coralbricks sources           # list available connectors
coralbricks connect notion    # OAuth in your browser, or prompt for an API key
coralbricks sync notion       # pull Airbyte image, run locally, upload to managed S3
coralbricks connections       # list what you've connected
coralbricks runs notion       # recent sync history for this source
```

## Commands

| Command | What it does |
| --- | --- |
| `coralbricks login` | Prompts for an API key (or `--api-key ak_…`), validates against the backend, stores in `~/.coralbricks/config.json` (mode 0600) |
| `coralbricks logout` | Removes the stored API key |
| `coralbricks whoami` | Re-validates the stored key and prints the logged-in user |
| `coralbricks sources` | Lists connectors available on your account (name, auth type) |
| `coralbricks connect <source>` | Connects a data source — OAuth in the browser (loopback pattern) or interactive API-key prompts. One connection per source; re-running refreshes credentials in place. |
| `coralbricks disconnect <source>` | Removes a previously-configured connection for a source. Prompts for confirmation (use `-y` to skip). |
| `coralbricks sync <source>` | Pulls the Airbyte source Docker image, runs it on your machine, and uploads records (gzipped JSONL, raw Airbyte Protocol shape) to our managed S3. Credentials are STS session creds scoped to exactly this run's prefix for 1 hour. Requires Docker Desktop / Engine running. |
| `coralbricks connections` | Lists the connections you've already set up |
| `coralbricks runs <source>` | Shows recent sync runs for a source (status, records, bytes, timestamp). |

## Configuration

| Env var | Purpose | Default |
| --- | --- | --- |
| `CORALBRICKS_API_KEY` | Overrides the stored key (useful in CI) | — |
| `CORALBRICKS_SERVER_URL` | Override the backend URL | `https://backend.coralbricks.ai` |
| `CORALBRICKS_ALLOWED_BUCKETS` | Comma-separated S3 buckets the CLI will accept as a sync destination. Defaults to the prod managed bucket; set this only if you're testing against a non-prod backend. | `coralbricks-connectors` |

## Development

```bash
cd cli
pip install -e '.[dev]'
pytest
```

Tests use `responses` to mock the backend — no live services required.

## License

Apache 2.0 © Coral Bricks AI.
