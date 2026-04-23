# coralbricks

> Pip-installable CLI for Coral Bricks AI — connect data sources from any
> of 600+ Airbyte connectors and run syncs on your own machine.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

```bash
pip install coralbricks
```

## 30-second quickstart

```bash
coralbricks login             # paste an API key from coralbricks.ai/settings/api-keys
coralbricks sources           # list available connectors
coralbricks connect notion    # OAuth in your browser (coming in v0.2)
coralbricks sync <id>         # pull Airbyte Docker, write JSONL to managed S3 (coming in v0.2)
```

## Status

**v0.1 (current, work-in-progress)** ships the auth + discovery surface:

| Command | What it does |
| --- | --- |
| `coralbricks login` | Prompts for an API key (or `--api-key ak_…`), validates against the backend, stores in `~/.coralbricks/config.json` (mode 0600) |
| `coralbricks logout` | Removes the stored API key |
| `coralbricks whoami` | Re-validates the stored key and prints user id / email |
| `coralbricks sources` | Lists connectors available on your account (name, auth type) |
| `coralbricks connections` | Lists the connections you've already set up |

**v0.2 (in progress)** — `coralbricks connect <source>` with browser-brokered OAuth loopback and API-key prompts.

**v0.3 (in progress)** — `coralbricks sync <id>` runs the source's Airbyte Docker image on your laptop and uploads JSONL to our managed S3 bucket using scoped STS credentials.

## Configuration

| Env var | Purpose | Default |
| --- | --- | --- |
| `CORALBRICKS_API_KEY` | Overrides the stored key (useful in CI) | — |
| `CORALBRICKS_SERVER_URL` | Override the backend URL | `https://backend.coralbricks.ai` |

## Development

```bash
cd cli
pip install -e '.[dev]'
pytest
```

Tests use `responses` to mock the backend — no live services required.

## License

Apache 2.0 © Coral Bricks AI.
