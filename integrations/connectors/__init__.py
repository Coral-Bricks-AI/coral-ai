"""Coral Bricks data connectors.

Umbrella package for the readers and ingestion helpers that pull data
from external sources into the shape ``coralbricks.context_prep``
expects (``{id, text, source, metadata}`` dicts). Each connector lives
under its own submodule:

- ``coralbricks.connectors.airbyte`` — reads Airbyte destination output
  (Local JSON / S3 JSONL).

Additional connectors (PyAirbyte live sync, direct Slack / Notion
readers, …) are registered as sibling submodules.
"""

__version__ = "0.2.0"
