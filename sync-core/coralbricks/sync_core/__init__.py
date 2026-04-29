"""Coral Bricks Airbyte sync core — Docker runner + S3 writer.

Shared between the OSS CLI (`coralbricks sync`) and the closed-source
ECS supervisor that wraps each sync as a one-shot container task.
Both run the exact same protocol parser and write the exact same S3
envelope, so downstream consumers only need to handle one shape.
"""

from __future__ import annotations

__version__ = "0.1.0"
