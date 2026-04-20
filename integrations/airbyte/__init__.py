"""Read Airbyte destination output as coralbricks.context_prep records.

Public API::

    from coralbricks.airbyte import read_airbyte_output

    records = read_airbyte_output(
        "/tmp/airbyte_local/hackernews/",
        stream="stories",
        text_field="title",
    )

See the package README for end-to-end examples.
"""

from __future__ import annotations

from .reader import read_airbyte_output

__all__ = ["read_airbyte_output"]
__version__ = "0.1.0"
