"""Record normalization helper.

Coral Bricks prep operates on plain ``dict`` records — the universal
record shape across pandas, duckdb, parquet readers, message queues,
and JSON APIs. We do not ship a custom ``Document`` class; doing so
forces every caller to convert in and out for no behavioural gain.

A "record" is a dict with at minimum::

    {"id": str, "text": str}

Optional fields the prep verbs use when present:

    "source":   str  - origin URI / path / row-id (free form).
    "metadata": dict - extra per-record fields (extractors write into
                       ``metadata["extractions"]``; cleaners write
                       ``metadata["cleaned_by"]`` etc).

This module provides a single helper, :func:`normalize_records`, that
coerces strings / mixed inputs into the canonical dict shape so each
verb does not need to repeat the same boilerplate.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any


def new_record_id(prefix: str = "rec") -> str:
    """Return a fresh stable id for a record (used when input has none)."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def normalize_records(items: Iterable[Any]) -> list[dict[str, Any]]:
    """Coerce strings / dicts into the canonical record shape.

    - ``str`` becomes ``{"id": ..., "text": s, "source": None, "metadata": {}}``.
    - ``dict`` is shallow-copied; ``id`` is generated if missing; ``text``
      is coerced to ``str``; ``metadata`` is coerced to a dict.
    - Anything else is stringified into the ``text`` field.

    The returned list is independent: callers may mutate the returned
    dicts (e.g. enrichers writing extractions into ``metadata``) without
    affecting the input.
    """
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            rec = dict(item)
            rec.setdefault("id", new_record_id())
            rec["id"] = str(rec["id"])
            rec["text"] = str(rec.get("text", ""))
            rec.setdefault("source", None)
            md = rec.get("metadata")
            rec["metadata"] = dict(md) if isinstance(md, dict) else {}
            out.append(rec)
        elif isinstance(item, str):
            out.append(
                {
                    "id": new_record_id(),
                    "text": item,
                    "source": None,
                    "metadata": {},
                }
            )
        else:
            out.append(
                {
                    "id": new_record_id(),
                    "text": str(item),
                    "source": None,
                    "metadata": {},
                }
            )
    return out


__all__ = ["normalize_records", "new_record_id"]
