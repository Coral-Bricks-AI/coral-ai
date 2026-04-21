"""Read Airbyte destination output into list[dict] records.

Airbyte's Local JSON and S3 destinations both write JSON Lines where each
line is an envelope wrapping the source row. Older destinations use
``{_airbyte_ab_id, _airbyte_emitted_at, _airbyte_data}``; newer
destinations-v2 (S3 JSONL included) use ``{_airbyte_raw_id,
_airbyte_extracted_at, _airbyte_generation_id, _airbyte_meta,
_airbyte_data}``. In both cases the source columns are nested inside
``_airbyte_data``. This module handles both, plus a defensive
flat-column fallback if a destination ever emits columns at the top
level.

Output records follow ``coralbricks.context_prep``'s canonical shape:
``{"id": ..., "text": ..., "source": ..., "metadata": {...}}``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

__all__ = ["read_airbyte_output"]

# Envelope keys Airbyte sets on the outer JSON line. Kept here so the
# flat-column fallback knows what NOT to treat as source data.
_AIRBYTE_ENVELOPE_KEYS: frozenset[str] = frozenset(
    {
        "_airbyte_ab_id",
        "_airbyte_raw_id",
        "_airbyte_emitted_at",
        "_airbyte_extracted_at",
        "_airbyte_meta",
        "_airbyte_generation_id",
        "_airbyte_data",
    }
)

# Keys that map to the Airbyte record id, in preference order.
_ID_ENVELOPE_KEYS: tuple[str, ...] = ("_airbyte_raw_id", "_airbyte_ab_id")


def read_airbyte_output(
    path: str | Path,
    *,
    stream: str | None = None,
    text_field: str | Sequence[str] | Callable[[dict[str, Any]], str] = "text",
    id_field: str | Callable[[dict[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    """Read Airbyte JSONL destination output as plain dict records.

    Args:
        path: A single ``.jsonl`` file, or a directory walked recursively
            for ``*.jsonl`` files (sorted by path for deterministic order).
        stream: Optional filename filter. When ``path`` is a directory,
            only files whose name contains this substring are read. The
            value is also used verbatim as the ``source`` field on output
            records when provided; otherwise ``source`` is derived from
            the filename (trailing digit-only segments like
            ``stories_20260420_00001`` are stripped).
        text_field: How to produce each record's ``text``:

            - ``str``: name of a column in the source row.
            - sequence of ``str``: concatenate those columns, space-joined,
              skipping missing / empty values.
            - callable ``(source_row) -> str``: user-defined extractor.

        id_field: How to produce each record's ``id``:

            - ``str``: name of a column in the source row.
            - callable ``(source_row) -> str``: user-defined extractor.
            - ``None``: fall back to the Airbyte envelope's raw id
              (``_airbyte_raw_id`` first, then ``_airbyte_ab_id``).

    Returns:
        A ``list[dict]`` where each dict has ``id``, ``text``, ``source``,
        ``metadata`` — the record shape every
        ``coralbricks.context_prep`` verb already accepts. ``metadata``
        carries the full source row plus a nested ``_airbyte`` block
        with the envelope metadata (raw id, extraction timestamp,
        generation id, meta).

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: a JSONL line cannot be decoded, or a line's top-level
            value is not a JSON object.
        KeyError: an explicit ``id_field`` column is missing, or a row
            lacks any id when ``id_field`` is ``None``.
    """
    p = Path(path)
    files = _resolve_files(p, stream)
    if not files:
        return []

    records: list[dict[str, Any]] = []
    for file in files:
        file_stream = _stream_from_filename(file.stem)
        records.extend(
            _read_file(file, stream or file_stream, text_field, id_field)
        )
    return records


def _resolve_files(p: Path, stream: str | None) -> list[Path]:
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(p.rglob("*.jsonl"))
        if stream is not None:
            files = [f for f in files if stream in f.name]
        return files
    raise FileNotFoundError(f"path does not exist: {p}")


def _read_file(
    file: Path,
    source: str | None,
    text_field: str | Sequence[str] | Callable[[dict[str, Any]], str],
    id_field: str | Callable[[dict[str, Any]], str] | None,
) -> Iterable[dict[str, Any]]:
    with file.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                envelope = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{file}:{line_no}: invalid JSON ({exc.msg})"
                ) from exc

            if not isinstance(envelope, dict):
                raise ValueError(
                    f"{file}:{line_no}: expected a JSON object, got "
                    f"{type(envelope).__name__}"
                )

            data = _extract_source_row(envelope)
            yield {
                "id": _extract_id(envelope, data, id_field),
                "text": _extract_text(data, text_field),
                "source": source,
                "metadata": {
                    "_airbyte": {
                        k: envelope[k]
                        for k in _AIRBYTE_ENVELOPE_KEYS
                        if k in envelope and k != "_airbyte_data"
                    },
                    **data,
                },
            }


def _extract_source_row(envelope: dict[str, Any]) -> dict[str, Any]:
    """Pull source columns out of the envelope.

    Prefers ``_airbyte_data``; falls back to flat top-level columns if the
    destination didn't use that wrapper.
    """
    data = envelope.get("_airbyte_data")
    if isinstance(data, dict):
        return data
    return {k: v for k, v in envelope.items() if k not in _AIRBYTE_ENVELOPE_KEYS}


def _extract_id(
    envelope: dict[str, Any],
    data: dict[str, Any],
    spec: str | Callable[[dict[str, Any]], str] | None,
) -> str:
    if callable(spec):
        return str(spec(data))
    if isinstance(spec, str):
        if spec not in data:
            raise KeyError(f"id_field {spec!r} not found in source row")
        return str(data[spec])
    for key in _ID_ENVELOPE_KEYS:
        val = envelope.get(key)
        if val is not None:
            return str(val)
    raise KeyError(
        "no id found — pass id_field='<column>' or a callable, or ensure "
        "the Airbyte envelope carries _airbyte_raw_id or _airbyte_ab_id"
    )


def _extract_text(
    data: dict[str, Any],
    spec: str | Sequence[str] | Callable[[dict[str, Any]], str],
) -> str:
    if callable(spec):
        return str(spec(data))
    if isinstance(spec, str):
        val = data.get(spec)
        return "" if val is None else str(val)
    parts: list[str] = []
    for key in spec:
        val = data.get(key)
        if val is None:
            continue
        stripped = str(val).strip()
        if stripped:
            parts.append(stripped)
    return " ".join(parts)


def _stream_from_filename(stem: str) -> str | None:
    """Best-effort stream name from a filename stem.

    Airbyte Local JSON writes filenames like ``<stream>_<epoch>_<idx>.jsonl``
    or just ``<stream>.jsonl``. Strip trailing purely-numeric segments so
    ``stories_20260420_00001`` collapses to ``stories``.
    """
    parts = stem.split("_")
    while len(parts) > 1 and parts[-1].isdigit():
        parts.pop()
    return "_".join(parts) or None
