"""Document cleaning via trafilatura.

Coral Bricks deliberately ships **no custom text-cleaning rules**. The
only supported cleaner is `trafilatura`'s main-content extraction so
behaviour stays consistent across reference indices.

Inputs may be HTML strings or dict records (with ``"text"`` holding
the raw HTML). Outputs are dict records whose ``text`` is the
trafilatura-extracted main content. Any extracted title / publication
date is attached to ``metadata``.
"""

from __future__ import annotations

import threading
from typing import Any, Iterable, Union

from ._records import normalize_records

# trafilatura is built on lxml/libxml2 which is not safe under concurrent
# parsing from multiple threads. Hold a process-wide lock around
# trafilatura calls.
_PARSE_LOCK = threading.Lock()


class TrafilaturaUnavailable(RuntimeError):
    """Raised when trafilatura is not installed."""


def _require_trafilatura():
    try:
        import trafilatura  # type: ignore
    except ImportError as exc:
        raise TrafilaturaUnavailable(
            "clean() requires trafilatura. Install with `pip install trafilatura` "
            "(or `pip install 'coralbricks-context-prep[cleaners]'`)."
        ) from exc
    return trafilatura


def clean_html(html: str) -> dict[str, Any]:
    """Extract main-content text + metadata from a single HTML string.

    Returns a dict with keys ``text`` (str), ``title`` (str | None),
    ``published_date`` (str | None). ``text`` is empty if trafilatura
    finds no main content.
    """
    if not html or not html.strip():
        return {"text": "", "title": None, "published_date": None}

    trafilatura = _require_trafilatura()
    with _PARSE_LOCK:
        body = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_recall=True,
        )
        meta = trafilatura.metadata.extract_metadata(html)

    title: str | None = None
    pub: str | None = None
    if meta:
        if meta.title:
            title = meta.title.strip() or None
        raw_date = getattr(meta, "date", None)
        if raw_date:
            pub = str(raw_date)

    return {
        "text": (body or "").strip(),
        "title": title,
        "published_date": pub,
    }


def clean_documents(
    docs: Union[str, dict, Iterable[Any]],
    *,
    drop_empty: bool = True,
) -> list[dict[str, Any]]:
    """Run trafilatura over each input and return cleaned records.

    Args:
        docs: A single string/dict or an iterable. Strings are treated
            as raw HTML. Dicts use the ``text`` field as the HTML source.
        drop_empty: When True (default), drop records where trafilatura
            extracted no main content.
    """
    if isinstance(docs, (str, dict)):
        items: list[Any] = [docs]
    else:
        items = list(docs)
    records = normalize_records(items)

    cleaned: list[dict[str, Any]] = []
    for rec in records:
        result = clean_html(rec["text"])
        if drop_empty and not result["text"]:
            continue
        new_meta = dict(rec.get("metadata") or {})
        new_meta["cleaned_by"] = "trafilatura"
        if result["title"]:
            new_meta.setdefault("title", result["title"])
        if result["published_date"]:
            new_meta.setdefault("published_date", result["published_date"])
        cleaned.append(
            {
                "id": rec["id"],
                "text": result["text"],
                "source": rec.get("source"),
                "metadata": new_meta,
            }
        )
    return cleaned


__all__ = [
    "TrafilaturaUnavailable",
    "clean_html",
    "clean_documents",
]
