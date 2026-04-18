"""Record joiners.

A "join" in prep is the dict-of-records analog of a SQL/dataframe join:
combine fields from a left and right table on a shared key. Useful for
pulling in reference data (cik→ticker, place→country, etc.) before
indexing.

This is intentionally minimal — for big jobs use pandas / pyarrow /
duckdb directly. The verb is here so a project recipe can stay in
pure Python.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable, Literal, Sequence, Union

How = Literal["inner", "left", "right", "outer"]
Record = dict[str, Any]
KeyFn = Callable[[Record], Any]


def _coerce_records(items: Iterable[Any]) -> list[Record]:
    out: list[Record] = []
    for it in items:
        if isinstance(it, dict):
            out.append(dict(it))
        else:
            out.append({"value": it})
    return out


def _make_key_fn(on: Union[str, Sequence[str], KeyFn]) -> KeyFn:
    if callable(on):
        return on
    if isinstance(on, str):
        return lambda r, _k=on: r.get(_k)
    keys = tuple(on)
    return lambda r, _ks=keys: tuple(r.get(k) for k in _ks)


def _merge(left: Record, right: Record, *, suffixes: tuple[str, str]) -> Record:
    merged: Record = dict(left)
    l_suffix, r_suffix = suffixes
    for k, v in right.items():
        if k in merged and merged[k] != v:
            merged[f"{k}{l_suffix}"] = merged.pop(k)
            merged[f"{k}{r_suffix}"] = v
        else:
            merged.setdefault(k, v)
    return merged


def join_records(
    left: Iterable[Any],
    right: Iterable[Any],
    *,
    on: Union[str, Sequence[str], KeyFn],
    right_on: Union[str, Sequence[str], KeyFn, None] = None,
    how: How = "left",
    suffixes: tuple[str, str] = ("_left", "_right"),
) -> list[Record]:
    """Hash-join ``left`` and ``right`` on the given key.

    Args:
        left, right: Iterables of dict records.
        on: Column name, tuple of column names, or callable producing the key.
        right_on: If the right side keys differ, supply them here. Defaults to ``on``.
        how: ``"inner"``, ``"left"``, ``"right"``, or ``"outer"``.
        suffixes: Used to disambiguate columns present in both sides
            with different values.
    """
    left_recs = _coerce_records(left)
    right_recs = _coerce_records(right)
    left_key_fn = _make_key_fn(on)
    right_key_fn = _make_key_fn(right_on if right_on is not None else on)

    right_index: dict[Any, list[Record]] = defaultdict(list)
    for r in right_recs:
        right_index[right_key_fn(r)].append(r)

    matched_right_keys: set[Any] = set()
    out: list[Record] = []

    for l in left_recs:
        k = left_key_fn(l)
        matches = right_index.get(k, [])
        if matches:
            matched_right_keys.add(k)
            for r in matches:
                out.append(_merge(l, r, suffixes=suffixes))
        elif how in {"left", "outer"}:
            out.append(dict(l))

    if how in {"right", "outer"}:
        for r in right_recs:
            k = right_key_fn(r)
            if k not in matched_right_keys:
                out.append(dict(r))

    return out


__all__ = ["join_records", "How"]
