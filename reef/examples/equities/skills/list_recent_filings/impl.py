# Copyright 2026 Coral Bricks AI Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Recent-filings lookup over the in-repo companies corpus.

Demonstrates stacking :func:`~reef.decorators.time_bounded` on top of
:func:`~reef.skill_fn.skill_fn`. The two decorators do independent jobs
(one registers the callable for dispatch, one stamps the temporal
contract) and compose in either order; we put ``@time_bounded`` on the
outside so a reader eyeballing the file sees the constraint first.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Optional

from reef.decorators import time_bounded
from reef.skill_fn import skill_fn

_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "companies.json"


with _DATA_PATH.open(encoding="utf-8") as _f:
    _COMPANIES: list[dict[str, Any]] = json.load(_f)
_BY_TICKER: dict[str, dict[str, Any]] = {c["ticker"]: c for c in _COMPANIES}


def _parse_iso(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


@time_bounded(asof_arg="as_of_iso", filter_field="filing_date", mode="clamp")
@skill_fn(
    skill_id="list_recent_filings",
    description=(
        "Return the k most recent SEC filings for a ticker, on or before "
        "as_of_iso. Rows carry filing_date + form_type + headline."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Ticker symbol from a search_companies result (e.g. 'NVDA').",
            },
            "as_of_iso": {
                "type": "string",
                "description": "Optional ISO date (YYYY-MM-DD) upper bound on filing_date.",
            },
            "k": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 10,
                "description": "Max filings to return, most recent first.",
            },
        },
        "required": ["ticker"],
    },
)
def list_recent_filings(
    *,
    ticker: str,
    as_of_iso: Optional[str] = None,
    k: int = 5,
) -> dict[str, Any]:
    company = _BY_TICKER.get(ticker.upper())
    if company is None:
        return {
            "error": (
                f"unknown ticker={ticker!r}; call search_companies first "
                f"to get a valid ticker."
            )
        }

    cutoff = _parse_iso(as_of_iso)
    filings: list[dict[str, Any]] = list(company.get("filings") or [])
    if cutoff is not None:
        filings = [
            f for f in filings
            if (_parse_iso(f.get("filing_date")) or date.min) <= cutoff
        ]
    filings.sort(key=lambda f: f.get("filing_date") or "", reverse=True)
    filings = filings[: max(1, min(int(k), 10))]

    if not filings:
        return {
            "ticker": company["ticker"],
            "as_of_iso": as_of_iso,
            "filings": [],
            "answer_summary_block": (
                f"No filings for **{company['name']} ({company['ticker']})** "
                f"on or before {as_of_iso or 'the corpus horizon'}."
            ),
        }

    most_recent = filings[0]
    summary_block = (
        f"**{company['name']} ({company['ticker']})** — most recent filing on "
        f"or before {as_of_iso or 'the corpus horizon'} is a "
        f"**{most_recent['form_type']}** dated **{most_recent['filing_date']}**: "
        f"\"{most_recent['headline']}\" ({len(filings)} filing(s) returned)."
    )
    return {
        "ticker": company["ticker"],
        "as_of_iso": as_of_iso,
        "filings": filings,
        "answer_summary_block": summary_block,
    }
