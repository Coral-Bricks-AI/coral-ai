# Copyright 2026 Coral Bricks AI Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Trailing 1-year price return computation over the in-repo companies corpus.

Composes :func:`~reef.decorators.time_bounded` with
:func:`~reef.skill_fn.skill_fn`. The ``validate`` mode means: if the
model tries to compute a return "as of" a date past the run's asof,
the runtime rejects the call outright (as opposed to ``clamp`` which
would silently rewind, or ``inject`` which would overwrite). That's the
right posture here because a return "as of tomorrow" isn't a partial
answer — it's the wrong question, and quietly answering it as of asof
would hide the misuse.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
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


def _price_at_or_before(company: dict[str, Any], target: date) -> Optional[tuple[date, float]]:
    """Latest (date, price) snapshot ≤ ``target``, or ``None`` if the
    corpus doesn't reach that far back.

    The corpus carries a monthly price grid per ticker (see
    ``data/companies.json``), so this nearest-neighbor lookup lands
    within a month of any target inside the horizon -- close enough
    that a trailing-1y window is a real 12 months rather than a snap
    back to a distant anchor. Still a lookup, not interpolation; a real
    implementation would hit a price service.
    """
    history = company.get("price_history") or []
    parsed = [(_parse_iso(row.get("date")), float(row["price"])) for row in history]
    candidates = [(d, p) for (d, p) in parsed if d is not None and d <= target]
    if not candidates:
        return None
    return max(candidates, key=lambda dp: dp[0])


@time_bounded(asof_arg="as_of_iso", mode="validate")
@skill_fn(
    skill_id="compute_total_return",
    description=(
        "Compute trailing 1-year price return for a ticker as of an optional "
        "as_of_iso. Returns pct_return_1y + a grader-ready answer_summary_block."
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
                "description": (
                    "Optional ISO date (YYYY-MM-DD) at which to anchor the "
                    "'now' price. Defaults to the corpus's latest snapshot."
                ),
            },
        },
        "required": ["ticker"],
    },
)
def compute_total_return(
    *,
    ticker: str,
    as_of_iso: Optional[str] = None,
) -> dict[str, Any]:
    company = _BY_TICKER.get(ticker.upper())
    if company is None:
        return {
            "error": (
                f"unknown ticker={ticker!r}; "
                f"call search_companies first to get a valid ticker."
            )
        }

    # Resolve "now": either the caller's as_of_iso (already validated
    # by the enforcer to be <= run asof) or the corpus's latest anchor.
    corpus_latest = _parse_iso(company.get("price_asof")) or date(2026, 7, 15)
    now_target = _parse_iso(as_of_iso) or corpus_latest
    now = _price_at_or_before(company, now_target)
    if now is None:
        return {
            "error": (
                f"no price snapshot on or before {now_target.isoformat()} "
                f"for {ticker!r}; corpus horizon is limited."
            )
        }
    now_date, price_now = now

    # 1 year prior anchor.
    prior_target = now_date - timedelta(days=365)
    prior = _price_at_or_before(company, prior_target)
    if prior is None:
        return {
            "error": (
                f"no price snapshot on or before {prior_target.isoformat()} "
                f"(1y before {now_date.isoformat()}) for {ticker!r}; "
                f"corpus horizon is limited."
            )
        }
    prior_date, price_1y_ago = prior

    pct = round((price_now - price_1y_ago) / price_1y_ago * 100, 1)
    direction = "returned" if pct >= 0 else "lost"
    summary_block = (
        f"**{company['name']} ({company['ticker']})** {direction} "
        f"**{pct:+.1f}%** over the trailing 12 months "
        f"(${price_1y_ago:.2f} on {prior_date.isoformat()} → "
        f"${price_now:.2f} on {now_date.isoformat()}). "
        f"Price return only; does not include dividends."
    )
    return {
        "ticker": company["ticker"],
        "as_of_iso": now_date.isoformat(),
        "pct_return_1y": pct,
        "price_now": price_now,
        "price_now_date": now_date.isoformat(),
        "price_1y_ago": price_1y_ago,
        "price_1y_ago_date": prior_date.isoformat(),
        "answer_summary_block": summary_block,
    }
