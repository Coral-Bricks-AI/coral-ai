#!/usr/bin/env python3
# Copyright 2026 Coral Bricks AI Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Skinny end-to-end example: one specialist, three skills, BM25 + 1y return math + recent filings.

Usage::

    export LLM_API_KEY=sk-...
    python reef/examples/equities/ask.py "How has NVDA performed over the last year?"
    python reef/examples/equities/ask.py --model anthropic/claude-sonnet-4-6 "..."

    # Constrain the run to an as-of date. The @time_bounded skills honor it:
    #   list_recent_filings clamps its as_of_iso to this date
    #   compute_total_return validates against this date and rejects future values
    python reef/examples/equities/ask.py --asof 2025-12-31 \\
        "What did NVDA report last, and how had the stock done at that point?"

The framework hello-world. No planner, no synthesizer, no
SpecialistConfig -- just :func:`reef.react.run_react` wired to a
persona prompt and two skill-dispatch tools, optionally scoped by
:class:`~reef.constraints.HarnessConstraints`.

``make_load_skill_tool`` is the factory the framework ships for the
``load_skill`` Tool; we close it over this example's ``SKILLS`` dict.
``INVOKE_SKILL_FN`` is reused straight from the framework (the
``@skill_fn`` decorator registers into a process-global registry that
this example's ``impl.py`` modules populate when ``load_skills(...)``
imports them).

The data in ``data/companies.json`` is **mock and illustrative** — ~20
well-known tickers with fabricated point-in-time prices, monthly
price histories, and filings. Do not mistake this for live market data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from reef.constraints import HarnessConstraints
from reef.context import harness_context
from reef.react import run_react
from reef.skills_loader import load_skills, render_index, render_loaded
from reef.skill_tools import INVOKE_SKILL_FN, make_load_skill_tool

HERE = Path(__file__).resolve().parent

# Load this example's skills and import their impl.py modules. The import
# pass registers the @skill_fn-decorated callables with the global registry
# that reef.skill_tools.INVOKE_SKILL_FN dispatches against.
SKILLS = load_skills(
    HERE / "skills",
    module_prefix="reef.examples.equities._skills",
)


LOAD_SKILL = make_load_skill_tool(
    lambda ids: render_loaded(list(ids), skills=SKILLS),
)


# Render the skill index once at module load and stitch it into the persona.
_INDEX = render_index(SKILLS)
_PROMPT = (HERE / "analyst.md").read_text(encoding="utf-8").replace(
    "{skill_index}", _INDEX
)


def ask(
    question: str,
    model: str = "openai/gpt-5-mini",
    *,
    asof: Optional[str] = None,
    tool_budget: int = 20,
) -> str | None:
    """Run the equity analyst on one question; return its final natural-language answer.

    When ``asof`` is set, wraps ``run_react`` in a
    :func:`~reef.context.harness_context` so the default
    :class:`~reef.enforcement.LocalEnforcer` applies the ``@time_bounded``
    contract on every skill dispatch.
    """

    def _run() -> str | None:
        traj = run_react(
            model=model,
            system_prompt=_PROMPT,
            user_message=question,
            tools=[LOAD_SKILL, INVOKE_SKILL_FN],
            max_steps=6,
            log_label="equities.analyst",
        )
        if traj.final_message is None:
            return None
        return traj.final_message.get("content") or ""

    if asof is None:
        return _run()

    constraints = HarnessConstraints(asof=asof, tool_budget=tool_budget)
    with harness_context(constraints):
        return _run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ask the equity analyst a question.",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-5-mini",
        help=(
            "<provider>/<model> string passed to reef.llm.chat "
            "(default: openai/gpt-5-mini). The matching <PROVIDER>_API_KEY "
            "env var must be set; reef.llm raises a clear error if not. "
            "Examples: anthropic/claude-sonnet-4-6, together/kimi-k2.6, "
            "lilac/moonshotai/kimi-k2.6."
        ),
    )
    parser.add_argument(
        "--asof",
        default=None,
        help=(
            "Optional ISO date (YYYY-MM-DD) to scope the run. When set, "
            "@time_bounded skills honor it via LocalEnforcer: "
            "list_recent_filings clamps, compute_total_return validates."
        ),
    )
    parser.add_argument(
        "--tool-budget",
        type=int,
        default=20,
        help=(
            "Max tool calls per run when --asof is set (ignored otherwise). "
            "Default 20 is generous for this 3-skill demo."
        ),
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="The question to ask. Defaults to a sample NVDA question if omitted.",
    )
    args = parser.parse_args()
    q = " ".join(args.question) or "How has NVDA performed over the last year?"
    print(f"Q: {q}")
    if args.asof:
        print(f"   (asof={args.asof}, tool_budget={args.tool_budget})")
    print()
    answer = ask(q, model=args.model, asof=args.asof, tool_budget=args.tool_budget)
    print(f"A: {answer}")
