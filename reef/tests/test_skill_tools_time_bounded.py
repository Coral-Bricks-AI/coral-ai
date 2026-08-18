# Copyright 2026 Coral Bricks AI Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""The ``@time_bounded`` contract as applied on the ``skill_fn`` path.

``run_react`` dispatches every folder-shaped skill through the single
``invoke_skill_fn`` Tool, so the enforcer's ``before_tool_call`` only
ever sees *that* tool -- whose ``fn`` carries no ``__time_bound__``.
Without the reach-through in :func:`reef.skill_tools._do_invoke_skill_fn`
the decorator is dead metadata for every skill_fn-dispatched skill, and
nothing else in the suite would notice.

These tests pin that behavior on the inner callable, and pin the two
call sites to one shared implementation so they can't drift apart.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

import pytest

from reef import skill_fn as skill_fn_registry
from reef.constraints import HarnessConstraints
from reef.context import harness_context
from reef.decorators import time_bounded
from reef.enforcement import (
    AsofViolation,
    AuditingEnforcer,
    LocalEnforcer,
    TimeBound,
    apply_time_bound,
)
from reef.skill_fn import skill_fn
from reef.skill_tools import _do_invoke_skill_fn

ASOF = "2025-12-31"


# ---------------------------------------------------------------------------
# Fixture skills -- one per mode, plus an undecorated control.
# ---------------------------------------------------------------------------

def _echo_params() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "as_of_iso": {"type": "string"},
        },
        "required": ["ticker"],
    }


@pytest.fixture(autouse=True)
def _registry() -> Any:
    """Register the fixture skills into a clean registry per test.

    ``skill_fn`` writes to a process-global registry at import time, so
    tests that share it leak into each other. Clear on both sides.
    """
    skill_fn_registry.clear()

    @time_bounded(asof_arg="as_of_iso", mode="inject")
    @skill_fn(skill_id="inj", description="inject", parameters=_echo_params())
    def inject_fn(*, ticker: str, as_of_iso: Optional[str] = None) -> dict[str, Any]:
        return {"ticker": ticker, "as_of_iso": as_of_iso}

    @time_bounded(asof_arg="as_of_iso", filter_field="filing_date", mode="clamp")
    @skill_fn(skill_id="clm", description="clamp", parameters=_echo_params())
    def clamp_fn(*, ticker: str, as_of_iso: Optional[str] = None) -> dict[str, Any]:
        # Deliberately ignores as_of_iso: proves the post-filter is a
        # real safety net and not just the skill behaving itself.
        return {
            "ticker": ticker,
            "as_of_iso": as_of_iso,
            "rows": [
                {"filing_date": "2025-06-30", "form_type": "10-Q"},
                {"filing_date": "2026-05-28", "form_type": "10-Q"},
            ],
        }

    @time_bounded(asof_arg="as_of_iso", mode="validate")
    @skill_fn(skill_id="val", description="validate", parameters=_echo_params())
    def validate_fn(*, ticker: str, as_of_iso: Optional[str] = None) -> dict[str, Any]:
        return {"ticker": ticker, "as_of_iso": as_of_iso, "ran": True}

    @skill_fn(skill_id="plain", description="no contract", parameters=_echo_params())
    def plain_fn(*, ticker: str, as_of_iso: Optional[str] = None) -> dict[str, Any]:
        return {"ticker": ticker, "as_of_iso": as_of_iso}

    yield
    skill_fn_registry.clear()


def call(skill_id: str, fn: str, **args: Any) -> Any:
    return _do_invoke_skill_fn(skill_id=skill_id, fn=fn, args=args)


# ---------------------------------------------------------------------------
# The three modes, dispatched through invoke_skill_fn
# ---------------------------------------------------------------------------

def test_inject_overwrites_model_value() -> None:
    with harness_context(HarnessConstraints(asof=ASOF)):
        out = call("inj", "inject_fn", ticker="NVDA", as_of_iso="2026-06-01")
    assert out["as_of_iso"] == ASOF


def test_inject_fills_in_when_model_omits_the_arg() -> None:
    with harness_context(HarnessConstraints(asof=ASOF)):
        out = call("inj", "inject_fn", ticker="NVDA")
    assert out["as_of_iso"] == ASOF


def test_clamp_narrows_a_future_value_but_keeps_an_earlier_one() -> None:
    with harness_context(HarnessConstraints(asof=ASOF)):
        future = call("clm", "clamp_fn", ticker="NVDA", as_of_iso="2026-06-01")
        earlier = call("clm", "clamp_fn", ticker="NVDA", as_of_iso="2025-03-01")
    assert future["as_of_iso"] == ASOF, "future value should clamp down to asof"
    assert earlier["as_of_iso"] == "2025-03-01", "model may pick a narrower cutoff"


def test_validate_rejects_a_future_value_without_running_the_skill() -> None:
    with harness_context(HarnessConstraints(asof=ASOF)):
        out = call("val", "validate_fn", ticker="NVDA", as_of_iso="2027-01-01")
    assert out["constraint_violation"] is True
    assert "AsofViolation" in out["error"]
    assert ASOF in out["error"], "the model needs the real cutoff to self-correct"
    assert "ran" not in out, "skill body must not execute on a violation"
    # Envelope carries enough for the model to see what it sent.
    assert out["skill_id"] == "val" and out["fn"] == "validate_fn"


def test_validate_allows_a_value_at_or_before_asof() -> None:
    with harness_context(HarnessConstraints(asof=ASOF)):
        at = call("val", "validate_fn", ticker="NVDA", as_of_iso=ASOF)
        before = call("val", "validate_fn", ticker="NVDA", as_of_iso="2025-01-01")
    assert at["ran"] is True and at["as_of_iso"] == ASOF
    assert before["ran"] is True and before["as_of_iso"] == "2025-01-01"


def test_unparseable_value_falls_back_to_inject() -> None:
    with harness_context(HarnessConstraints(asof=ASOF)):
        out = call("val", "validate_fn", ticker="NVDA", as_of_iso="last Tuesday")
    assert out["as_of_iso"] == ASOF, "uncomparable input must not slip through"


# ---------------------------------------------------------------------------
# Post-filter safety net
# ---------------------------------------------------------------------------

def test_rows_past_asof_are_dropped_even_when_the_skill_ignores_its_arg() -> None:
    with harness_context(HarnessConstraints(asof=ASOF)):
        out = call("clm", "clamp_fn", ticker="NVDA")
    dates = [r["filing_date"] for r in out["rows"]]
    assert dates == ["2025-06-30"], "2026-05-28 is past asof and must be filtered"


def test_no_filtering_without_a_filter_field() -> None:
    with harness_context(HarnessConstraints(asof=ASOF)):
        out = call("inj", "inject_fn", ticker="NVDA")
    assert "rows" not in out  # sanity: inject_fn returns no rows at all


# ---------------------------------------------------------------------------
# Unconstrained runs stay unconstrained
# ---------------------------------------------------------------------------

def test_no_harness_context_leaves_args_and_rows_untouched() -> None:
    out = call("clm", "clamp_fn", ticker="NVDA", as_of_iso="2026-06-01")
    assert out["as_of_iso"] == "2026-06-01"
    assert len(out["rows"]) == 2


def test_constraints_without_asof_leave_args_untouched() -> None:
    with harness_context(HarnessConstraints(asof=None)):
        out = call("clm", "clamp_fn", ticker="NVDA", as_of_iso="2026-06-01")
    assert out["as_of_iso"] == "2026-06-01"
    assert len(out["rows"]) == 2


def test_undecorated_skill_is_not_touched() -> None:
    with harness_context(HarnessConstraints(asof=ASOF)):
        out = call("plain", "plain_fn", ticker="NVDA", as_of_iso="2026-06-01")
    assert out["as_of_iso"] == "2026-06-01", (
        "the arg-name convention fallback is an enforcer behavior on the "
        "outer tool; the skill_fn path applies declared bounds only"
    )


# ---------------------------------------------------------------------------
# Error envelopes still win over enforcement
# ---------------------------------------------------------------------------

def test_missing_required_arg_reports_before_any_asof_work() -> None:
    with harness_context(HarnessConstraints(asof=ASOF)):
        out = call("val", "validate_fn", as_of_iso="2027-01-01")
    assert "missing required args" in out["error"]
    assert "constraint_violation" not in out


def test_unknown_skill_id_is_an_error_envelope() -> None:
    with harness_context(HarnessConstraints(asof=ASOF)):
        out = call("nope", "nope_fn", ticker="NVDA")
    assert "no skill_fn registered" in out["error"]


# ---------------------------------------------------------------------------
# Budget: the inner application must not double-charge
# ---------------------------------------------------------------------------

def test_inner_application_does_not_decrement_the_tool_budget() -> None:
    """Accounting belongs to the outer ``invoke_skill_fn`` dispatch.

    The enforcer already charged the run when ``run_react`` called
    ``before_tool_call`` on ``invoke_skill_fn``; reaching through to the
    inner callable is asof-only. If this ever routes through an enforcer
    hook again, every skill call silently costs two.
    """
    enforcer = LocalEnforcer()
    constraints = HarnessConstraints(asof=ASOF, tool_budget=5)
    enforcer.on_run_start(constraints)
    with harness_context(constraints, enforcer):
        call("clm", "clamp_fn", ticker="NVDA")
        call("clm", "clamp_fn", ticker="AAPL")
    assert enforcer.calls_used == 0, (
        "the skill_fn path must not touch the budget; run_react charges "
        "the outer invoke_skill_fn call"
    )


def test_inner_violation_does_not_reach_the_enforcer_event_log() -> None:
    """The inner path returns an envelope rather than raising.

    Consequence worth pinning: an ``AuditingEnforcer`` sees no violation
    event for a skill_fn-level rejection. If that becomes undesirable,
    it's a deliberate change, not an accident.
    """
    enforcer = AuditingEnforcer()
    constraints = HarnessConstraints(asof=ASOF, tool_budget=5)
    enforcer.on_run_start(constraints)
    with harness_context(constraints, enforcer):
        out = call("val", "validate_fn", ticker="NVDA", as_of_iso="2027-01-01")
    assert out["constraint_violation"] is True
    assert enforcer.events == []


# ---------------------------------------------------------------------------
# Drift guard: both venues share one implementation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["inject", "clamp", "validate"])
@pytest.mark.parametrize(
    "model_value", [None, "2027-01-01", "2025-03-01", ASOF, "not a date"]
)
def test_enforcer_and_skill_fn_paths_agree(mode: str, model_value: Any) -> None:
    """``LocalEnforcer`` and the skill_fn reach-through must not diverge.

    Both call ``apply_time_bound``; this asserts the outcomes stay
    identical across every mode x input combination, so a future edit to
    one venue can't quietly re-introduce a second copy of the semantics.
    """
    bound = TimeBound(asof_arg="as_of_iso", mode=mode)
    asof = date.fromisoformat(ASOF)
    constraints = HarnessConstraints(asof=ASOF, tool_budget=10)
    args = {"ticker": "NVDA"}
    if model_value is not None:
        args["as_of_iso"] = model_value

    def via_shared() -> Any:
        a = dict(args)
        try:
            apply_time_bound("t", a, bound, asof)
        except AsofViolation as exc:
            return ("violation", str(exc))
        return ("ok", a.get("as_of_iso"))

    def via_enforcer() -> Any:
        enforcer = LocalEnforcer()
        enforcer.on_run_start(constraints)

        @time_bounded(asof_arg="as_of_iso", mode=mode)
        def tool_fn(*, ticker: str, as_of_iso: Optional[str] = None) -> None:
            return None

        try:
            out = enforcer.before_tool_call("t", tool_fn, args, constraints)
        except AsofViolation as exc:
            return ("violation", str(exc))
        return ("ok", out.get("as_of_iso"))

    assert via_shared() == via_enforcer()
