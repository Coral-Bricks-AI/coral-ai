"""Recipe = an ordered chain of verb calls + a tiny in-memory runner.

A ``Recipe`` captures a verb chain (``recipe.plan()`` prints the DAG)
and runs the verbs in declared order, in-process, in this Python
interpreter.

**Scope:** Recipe is a prototyping convenience. It is single-process,
in-memory, no checkpointing, no retries, no parallelism. For
million-row jobs, drop Recipe entirely and call the prep primitives
(``context_prep.chunkers``, ``.embedders``, ``.cleaners``,
``.enrichers``, ``.graph``) directly inside your orchestrator's tasks
(Airflow, Prefect, Dagster, Ray, Spark, etc.). For partial-graph
hydration in distributed jobs, use
:func:`context_prep.graph.merge_graphs` as the reduce step.

Usage::

    recipe = Recipe("sec_filings_2026q1")
    recipe.add(clean, html_records)
    recipe.add(chunk, "<upstream>", strategy="sliding_token", target_tokens=512)
    recipe.run()                    # executes top to bottom
    recipe.plan()                   # returns a list of (verb, args) tuples
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VerbCall:
    """One pending verb invocation inside a Recipe."""

    name: str  # 'clean' / 'chunk' / 'embed' / etc.
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    label: str | None = None  # human-readable id for plan output

    def describe(self) -> str:
        # Shorten artifact reprs in the plan output.
        def _short(v: Any) -> str:
            from .artifact import Artifact

            if isinstance(v, Artifact):
                return f"<Artifact {v.kind.value}:{v.artifact_id}>"
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], Artifact):
                return f"<{len(v)} artifacts>"
            return repr(v)

        a = ", ".join(_short(v) for v in self.args)
        kw = ", ".join(f"{k}={_short(v)}" for k, v in self.kwargs.items())
        sig = ", ".join(p for p in (a, kw) if p)
        return f"{self.label or self.name}({sig})"


@dataclass
class Recipe:
    name: str
    project_id: int | None = None
    calls: list[VerbCall] = field(default_factory=list)
    _results: list[Any] = field(default_factory=list, init=False, repr=False)

    def add(
        self,
        verb: Callable[..., Any],
        *args: Any,
        label: str | None = None,
        **kwargs: Any,
    ) -> VerbCall:
        """Schedule a verb call. Returns the :class:`VerbCall` so callers
        can reference it before :meth:`run` is invoked.
        """
        call = VerbCall(
            name=getattr(verb, "__name__", "verb"),
            fn=verb,
            args=args,
            kwargs=kwargs,
            label=label,
        )
        self.calls.append(call)
        return call

    def plan(self) -> list[str]:
        """Return a human-readable plan of the recipe."""
        return [f"  {i + 1:>2}. {c.describe()}" for i, c in enumerate(self.calls)]

    def print_plan(self) -> None:
        print(f"Recipe: {self.name}")
        for line in self.plan():
            print(line)

    def run(self) -> list[Any]:
        """Execute the verbs top-to-bottom and return the collected results."""
        self._results = []
        for call in self.calls:
            logger.info("recipe %s: running %s", self.name, call.name)
            result = call.fn(*call.args, **call.kwargs)
            self._results.append(result)
        return list(self._results)
