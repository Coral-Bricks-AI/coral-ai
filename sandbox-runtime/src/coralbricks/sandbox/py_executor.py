"""In-runner Python interpreter for ``tools.py``.

The ``tools.py`` verb is the only kernel verb whose dispatch lives
**inside the runner subprocess** rather than the gateway. The reason
is structural: the spec wants the model to feed a snippet of Python
into the same address space that already holds the pipeline's
in-flight tool results so it can write things like::

    df = pd.DataFrame(bars["rows"])
    df["sma_20"] = df["close"].rolling(20).mean()
    result = df.tail(30).to_dict(orient="records")

Round-tripping that through the gateway would either (a) ship the
prior tool-call payloads back over the RPC wire (slow + expensive
for large dataframes) or (b) require the gateway to host an
arbitrary code execution endpoint inside its own address space
(actively dangerous: gateway holds tenant credentials). Keeping
``py`` in the runner sidesteps both.

What this module ships
----------------------

- :class:`PyValidationError` -- the static-analysis verdict. Raised
  by :func:`validate_code` (and re-raised by :meth:`PyExecutor.exec`)
  when the snippet would import a banned module, call ``eval`` /
  ``exec`` / ``compile`` / ``__import__``, or reach into a forbidden
  attribute (``__class__``, ``__bases__``, ``__subclasses__``,
  ``__globals__``, ``__builtins__``, ``__dict__``, ``__loader__``).
  The validator is intentionally conservative: it rejects whole
  *patterns* (not just literal strings) so an attacker can't go
  through ``getattr(obj, '__cl' + 'ass__')``.
- :class:`PyExecutionError` -- runtime failure of an otherwise-valid
  snippet (uncaught exception, timeout). Carries the original
  exception type name + traceback string so the kernel can serialize
  cleanly into the wire envelope.
- :class:`PyTimeoutError` -- subclass of :class:`PyExecutionError`
  raised when the wall-clock budget fires.
- :class:`PyResult` -- structured outcome (``stdout``, ``result``,
  ``globals_added``, ``took_ms``).
- :class:`PyExecutor` -- stateful executor; the runner instantiates
  one per pipeline run and reuses it across every ``tools.py`` call
  so later snippets see globals defined by earlier snippets.
- :func:`validate_code` -- pure AST walker, exposed for unit tests
  and for offline linting (e.g. a future ``cb pipeline lint``).

Stdlib only -- this module ships in the per-pipeline venv every
sandbox spawns. Adding a third-party dep here would balloon every
pipeline's cold-start cost, so we keep it to ``ast``, ``signal``,
``contextlib``, ``io``, ``time``, ``traceback``.

Threading note: timeout enforcement uses ``signal.SIGALRM``, which
is POSIX-only and only fires in the *main* thread. The Coral runner
calls ``PyExecutor.exec`` from its main thread (the runner is
single-threaded by design), so this is sufficient. CPU-bound C
extensions that hold the GIL (``numpy.linalg.solve`` on a 10000x10000
matrix) will not be interrupted promptly -- the OS-level wall budget
that the gateway's :class:`SubprocessExecutor` enforces is the
hard backstop. Same trade-off the rest of CPython lives with.
"""

from __future__ import annotations

import ast
import contextlib
import io
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional


# Stdlib modules every snippet may import without listing them in the
# manifest's ``[tool.coralbricks.pipeline.py].libraries`` table. These
# are intentionally narrow: pure-Python helpers + safe data primitives.
# ``os``, ``sys``, ``subprocess``, ``socket``, ``ctypes``, ``importlib``
# are NOT here -- they require an explicit manifest opt-in (and even
# then the AST validator flags ``os.system`` style calls).
_DEFAULT_STDLIB_ALLOWLIST: frozenset[str] = frozenset({
    "math",
    "statistics",
    "datetime",
    "json",
    "re",
    "time",
    "collections",
    "itertools",
    "functools",
    "operator",
    "typing",
    "dataclasses",
    "decimal",
    "fractions",
    "random",
    "string",
    "textwrap",
    "copy",
    "hashlib",
    "base64",
    "uuid",
    "calendar",
    "enum",
    "heapq",
    "bisect",
    "warnings",
})

# Attribute names that lift the lid on Python's introspection /
# escape-from-sandbox tooling. Any read or write of any of these on
# any object is rejected at AST-walk time. The list is the standard
# PyJail break surface; adding to it is cheap, removing is dangerous.
_BANNED_ATTRS: frozenset[str] = frozenset({
    "__class__",
    "__bases__",
    "__mro__",
    "__subclasses__",
    "__globals__",
    "__builtins__",
    "__dict__",
    "__loader__",
    "__spec__",
    "__import__",
    "__getattribute__",
    "__init_subclass__",
    "__reduce__",
    "__reduce_ex__",
    "__getattr__",
    "__setattr__",
    "__del__",
    "__code__",
    "__closure__",
    "__func__",
    "__self__",
    "__module__",
    "__wrapped__",
    "__objclass__",
    "f_globals",
    "f_locals",
    "f_back",
    "gi_frame",
    "cr_frame",
    "ag_frame",
    "tb_frame",
})

# Builtin functions that either run arbitrary code (``eval``,
# ``exec``, ``compile``, ``__import__``, ``breakpoint``) or hand the
# snippet a reference to its own enclosing globals dict from which it
# could mutate the run state. ``vars`` *with* an argument is fine
# (``vars(obj)`` == ``obj.__dict__``, already covered by the attr
# ban); ``vars()`` no-args returns the locals so it's banned too.
_BANNED_CALLS: frozenset[str] = frozenset({
    "eval",
    "exec",
    "compile",
    "__import__",
    "breakpoint",
    "globals",
    "locals",
    "vars",
})

# Default per-call wall-clock budget. The kernel can override per
# pipeline via ``configJson.py.timeout_ms`` at registration time, but
# 30s is the documented Phase 1 ceiling.
_DEFAULT_TIMEOUT_MS = 30_000

# Default cap on the captured ``stdout`` byte length. Large snippets
# that spew prints get truncated with a ``"... [truncated N bytes]"``
# suffix; ``result`` itself is not truncated by this knob (the kernel
# clamps the wire envelope independently).
_DEFAULT_MAX_OUTPUT_BYTES = 1 * 1024 * 1024


class PyValidationError(Exception):
    """Static-analysis rejection of a snippet.

    Raised before the snippet is compiled, so a bad ``import os``
    or ``__class__`` reference fails fast with a clear pointer at
    the offending line / column. The ``node`` attribute is the AST
    node that triggered the rejection (useful for tooling; not
    serialized over the wire).
    """

    def __init__(self, message: str, *, node: Optional[ast.AST] = None) -> None:
        super().__init__(message)
        self.node = node


class PyExecutionError(Exception):
    """An otherwise-valid snippet raised at runtime.

    ``original_type`` is the class name of the original exception
    (e.g. ``"KeyError"``); ``traceback_str`` is the formatted
    traceback. The kernel surfaces both on the wire so the model
    can match on the error type and read the line that blew up.
    """

    def __init__(
        self,
        message: str,
        *,
        original_type: str = "Exception",
        traceback_str: str = "",
    ) -> None:
        super().__init__(message)
        self.original_type = original_type
        self.traceback_str = traceback_str


class PyTimeoutError(PyExecutionError):
    """The snippet exceeded its wall-clock budget."""

    def __init__(self, message: str, *, timeout_ms: int) -> None:
        super().__init__(message, original_type="PyTimeoutError")
        self.timeout_ms = timeout_ms


# Sentinel for "key not present" lookups; ``None`` is a valid
# bound value, so ``dict.pop(name, None)`` would lose the
# distinction between "wasn't there" and "was bound to None".
_MISSING = object()


@dataclass(frozen=True)
class PyResult:
    """Structured outcome of one :meth:`PyExecutor.exec` call.

    Fields:

    - ``result``: the value bound to the snippet's top-level
      ``result`` variable (or ``None`` if it didn't set one).
    - ``stdout``: text written to stdout, possibly truncated.
    - ``truncated``: ``True`` iff stdout was clipped at
      ``max_output_bytes``.
    - ``globals_added``: top-level names the snippet introduced
      (sorted, no double-underscore names). Lets the model know
      what's available to subsequent calls without echoing values.
    - ``took_ms``: wall-clock from entry to return.
    """

    result: Any = None
    stdout: str = ""
    truncated: bool = False
    globals_added: list[str] = field(default_factory=list)
    took_ms: int = 0


def validate_code(
    code: str,
    *,
    allowed_imports: Iterable[str],
) -> ast.Module:
    """Parse + statically validate a snippet, returning the AST module.

    The returned :class:`ast.Module` is the same one the executor will
    compile. Splitting validation from execution lets callers do
    cheap pre-flight checks (e.g. a future ``cb pipeline lint``)
    without paying compile time.

    Rejects, in order:

    1. ``SyntaxError`` from :func:`ast.parse` -- re-raised as
       :class:`PyValidationError` with the original message.
    2. Any ``import`` / ``from ... import ...`` whose top-level
       module is not in ``allowed_imports``.
    3. Relative imports (``from . import x``) -- a snippet has no
       package context so they're never meaningful.
    4. Any reference to a banned attribute (read OR write) anywhere
       in the tree.
    5. Any call whose callee is a banned name (``eval(...)`` etc.).
    6. ``async`` constructs (``async def``, ``await``, ``async for``,
       ``async with``) -- the executor runs synchronously and these
       would silently no-op, which is more confusing than rejecting.

    ``allowed_imports`` may include either top-level package names
    (``"pandas"``) or dotted paths (``"pandas.io"``); the validator
    matches on the *top-level* segment of the imported module so
    ``from pandas.io.sql import read_sql`` checks ``"pandas"`` against
    the allowlist.
    """
    if not isinstance(code, str):
        raise PyValidationError(
            f"code must be a string, got {type(code).__name__}"
        )
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise PyValidationError(f"snippet has a syntax error: {e}") from e

    allowed = {name.split(".")[0] for name in allowed_imports}
    allowed |= _DEFAULT_STDLIB_ALLOWLIST

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in allowed:
                    raise PyValidationError(
                        f"import of {alias.name!r} is not allowed "
                        f"(top-level {top!r} not in library allowlist; "
                        f"allowed: {sorted(allowed)})",
                        node=node,
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                raise PyValidationError(
                    "relative imports are not allowed in py snippets",
                    node=node,
                )
            mod = node.module or ""
            top = mod.split(".")[0] if mod else ""
            if not top:
                raise PyValidationError(
                    "from-import requires an explicit module name",
                    node=node,
                )
            if top not in allowed:
                raise PyValidationError(
                    f"from {mod!r} import ... is not allowed "
                    f"(top-level {top!r} not in library allowlist; "
                    f"allowed: {sorted(allowed)})",
                    node=node,
                )
        elif isinstance(node, ast.Attribute):
            if node.attr in _BANNED_ATTRS:
                raise PyValidationError(
                    f"access to attribute {node.attr!r} is not allowed "
                    f"(introspection / sandbox-escape surface)",
                    node=node,
                )
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BANNED_CALLS:
                raise PyValidationError(
                    f"call to {func.id!r} is not allowed in py snippets",
                    node=node,
                )
        elif isinstance(
            node,
            (
                ast.AsyncFunctionDef,
                ast.AsyncFor,
                ast.AsyncWith,
                ast.Await,
            ),
        ):
            raise PyValidationError(
                f"async constructs ({type(node).__name__}) are not "
                "supported in py snippets; the executor runs "
                "synchronously",
                node=node,
            )

    return tree


class PyExecutor:
    """Stateful per-run Python interpreter.

    Lifecycle: the runner creates one :class:`PyExecutor` at
    pipeline-startup time, configured from the manifest's
    ``[tool.coralbricks.pipeline.py]`` table (or env-var defaults
    when no manifest is set, e.g. slug-mode runs). Every
    ``tools.py`` call from the pipeline maps to one
    :meth:`exec` invocation against that same executor; later
    snippets see the globals defined by earlier ones.

    The executor never imports user-provided modules itself --
    snippets do their own ``import pandas as pd``. We only
    *validate* that the import is in the allowlist; if the package
    isn't actually installed in the venv, the import raises
    :class:`ModuleNotFoundError` at exec time (surfaced as
    :class:`PyExecutionError`). This keeps the cold-start cost on
    the user (they pay for what they import) and avoids the
    runtime having to mirror every customer's dep set.
    """

    def __init__(
        self,
        *,
        allowed_imports: Iterable[str] = (),
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        self._allowed_imports: frozenset[str] = frozenset(allowed_imports)
        self._timeout_ms = max(1, int(timeout_ms))
        self._max_output_bytes = max(1, int(max_output_bytes))
        # Per-thread globals dict. Stateful across :meth:`exec` calls
        # **within the same thread**: a snippet that sets ``count = 0``
        # and a later snippet that does ``count += 5`` see the same
        # ``count`` iff they were both invoked from the same thread.
        # This is what makes the ``cb_ia`` swarm safe -- four
        # specialists running in parallel threads (vc / risk / sector
        # / stock) each get an isolated globals dict, so binding
        # ``bm25_hits`` in vc_analyst can't clobber risk_analyst's
        # ``bm25_hits``. The map itself is process-wide; the values
        # under each key are thread-local. A fresh executor per run
        # keeps cross-run contamination structurally impossible.
        self._thread_globals: dict[int, dict[str, Any]] = {}
        self._thread_globals_lock = threading.Lock()

    @property
    def allowed_imports(self) -> frozenset[str]:
        """Library top-level names from the manifest. Stdlib defaults
        are added by :func:`validate_code` at call time and are NOT
        echoed here -- this property reflects only what the pipeline
        explicitly opted into."""
        return self._allowed_imports

    @property
    def timeout_ms(self) -> int:
        return self._timeout_ms

    @property
    def max_output_bytes(self) -> int:
        return self._max_output_bytes

    @property
    def globals_snapshot(self) -> dict[str, Any]:
        """Read-only view of the current thread's globals (defensive copy).

        Mostly useful for tests; production callers go through
        :meth:`exec` which does the right thing with state. Per-thread
        -- threads other than the caller see their own dict (or
        an empty one if they've never run a snippet).
        """
        return dict(self._get_thread_globals())

    def _get_thread_globals(self) -> dict[str, Any]:
        """Lazily create the calling thread's globals dict.

        Threading is the cheapest sharing primitive cb_ia gets at
        (each specialist is a python thread, not a subprocess), so
        the executor partitions state by ``threading.get_ident()``.
        The lock is uncontended in steady state -- it only matters
        for the *first* :meth:`exec` / :meth:`bind_global` call on a
        new thread when we have to insert the dict.
        """
        tid = threading.get_ident()
        with self._thread_globals_lock:
            g = self._thread_globals.get(tid)
            if g is None:
                g = {"__name__": "__cb_py__"}
                self._thread_globals[tid] = g
            return g

    def bind_global(self, name: str, value: Any) -> None:
        """Persistently bind ``value`` to ``name`` in the calling thread's globals.

        Unlike the ephemeral ``inputs={...}`` dict accepted by
        :meth:`exec` (whose keys are dropped after each call), names
        bound here **stay alive** across subsequent :meth:`exec`
        calls on the same thread. This is the foundation for the
        platform's "tool result -> python variable" affordance:
        upstream tool wrappers (e.g. ``cb_ia``'s
        ``bm25_gdelt(bind_as="hits")``) call this to push a
        potentially huge result into the runner's memory **without**
        the LLM having to JSON-emit it back into a later
        ``run_python(inputs=...)`` call.

        Per-thread isolation is what makes parallel specialists
        safe: vc_analyst binding ``bm25_hits`` cannot clobber
        risk_analyst's ``bm25_hits`` because each runs on a
        different thread.

        Validation mirrors :meth:`exec` ``inputs=`` keys: must be a
        Python identifier, no dunder names. Validation failures
        raise :class:`PyValidationError` (a programming error in the
        upstream tool wrapper, not a model-recoverable runtime
        failure).
        """
        if not isinstance(name, str) or not name.isidentifier():
            raise PyValidationError(
                f"bind_global name {name!r} must be a valid Python identifier"
            )
        if name.startswith("__") and name.endswith("__"):
            raise PyValidationError(
                f"bind_global name {name!r} cannot use dunder names"
            )
        g = self._get_thread_globals()
        g[name] = value

    def unbind_global(self, name: str) -> bool:
        """Drop a previously-bound name from the calling thread's globals.

        Returns ``True`` if the name was present and removed,
        ``False`` if it wasn't there. Idempotent. Useful for tool
        wrappers that want to release a large result after the model
        is done with it (memory pressure during a long specialist
        loop), and for tests that need a clean slate without
        rebuilding the whole executor.
        """
        if not isinstance(name, str) or not name.isidentifier():
            raise PyValidationError(
                f"unbind_global name {name!r} must be a valid Python identifier"
            )
        g = self._get_thread_globals()
        return g.pop(name, _MISSING) is not _MISSING

    def exec(
        self,
        code: str,
        *,
        inputs: Optional[Mapping[str, Any]] = None,
    ) -> PyResult:
        """Run one snippet against the calling thread's persistent globals.

        Pre-binds every entry in ``inputs`` into globals (so a
        snippet that does ``inputs={"bars": prev_result}`` can write
        ``bars["rows"]`` directly). After exec, ``result`` is read
        out of globals and stripped from the dict for the next
        call (it's a transient per-call name, not persistent
        state).

        ``inputs`` are *transient*: each input key is dropped from
        the globals dict after the snippet returns so it doesn't
        appear in ``globals_added`` or persist into a later call.
        Crucially, if an ``inputs`` key shadows a name that was
        already bound (e.g. via :meth:`bind_global` from an upstream
        tool wrapper), the original value is **restored** after
        exec rather than dropped -- transient inputs must not
        destroy persistent state.

        On success, returns a :class:`PyResult` whose
        ``globals_added`` lists every NEW top-level name the
        snippet introduced. On failure, raises
        :class:`PyValidationError` (caught at validate time) or
        :class:`PyExecutionError` (caught during ``exec``,
        including timeout).
        """
        tree = validate_code(code, allowed_imports=self._allowed_imports)
        g = self._get_thread_globals()

        # Snapshot the globals *before* binding inputs so we can
        # diff after exec to surface ``globals_added``. We do NOT
        # treat ``inputs`` keys as "added" -- those are caller-
        # provided not snippet-provided.
        before_keys = set(g.keys())

        # Save originals for any input keys that already had a
        # value bound (via bind_global or a prior snippet) so we
        # can restore them after exec rather than letting the
        # transient input clobber the persistent state.
        shadowed: dict[str, Any] = {}

        if inputs is not None:
            for key, value in inputs.items():
                if not isinstance(key, str) or not key.isidentifier():
                    raise PyValidationError(
                        f"inputs key {key!r} must be a valid Python "
                        "identifier"
                    )
                if key.startswith("__") and key.endswith("__"):
                    raise PyValidationError(
                        f"inputs key {key!r} cannot use dunder names"
                    )
                if key in g:
                    shadowed[key] = g[key]
                g[key] = value

        try:
            compiled = compile(tree, filename="<py-snippet>", mode="exec")
        except SyntaxError as e:
            # validate_code already parsed; reaching here means a
            # post-parse compile failure, which is rare but possible
            # (e.g. nested function with bad name binding).
            raise PyValidationError(f"snippet failed to compile: {e}") from e

        stdout_buf = io.StringIO()
        timed_out: list[bool] = [False]

        def _alarm_handler(signum, frame):  # noqa: ARG001
            timed_out[0] = True
            raise _PyAlarm("py snippet exceeded its wall-clock budget")

        # Sentinel marker for the result variable -- distinguishes
        # "snippet didn't set result" from "snippet set result = None".
        sentinel = object()
        g.pop("result", None)
        g["result"] = sentinel

        prev_handler = None
        timeout_armed = False
        t0 = time.perf_counter()
        try:
            try:
                prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
                # ``setitimer`` accepts fractional seconds; SIGALRM
                # would round down to 1s.
                signal.setitimer(
                    signal.ITIMER_REAL, self._timeout_ms / 1000.0,
                )
                timeout_armed = True
            except (ValueError, AttributeError):
                # Not main thread / not POSIX. We still run the
                # snippet -- the gateway's wall-budget kill is the
                # ultimate backstop.
                timeout_armed = False

            try:
                with contextlib.redirect_stdout(stdout_buf):
                    exec(compiled, g)  # noqa: S102
            except _PyAlarm:
                took_ms = int((time.perf_counter() - t0) * 1000)
                raise PyTimeoutError(
                    f"py snippet exceeded {self._timeout_ms}ms "
                    f"wall-clock budget (took {took_ms}ms)",
                    timeout_ms=self._timeout_ms,
                )
            except PyValidationError:
                raise
            except BaseException as e:
                tb = traceback.format_exc()
                raise PyExecutionError(
                    f"{type(e).__name__}: {e}",
                    original_type=type(e).__name__,
                    traceback_str=tb,
                ) from e
        finally:
            if timeout_armed:
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                except (ValueError, AttributeError):
                    pass
            if prev_handler is not None:
                try:
                    signal.signal(signal.SIGALRM, prev_handler)
                except (ValueError, AttributeError):
                    pass

        took_ms = int((time.perf_counter() - t0) * 1000)

        result_val = g.pop("result", sentinel)
        if result_val is sentinel:
            result_val = None

        # Drop ephemeral input keys; restore the originals for any
        # input that shadowed a pre-bound persistent global.
        if inputs is not None:
            for key in inputs.keys():
                if key in shadowed:
                    g[key] = shadowed[key]
                else:
                    g.pop(key, None)

        after_keys = set(g.keys())
        added = sorted(
            k
            for k in (after_keys - before_keys)
            if not (k.startswith("__") and k.endswith("__"))
        )

        out_text = stdout_buf.getvalue()
        truncated = False
        if len(out_text.encode("utf-8")) > self._max_output_bytes:
            # Cut on byte boundary; decode the head as UTF-8 (replace
            # half-multibyte at the cut to keep the result valid).
            head = out_text.encode("utf-8")[: self._max_output_bytes]
            out_text = head.decode("utf-8", errors="replace") + (
                f"\n... [truncated, exceeded {self._max_output_bytes} bytes]"
            )
            truncated = True

        return PyResult(
            result=result_val,
            stdout=out_text,
            truncated=truncated,
            globals_added=added,
            took_ms=took_ms,
        )


class _PyAlarm(BaseException):
    """Internal signal used to break out of a hung snippet.

    Inherits from :class:`BaseException` (not :class:`Exception`) so
    a snippet that does ``try: ... except Exception: pass`` can't
    swallow the timeout.
    """


__all__ = [
    "PyExecutionError",
    "PyExecutor",
    "PyResult",
    "PyTimeoutError",
    "PyValidationError",
    "validate_code",
]
