"""``coralbricks.sandbox.tools`` -- the Coral-managed tool kernel.

Most helpers here are one-line wrappers over an RPC call to the
gateway. The gateway owns:

* the search/graph indices (it built them, it can read them, the
  sandbox cannot mount them),
* the per-tenant ACLs that say which indices a given pipeline run
  is allowed to touch,
* the metering/billing for any tool that costs money.

Because the gateway holds the credentials, pipelines never need to
ship API keys for these tools. The set of tools a pipeline is
allowed to call is declared in its manifest's ``[tool.coralbricks.pipeline].tools``
list; calls to undeclared tools will surface as :class:`RpcCallError`
with ``remote_type="ToolNotAllowed"``.

The one verb that is **NOT** an RPC call is :func:`py`. Per
``plans/08_TOOL_KERNEL_VERBS.md`` § ``py``, the Python interpreter
runs inside the runner subprocess so that snippets can read prior
tool-call results out of the in-process Python objects directly
(rather than re-shipping potentially huge dataframes back over the
wire) and so that arbitrary code execution never reaches the
gateway's address space (which holds tenant credentials). The local
:class:`PyExecutor` is configured from environment variables the
runner sets at pipeline-startup time.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Mapping, Optional, Sequence

from . import _rpc
from .py_executor import (
    PyExecutionError,
    PyExecutor,
    PyResult,
    PyTimeoutError,
    PyValidationError,
)

# Env vars the platform runner sets at pipeline-startup time so the
# in-runner :class:`PyExecutor` knows what the manifest opted into.
# All three are optional -- absent values fall back to the executor's
# documented defaults so slug-mode runs (no manifest) still work.
PY_LIBRARIES_ENV_VAR = "CORAL_PY_LIBRARIES"
"""Comma-separated top-level package names the snippet may import,
e.g. ``"pandas,numpy,duckdb"``. Stdlib basics
(:data:`coralbricks.sandbox.py_executor._DEFAULT_STDLIB_ALLOWLIST`)
are always allowed in addition to whatever this lists."""

PY_TIMEOUT_MS_ENV_VAR = "CORAL_PY_TIMEOUT_MS"
"""Per-call wall-clock budget. SIGALRM-based; main-thread only."""

PY_MAX_OUTPUT_BYTES_ENV_VAR = "CORAL_PY_MAX_OUTPUT_BYTES"
"""Per-call cap on captured stdout. The ``result`` value is not
truncated by this; it's bounded by the kernel envelope upstream."""

# Process-singleton executor + lock. Lazy-init on first call so
# importing :mod:`coralbricks.sandbox.tools` is free for pipelines
# that never use ``tools.py``.
_PY_EXECUTOR: Optional[PyExecutor] = None
_PY_EXECUTOR_LOCK = threading.Lock()


def ping(
    *,
    socket_path: Optional[str] = None,
    timeout_s: float = _rpc.DEFAULT_RPC_TIMEOUT_S,
) -> dict[str, Any]:
    """Round-trip the gateway to prove the RPC channel is up.

    Returns the dict the gateway echoes back, which currently includes
    at least ``{"pong": True, "request_id": "<uuid>"}``. Pipelines
    don't normally call this; it exists so the platform's smoke tests
    have a zero-side-effect probe.
    """
    return _rpc.call(
        "tools.ping",
        params={},
        socket_path=socket_path,
        timeout_s=timeout_s,
    )


def list_tools(
    *,
    socket_path: Optional[str] = None,
    timeout_s: float = _rpc.DEFAULT_RPC_TIMEOUT_S,
) -> dict[str, Any]:
    """Return the per-run tool allowlist + index registry snapshot.

    Useful for pipelines that want to validate or surface their
    available verbs / indices before issuing a real call. The wire
    shape mirrors the gateway's
    :meth:`gateway.rpc.tool_kernel.ToolKernel.list_tools`::

        {
          "tools": ["bm25", "ann", "sql", ...],
          "indices": [
            {
              "slug": "gdelt_events_v2",
              "description": "GDELT event-news",
              "hardware": "cpu",
              "capabilities": {
                "bm25": {
                  "fields": [
                    {"name": "title", "type": "text", "boost": 3.0,
                     "description": "Article headline"},
                    ...
                  ],
                  "default_fields": ["title^3", "body^2"]
                },
                "multihop": {
                  "predicates": [
                    {"name": "MENTIONED_IN", "description": "..."},
                    ...
                  ],
                  "node_types": [...]
                }
              }
            }
          ],
          "unrestricted": False
        }

    The ``capabilities`` blob is the typed schema declared at index
    registration time; pipelines like ``cb_ia`` read it at startup
    and render the field / predicate / column lists into their
    model-facing tool descriptions so the model never has to guess
    field names. Calls with unknown names get rejected by the
    gateway with :class:`ToolPolicyError` listing the valid set, so
    the model can self-correct on the next turn.
    """
    return _rpc.call(
        "tools.list",
        params={},
        socket_path=socket_path,
        timeout_s=timeout_s,
    )


def bm25(
    *,
    index: str,
    query: str,
    k: int = 10,
    fields: Optional[Sequence[str]] = None,
    filters: Optional[Mapping[str, Any]] = None,
    socket_path: Optional[str] = None,
    timeout_s: float = _rpc.DEFAULT_RPC_TIMEOUT_S,
) -> dict[str, Any]:
    """BM25 keyword search against a registered OpenSearch index.

    ``index`` is the slug the index was registered under; the
    gateway resolves it to the right OpenSearch endpoint + per-index
    BM25 field config. Pass ``fields`` to override the registered
    defaults (``["title^3", "body^2"]`` etc.); omit to use the
    registration's settings. ``filters`` is an optional pre-filter
    clause (OpenSearch DSL shape; passed through verbatim).

    Returns ``{"index": "<slug>", "hits": [{"id", "score",
    "source"}, ...]}``. ``k`` is clamped server-side to the index's
    per-call cap; the gateway does NOT echo the requested ``k``.
    """
    params: dict[str, Any] = {
        "index": index,
        "query": query,
        "k": int(k),
    }
    if fields is not None:
        params["fields"] = list(fields)
    if filters is not None:
        params["filters"] = dict(filters)
    return _rpc.call(
        "tools.bm25",
        params=params,
        socket_path=socket_path,
        timeout_s=timeout_s,
    )


def ann(
    *,
    index: str,
    text: str,
    k: int = 10,
    filters: Optional[Mapping[str, Any]] = None,
    socket_path: Optional[str] = None,
    timeout_s: float = _rpc.DEFAULT_RPC_TIMEOUT_S,
) -> dict[str, Any]:
    """Approximate nearest-neighbor search (HNSW).

    The pipeline passes plain ``text``; the gateway runs the
    index's configured embedder server-side and dispatches the
    vector. ``filters`` is a pre-filter clause applied before the
    vector scoring stage (OpenSearch filter shape; passed through
    verbatim today).
    """
    params: dict[str, Any] = {
        "index": index,
        "text": text,
        "k": int(k),
    }
    if filters is not None:
        params["filters"] = dict(filters)
    return _rpc.call(
        "tools.ann",
        params=params,
        socket_path=socket_path,
        timeout_s=timeout_s,
    )


def knn(
    *,
    index: str,
    text: str,
    k: int = 10,
    filters: Optional[Mapping[str, Any]] = None,
    socket_path: Optional[str] = None,
    timeout_s: float = _rpc.DEFAULT_RPC_TIMEOUT_S,
) -> dict[str, Any]:
    """Exact nearest-neighbor search (linear scan).

    Same wire shape as :func:`ann`; under the hood the gateway runs
    an exact ``script_score`` + ``cosineSimilarity`` rather than the
    HNSW path. The index registration's
    ``capabilities['knn']['max_corpus']`` gates which indices are
    eligible -- passing a too-large index surfaces as
    :class:`RpcCallError` with ``remote_type="ToolPolicyError"``.
    """
    params: dict[str, Any] = {
        "index": index,
        "text": text,
        "k": int(k),
    }
    if filters is not None:
        params["filters"] = dict(filters)
    return _rpc.call(
        "tools.knn",
        params=params,
        socket_path=socket_path,
        timeout_s=timeout_s,
    )


def get(
    *,
    index: str,
    id: str,  # noqa: A002 -- mirrors wire shape
    socket_path: Optional[str] = None,
    timeout_s: float = _rpc.DEFAULT_RPC_TIMEOUT_S,
) -> dict[str, Any]:
    """Fetch a single document from an index by id.

    Returns ``{"index", "found", "doc": {"id", "source"} | None}``.
    Heavy fields like ``embedding`` are stripped server-side; the
    body is returned at native length (callers truncate if they care).
    """
    return _rpc.call(
        "tools.get",
        params={"index": index, "id": id},
        socket_path=socket_path,
        timeout_s=timeout_s,
    )


def sql(
    *,
    index: str,
    query: str,
    socket_path: Optional[str] = None,
    timeout_s: float = _rpc.DEFAULT_RPC_TIMEOUT_S,
) -> dict[str, Any]:
    """DuckDB-over-Parquet ``SELECT``-only query against a registered index.

    The gateway validates the query is single-statement / SELECT-only
    before substituting the index registration's table->parquet URI
    map. Tables not in the registration's
    ``capabilities['sql'].tables`` allowlist are rejected. Returns
    ``{"index", "rows": [...], "row_count", "columns", "truncated",
    "took_ms"}``.
    """
    return _rpc.call(
        "tools.sql",
        params={"index": index, "query": query},
        socket_path=socket_path,
        timeout_s=timeout_s,
    )


def multihop(
    *,
    index: str,
    start_ids: Sequence[str],
    hops: int = 2,
    predicate_filter: Optional[Sequence[str]] = None,
    socket_path: Optional[str] = None,
    # Multihop on a large frontier can take longer than the default
    # RPC timeout; bump to 60s here. Callers can override.
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """BFS knowledge-graph traversal from one or more seed ids.

    ``hops`` is clamped server-side by the index's
    ``capabilities['multihop'].max_hops``. ``predicate_filter`` (if
    given) restricts traversal + return to edges whose ``relation``
    is in the set.

    Returns ``{"index", "nodes", "edges", "node_count",
    "edge_count", "hops_traversed", "hardware", "truncated",
    "took_ms"}``. Inspect ``truncated`` to decide whether to narrow
    the predicate filter; ``hardware`` reports whether the
    cuGraph (``"gpu"``) or DuckDB (``"cpu"``) tier served the call.
    """
    params: dict[str, Any] = {
        "index": index,
        "start_ids": list(start_ids),
        "hops": int(hops),
    }
    if predicate_filter is not None:
        params["predicate_filter"] = list(predicate_filter)
    return _rpc.call(
        "tools.multihop",
        params=params,
        socket_path=socket_path,
        timeout_s=timeout_s,
    )


def py(
    code: str,
    *,
    inputs: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Execute a Python snippet in the runner's stateful interpreter.

    Returns a JSON-friendly dict mirroring the kernel verb's wire
    shape:

    - ``ok`` (bool): ``True`` on success; ``False`` when the
      snippet raised. Validation errors raise
      :class:`PyValidationError` directly (they're a manifest /
      snippet bug, not a runtime failure the model should try to
      recover from).
    - ``result``: value the snippet bound to the top-level ``result``
      variable, or ``None`` if it didn't set one.
    - ``stdout`` (str): captured stdout, truncated at the
      configured ``CORAL_PY_MAX_OUTPUT_BYTES``.
    - ``truncated`` (bool): ``True`` when stdout was clipped.
    - ``globals_added`` (list[str]): top-level names the snippet
      introduced (so the model knows what's available to later
      ``tools.py`` calls without echoing the full values).
    - ``took_ms`` (int): wall-clock from entry to return.
    - ``error`` (dict | absent): present iff ``ok=False``;
      ``{"type": "<class>", "message": "<str>", "traceback": "<str>"}``.

    Unlike the RPC verbs, this never opens a socket. It runs
    in-process inside the runner subprocess; per the design doc the
    spec for the kernel verb maps cleanly onto this local-call
    surface (the model sees the same ``tools.py(code, inputs)``
    interface either way).

    Raises:
        PyValidationError: the snippet was rejected by the AST
            walker (banned import / banned attribute / etc.). This
            is a programming error, not a runtime failure; let it
            propagate.
    """
    executor = _get_or_create_py_executor()
    try:
        result = executor.exec(code, inputs=inputs)
    except PyValidationError:
        # Static rejections are programming errors -- propagate
        # rather than bury inside a structured response. Pipeline
        # authors will see a clear traceback pointing at the bad
        # line.
        raise
    except PyTimeoutError as e:
        return {
            "ok": False,
            "result": None,
            "stdout": "",
            "truncated": False,
            "globals_added": [],
            "took_ms": e.timeout_ms,
            "error": {
                "type": e.original_type,
                "message": str(e),
                "traceback": e.traceback_str,
            },
        }
    except PyExecutionError as e:
        return {
            "ok": False,
            "result": None,
            "stdout": "",
            "truncated": False,
            "globals_added": [],
            "took_ms": 0,
            "error": {
                "type": e.original_type,
                "message": str(e),
                "traceback": e.traceback_str,
            },
        }

    return {
        "ok": True,
        "result": result.result,
        "stdout": result.stdout,
        "truncated": result.truncated,
        "globals_added": result.globals_added,
        "took_ms": result.took_ms,
    }


def reset_py_executor() -> None:
    """Drop the process-global :class:`PyExecutor`.

    The next :func:`py` call will rebuild it from current env vars.
    Used by tests that flip ``CORAL_PY_LIBRARIES`` between cases;
    not part of the pipeline-author surface.
    """
    global _PY_EXECUTOR
    with _PY_EXECUTOR_LOCK:
        _PY_EXECUTOR = None


def bind_py_global(name: str, value: Any) -> None:
    """Bind ``value`` into the in-runner Python interpreter as a persistent global.

    This is the runner-side affordance that makes
    "tool result -> python variable" possible **without** the model
    having to JSON-emit the result back as a ``py(inputs={...})``
    argument. Pipeline-level tool wrappers call this after a
    kernel verb returns a large payload (full-text scraped article,
    100-row SQL result, hits list, ...): the wrapper pushes the
    full payload into the runner's interpreter under ``name`` and
    returns only a tiny summary (counts + preview) to the model.
    The model can then write::

        py(code="result = process(scraped_article_body)")

    and the snippet sees the full payload as a top-level global,
    paying zero extra LLM tokens to ferry the bytes.

    Per-thread isolation is structural: the underlying executor
    partitions globals by ``threading.get_ident()``, so calls from
    parallel worker threads (e.g. cb_ia's specialist fan-out) see
    only their own bindings.

    Raises :class:`PyValidationError` for an invalid name (not an
    identifier / dunder). Validation failures are programming
    errors in the wrapper, not runtime failures the model should
    handle.
    """
    executor = _get_or_create_py_executor()
    executor.bind_global(name, value)


def unbind_py_global(name: str) -> bool:
    """Drop a previously :func:`bind_py_global`'d name from the calling thread's globals.

    Returns ``True`` if the name was present and removed, ``False``
    if it wasn't there. Useful for tool wrappers that release a
    large bound payload after the model is done with it (long-loop
    memory pressure) and for tests that need a clean slate.
    """
    executor = _get_or_create_py_executor()
    return executor.unbind_global(name)


def _get_or_create_py_executor() -> PyExecutor:
    """Lazy-init the process-singleton executor from env vars.

    Thread-safe (the lock is uncontended in practice -- the runner
    is single-threaded -- but cheap to take). The executor's state
    (``globals_dict``) persists for the lifetime of the runner
    subprocess, which is exactly the lifetime of one pipeline run.
    """
    global _PY_EXECUTOR
    if _PY_EXECUTOR is not None:
        return _PY_EXECUTOR
    with _PY_EXECUTOR_LOCK:
        if _PY_EXECUTOR is not None:
            return _PY_EXECUTOR
        libs_raw = os.environ.get(PY_LIBRARIES_ENV_VAR, "")
        libs = tuple(
            x.strip() for x in libs_raw.split(",") if x.strip()
        )
        timeout_ms = _coerce_int_env(
            PY_TIMEOUT_MS_ENV_VAR, default=30_000,
        )
        max_output = _coerce_int_env(
            PY_MAX_OUTPUT_BYTES_ENV_VAR, default=1 * 1024 * 1024,
        )
        _PY_EXECUTOR = PyExecutor(
            allowed_imports=libs,
            timeout_ms=timeout_ms,
            max_output_bytes=max_output,
        )
        return _PY_EXECUTOR


def _coerce_int_env(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


__all__ = [
    "PY_LIBRARIES_ENV_VAR",
    "PY_MAX_OUTPUT_BYTES_ENV_VAR",
    "PY_TIMEOUT_MS_ENV_VAR",
    "ann",
    "bind_py_global",
    "bm25",
    "get",
    "knn",
    "list_tools",
    "multihop",
    "ping",
    "py",
    "reset_py_executor",
    "sql",
    "unbind_py_global",
]
