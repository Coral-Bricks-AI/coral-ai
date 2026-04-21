"""coralbricks.sandbox -- in-sandbox shim for Coral Bricks pipelines.

Pipeline code that runs inside a Coral sandbox imports this package
to talk to the gateway::

    from coralbricks.sandbox import tools, llm, cancel_event

    hits = tools.search(index="aapl_10k_2024", query="cash flow", top_k=5)
    answer = llm.chat(model="...", messages=[...])
    if cancel_event.is_set():
        return {"status": "cancelled"}

The gateway pre-installs this package into every per-pipeline venv
alongside the user's own pipeline package and mounts a per-run
Unix-domain socket at ``$CORAL_GATEWAY_SOCKET``. All public helpers
exposed from this namespace are thin RPC wrappers over that socket;
no third-party dependencies, no global state beyond the env var.

This package is **not** the path for third-party SaaS the pipeline
brings its own credentials for (Langfuse, LangSmith, custom DBs,
...). Those go through the manifest's ``egress`` allowlist and the
pipeline imports those SDKs directly. ``coralbricks.sandbox`` is
specifically for *Coral-managed* services: the indices the gateway
owns and the OSS-LLM keys the gateway holds.

See ``plans/04_SANDBOXES.md`` and ``plans/05_AGENTS_AND_PIPELINES.md``
in the closed-source platform repo for the full architecture.
"""

from __future__ import annotations

from . import cancel_event, llm, py_executor, tools
from ._rpc import (
    DEFAULT_RPC_TIMEOUT_S,
    RpcCallError,
    RpcConnectError,
    RpcError,
    RpcProtocolError,
    SOCKET_ENV_VAR,
)
from .py_executor import (
    PyExecutionError,
    PyExecutor,
    PyResult,
    PyTimeoutError,
    PyValidationError,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_RPC_TIMEOUT_S",
    "PyExecutionError",
    "PyExecutor",
    "PyResult",
    "PyTimeoutError",
    "PyValidationError",
    "RpcCallError",
    "RpcConnectError",
    "RpcError",
    "RpcProtocolError",
    "SOCKET_ENV_VAR",
    "__version__",
    "cancel_event",
    "llm",
    "py_executor",
    "tools",
]
