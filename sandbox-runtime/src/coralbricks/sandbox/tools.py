"""``coralbricks.sandbox.tools`` -- the Coral-managed tool kernel.

Every helper here is a one-line wrapper over an RPC call to the
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

Slice 5a only ships :func:`ping`. :func:`search` and friends land in
slice 5b once the gateway-side ``ToolKernel`` has real implementations
to dispatch to.
"""

from __future__ import annotations

from typing import Any, Optional

from . import _rpc


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


__all__ = ["ping"]
