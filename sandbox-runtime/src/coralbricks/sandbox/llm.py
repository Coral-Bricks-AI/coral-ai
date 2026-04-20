"""``coralbricks.sandbox.llm`` -- the Coral-managed LLM proxy.

Inference for the OSS models the gateway pays for (DeepInfra,
Cerebras, OpenAI, Bedrock, in-house GPU servers, self-hosted SGLang)
goes through this module. The gateway holds the provider keys, so:

* the sandbox never sees an upstream API key,
* the gateway can attach per-tenant rate limits + quotas + audit
  logs to every call, and
* swapping providers (e.g. moving a model from DeepInfra to a
  self-hosted GPU pool) is a gateway-only change.

Pipelines that want to use *their own* provider keys (OpenAI direct,
Anthropic, ...) declare those hostnames in the manifest's ``egress``
allowlist and call those SDKs directly. ``coralbricks.sandbox.llm``
is specifically for the OSS models the platform manages.

Public surface:

- :func:`ping` -- connectivity probe (slice 5a).
- :func:`list_models` -- report the per-run model allowlist.
- :func:`chat` -- OpenAI-shaped ``chat.completions``; routed
  gateway-side to DeepInfra / Cerebras / OpenAI / Bedrock /
  self-hosted SGLang based on the model id.
- :func:`embed` -- batched embeddings; routed gateway-side via the
  ``coralbricks.context_prep.embedders`` factory (DeepInfra
  ``di:bge-m3``, OpenAI ``text-embedding-3-large``, Coral
  ``coral_embed``, sentence-transformers, Bedrock).

Gateway-side errors surface as :class:`RpcCallError`. Common
``remote_type`` values you might want to handle:

- ``"LLMNotAllowed"`` -- the model isn't in your manifest's
  ``models`` list. Treat as a manifest bug, not a transient failure.
- ``"LLMUnconfigured"`` -- the gateway is missing a provider key /
  an optional dependency. Operator-side; report and fail.
- ``"LLMBackendError"`` -- the provider call itself failed (network,
  rate limit, server). Safe to retry with backoff.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from . import _rpc


def ping(
    *,
    socket_path: Optional[str] = None,
    timeout_s: float = _rpc.DEFAULT_RPC_TIMEOUT_S,
) -> dict[str, Any]:
    """Round-trip the LLM proxy to prove the RPC channel is up.

    Mirror of :func:`coralbricks.sandbox.tools.ping` but routed
    through the LLM proxy's namespace so the gateway-side smoke
    tests can verify both subsystems are wired.
    """
    return _rpc.call(
        "llm.ping",
        params={},
        socket_path=socket_path,
        timeout_s=timeout_s,
    )


def list_models(
    *,
    socket_path: Optional[str] = None,
    timeout_s: float = _rpc.DEFAULT_RPC_TIMEOUT_S,
) -> dict[str, Any]:
    """Return the per-run model allowlist + unrestricted flag.

    Useful for pipelines that want to surface or validate the
    available models before issuing a real :func:`chat` /
    :func:`embed` call. Output shape::

        {"models": ["meta-llama/Meta-Llama-3.1-8B-Instruct", ...],
         "unrestricted": False}
    """
    return _rpc.call(
        "llm.list",
        params={},
        socket_path=socket_path,
        timeout_s=timeout_s,
    )


def chat(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Optional[Sequence[Mapping[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    n: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Any] = None,
    response_format: Optional[Mapping[str, Any]] = None,
    parallel_tool_calls: Optional[bool] = None,
    user: Optional[str] = None,
    socket_path: Optional[str] = None,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Run a chat completion against a Coral-managed OSS model.

    ``messages`` is the standard OpenAI shape (``[{"role": ...,
    "content": ...}]``). ``tools`` follows the OpenAI tools schema
    when provided. The gateway routes ``model`` to the right
    provider; pipelines never see the API key.

    Returns the gateway envelope::

        {"model": "<model>",
         "response": {"id": ..., "choices": [...], "usage": {...}}}

    Streaming is not supported in this slice -- pass
    ``response`` chunks through the pipeline's own framework if
    needed.
    """
    params: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
    }
    # Only forward optional knobs the caller actually set so the
    # gateway-side handler doesn't have to filter `None`s out.
    for key, value in (
        ("tools", list(tools) if tools is not None else None),
        ("tool_choice", tool_choice),
        ("temperature", temperature),
        ("top_p", top_p),
        ("max_tokens", max_tokens),
        ("max_completion_tokens", max_completion_tokens),
        ("n", n),
        ("seed", seed),
        ("stop", stop),
        ("response_format", dict(response_format) if response_format is not None else None),
        ("parallel_tool_calls", parallel_tool_calls),
        ("user", user),
    ):
        if value is not None:
            params[key] = value
    return _rpc.call(
        "llm.chat",
        params=params,
        socket_path=socket_path,
        timeout_s=timeout_s,
    )


def embed(
    *,
    model: str,
    dimension: int,
    texts: Sequence[str],
    input_type: str = "product",
    batch_size: Optional[int] = None,
    socket_path: Optional[str] = None,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Embed a batch of texts via a Coral-managed OSS embedder.

    ``model`` follows the multi-provider scheme used by
    ``coralbricks.context_prep.embedders.create_embedder``:

    - ``"di:bge-m3"`` -- DeepInfra-hosted BGE-M3.
    - ``"text-embedding-3-large"`` -- OpenAI.
    - ``"coral_embed"`` -- the in-house gateway endpoint.
    - ``"st:sentence-transformers/all-MiniLM-L6-v2"`` -- ST locally
      (only available where ``sentence-transformers`` is installed
      in the gateway venv).
    - ``"amazon.titan-embed-text-v2:0"`` -- Bedrock.

    Returns::

        {"model": "<model>", "dimension": <int>,
         "vectors": [[float, ...], ...],
         "usage": {"prompt_tokens": ..., "total_tokens": ...}}
    """
    params: dict[str, Any] = {
        "model": model,
        "dimension": int(dimension),
        "texts": list(texts),
        "input_type": input_type,
    }
    if batch_size is not None:
        params["batch_size"] = int(batch_size)
    return _rpc.call(
        "llm.embed",
        params=params,
        socket_path=socket_path,
        timeout_s=timeout_s,
    )


__all__ = ["chat", "embed", "list_models", "ping"]
