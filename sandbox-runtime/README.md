# coralbricks-sandbox-runtime

In-sandbox shim for Coral Bricks pipelines.

This is the tiny package the Coral gateway pre-installs into every
per-pipeline virtualenv alongside the user's own pipeline package. It
exposes a small, stable surface for pipeline code to talk to the
gateway over a Unix-domain socket:

```python
from coralbricks.sandbox import tools, llm, cancel_event

hits = tools.search(index="aapl_10k_2024", query="cash flow", top_k=5)
answer = llm.chat(model="moonshotai/Kimi-K2.5-Turbo", messages=[...])
if cancel_event.is_set():
    return {"status": "cancelled"}
```

Internally each call is a length-prefixed JSON RPC over the UDS the
gateway mounts at `$CORAL_GATEWAY_SOCKET`.

## Design

- **Stdlib only.** This package lands inside every sandbox venv. Every
  dependency would balloon both cold-start install time and the
  attack surface inside the sandbox. The RPC client uses `socket`,
  `json`, and `struct` -- nothing else.
- **Coral-managed services only.** This shim is the path to *Coral*
  state and *Coral*-held credentials (the indices the gateway owns,
  the OSS-LLM keys the gateway pays for). Third-party SaaS that
  pipelines bring their own credentials for (Langfuse, LangSmith,
  custom DBs, ...) goes through the manifest's `egress` allowlist
  and the pipeline imports those SDKs directly.
- **No magic global state.** Every callable accepts an explicit
  `socket_path=...` for tests and a separate `_default_client()` for
  production use that reads `$CORAL_GATEWAY_SOCKET`. The sandbox
  runner sets that env var before exec'ing the pipeline.

See `plans/04_SANDBOXES.md` and `plans/05_AGENTS_AND_PIPELINES.md` in
the closed-source platform repo for the full architecture.
