"""``coralbricks.sandbox.cancel_event`` -- cooperative cancellation.

The gateway can SIGTERM a sandbox subprocess at any time (wall
timeout, user clicked cancel in the UI, tenant quota exhausted,
...). For pipelines that want to *cooperatively* notice cancellation
between tool calls -- for example to write a partial result before
shutdown -- we expose a tiny check that simply looks for a sentinel
file the gateway creates.

The sentinel-file approach is deliberately dumb:

* the sandbox can't trust signals to fire in any particular order
  vs. its Python frames,
* a file check is two syscalls and works identically across
  Linux/macOS development and Linux production,
* the gateway already needs a per-run scratch dir, so dropping a
  marker file in there costs nothing.

Sentinel path comes from ``$CORAL_CANCEL_FILE`` (set by the runner
before exec). When unset (e.g. unit tests that don't bother
setting it), :func:`is_set` always returns ``False``.
"""

from __future__ import annotations

import os

CANCEL_FILE_ENV_VAR = "CORAL_CANCEL_FILE"


def is_set() -> bool:
    """Return ``True`` once the gateway has flagged this run as cancelled.

    Pipelines should call this between their natural checkpoints
    (after each tool call, after each LLM round-trip, ...) and exit
    cleanly when it returns ``True``. Long CPU-bound stretches that
    can't poll will still be killed by the gateway's SIGTERM/SIGKILL
    process-group machinery.
    """
    path = os.environ.get(CANCEL_FILE_ENV_VAR)
    if not path:
        return False
    try:
        return os.path.exists(path)
    except OSError:
        # If the FS is gone we're being torn down anyway.
        return True


__all__ = ["CANCEL_FILE_ENV_VAR", "is_set"]
