"""Airbyte source Docker runner.

Pulls an Airbyte source image and invokes its protocol commands
(`check`, `discover`, `read`) locally on the user's machine. Follows
the Airbyte Protocol Docker contract: one container per invocation,
newline-delimited JSON on stdout, logs on stderr.

Mount point is `/config` (not `/airbyte`). The Airbyte Platform itself
uses `/config/` for the exact same reason we do: every Airbyte source
image bakes its runtime at `/airbyte/integration_code/`, so mounting
over `/airbyte` shadows the connector binary and the container fails
to init ("mkdir /airbyte/integration_code: read-only file system").

See: https://docs.airbyte.com/platform/understanding-airbyte/airbyte-protocol-docker

This module is deliberately free of backend/S3 concerns — Day 2 of M3
runs the full pull→discover→read loop locally before any upload path
is wired in. Higher layers (commands/sync.py, s3.py) compose it.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import docker
    from docker.errors import APIError, DockerException, ImageNotFound, NotFound
except ImportError as e:
    raise ImportError(
        "The 'docker' Python package is required. Run: pip install 'docker>=7.0'"
    ) from e


# Airbyte Protocol message types we care about. `read` emits RECORD/STATE/
# LOG/TRACE; `discover` emits CATALOG; `check` emits CONNECTION_STATUS;
# `spec` emits SPEC. We branch on `type` for the first four in iter_read_messages.
MSG_RECORD = "RECORD"
MSG_STATE = "STATE"
MSG_LOG = "LOG"
MSG_TRACE = "TRACE"
MSG_CATALOG = "CATALOG"
MSG_CONNECTION_STATUS = "CONNECTION_STATUS"
MSG_SPEC = "SPEC"


class RunnerError(Exception):
    """Anything we surface to the CLI user as an Airbyte / Docker failure."""


@dataclass
class DiscoverResult:
    streams: list[dict[str, Any]]
    raw_catalog: dict[str, Any]


@dataclass
class RunStats:
    exit_code: int
    stderr_tail: str = ""
    trace_error: dict[str, Any] | None = None


@dataclass
class AirbyteMessage:
    type: str
    raw: dict[str, Any]
    # Convenience pointers into `raw` for the types we branch on.
    record: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    log: dict[str, Any] | None = None
    trace: dict[str, Any] | None = None


# ---------- docker daemon preflight ----------


def get_docker_client() -> docker.DockerClient:
    """Open a Docker client + ping the daemon, with a friendly error on failure.

    Docker Desktop / Engine must be running — every subsequent step in
    the sync depends on it, so failing here gives the best UX.
    """
    try:
        client = docker.from_env()
    except DockerException as e:
        raise RunnerError(
            "Couldn't connect to Docker. Is Docker Desktop running?\n"
            "  Install: https://docs.docker.com/get-docker/"
        ) from e
    try:
        client.ping()
    except (APIError, DockerException) as e:
        raise RunnerError(
            "Docker daemon is not responding. Start Docker Desktop and retry."
        ) from e
    return client


# ---------- image pull ----------


def pull_image(
    client: docker.DockerClient,
    image: str,
    on_progress: Callable[[str], None] | None = None,
) -> None:
    """Pull an image, streaming progress lines to `on_progress`.

    docker-py's low-level API emits one JSON event per layer state change;
    we render a compact one-line summary per unique "status" value so the
    user sees motion without the full layer matrix.
    """
    try:
        repo, _, tag = image.partition(":")
        tag = tag or "latest"
        last_status: str | None = None
        for event in client.api.pull(repo, tag=tag, stream=True, decode=True):
            if "error" in event:
                raise RunnerError(f"docker pull failed: {event['error']}")
            status = event.get("status")
            if status and status != last_status:
                last_status = status
                if on_progress:
                    on_progress(status)
    except ImageNotFound as e:
        raise RunnerError(f"Image not found on Docker Hub: {image}") from e
    except APIError as e:
        raise RunnerError(f"docker pull failed for {image}: {e}") from e


# ---------- running a protocol command ----------


@contextmanager
def _airbyte_workdir(
    config: dict[str, Any], catalog: dict[str, Any] | None = None
) -> Iterator[Path]:
    """Prepare a mode-0700 tempdir with `config.json` + optional `catalog.json`.

    Mounted read-only at `/config` in the container (see module docstring
    for why not `/airbyte`). We never write a `state.json` in v0.1
    (full_refresh only).
    """
    tmp = tempfile.mkdtemp(prefix="coralbricks-airbyte-")
    try:
        os.chmod(tmp, stat.S_IRWXU)
        root = Path(tmp)
        (root / "config.json").write_text(json.dumps(config))
        if catalog is not None:
            (root / "catalog.json").write_text(json.dumps(catalog))
        yield root
    finally:
        # Best-effort cleanup — config contains tokens.
        for child in Path(tmp).glob("*"):
            with suppress(OSError):
                child.unlink()
        with suppress(OSError):
            os.rmdir(tmp)


def _run_protocol_command(
    client: docker.DockerClient,
    image: str,
    command: list[str],
    workdir: Path,
    *,
    on_stdout_line: Callable[[str], None],
) -> RunStats:
    """Start a source container, stream stdout JSON lines, collect stderr.

    Airbyte protocol requires `-i` (stdin open) — we pass `stdin_open=True`.
    Volume is mounted read-only: the source never needs to write back to
    the host. Network is the default bridge (no host/none override) so
    the source can reach the provider's API.
    """
    container = None
    try:
        container = client.containers.create(
            image=image,
            command=command,
            volumes={str(workdir): {"bind": "/config", "mode": "ro"}},
            stdin_open=True,
            tty=False,
            detach=True,
        )
        container.start()

        # Stdout stream — line-buffered JSON. stderr is collected after
        # wait() so LOG frames don't get mixed into the JSON stream.
        buf = b""
        for chunk in container.logs(stream=True, follow=True, stdout=True, stderr=False):
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    on_stdout_line(text)
        if buf.strip():
            on_stdout_line(buf.decode("utf-8", errors="replace").strip())

        wait_result = container.wait()
        exit_code = int(wait_result.get("StatusCode", 1))
        stderr_raw = container.logs(stdout=False, stderr=True) or b""
        stderr_text = stderr_raw.decode("utf-8", errors="replace")
        stderr_tail = "\n".join(stderr_text.strip().splitlines()[-20:])
        return RunStats(exit_code=exit_code, stderr_tail=stderr_tail)
    except NotFound as e:
        raise RunnerError(f"Image {image!r} is not present locally — pull it first.") from e
    except APIError as e:
        raise RunnerError(f"Docker API error running {image}: {e}") from e
    finally:
        if container is not None:
            with suppress(APIError):
                container.remove(force=True)


def _parse_message(line: str) -> AirbyteMessage | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    mtype = payload.get("type")
    if not isinstance(mtype, str):
        return None
    return AirbyteMessage(
        type=mtype,
        raw=payload,
        record=payload.get("record") if mtype == MSG_RECORD else None,
        state=payload.get("state") if mtype == MSG_STATE else None,
        log=payload.get("log") if mtype == MSG_LOG else None,
        trace=payload.get("trace") if mtype == MSG_TRACE else None,
    )


# ---------- discover ----------


def discover_streams(
    client: docker.DockerClient,
    image: str,
    config: dict[str, Any],
) -> DiscoverResult:
    """Run `discover` and return the catalog's streams.

    The source emits a single CATALOG message (possibly preceded by
    LOG / TRACE messages). We collect them all and pick the CATALOG.
    Protocol reference:
    https://docs.airbyte.com/platform/understanding-airbyte/airbyte-protocol#airbytemessage
    """
    catalog_payload: dict[str, Any] | None = None
    trace_error: dict[str, Any] | None = None

    def on_line(text: str) -> None:
        nonlocal catalog_payload, trace_error
        msg = _parse_message(text)
        if msg is None:
            return
        if msg.type == MSG_CATALOG:
            catalog_payload = msg.raw.get("catalog") or {}
        elif msg.type == MSG_TRACE and isinstance(msg.trace, dict) and msg.trace.get("type") == "ERROR":
            trace_error = msg.trace.get("error") or {}

    with _airbyte_workdir(config) as workdir:
        stats = _run_protocol_command(
            client,
            image,
            ["discover", "--config", "/config/config.json"],
            workdir,
            on_stdout_line=on_line,
        )

    if trace_error:
        raise RunnerError(
            f"discover failed: {trace_error.get('message') or trace_error}"
        )
    if stats.exit_code != 0:
        raise RunnerError(
            f"discover exited {stats.exit_code}\n{stats.stderr_tail or '(no stderr)'}"
        )
    if not catalog_payload:
        raise RunnerError("discover did not emit a CATALOG message")

    streams = catalog_payload.get("streams") or []
    if not isinstance(streams, list):
        raise RunnerError("discover returned a malformed catalog (streams not a list)")
    return DiscoverResult(streams=streams, raw_catalog=catalog_payload)


# ---------- ConfiguredCatalog ----------


def build_configured_catalog(
    streams: list[dict[str, Any]],
    stream_selection: str | list[str],
) -> dict[str, Any]:
    """Turn a discovered catalog into a ConfiguredCatalog for `read`.

    v0.1: every selected stream runs full_refresh/overwrite. Streams that
    don't advertise `full_refresh` in `supported_sync_modes` are skipped
    rather than failing the run — rare, but possible for CDC-only sources.

    Reference fields per:
    https://docs.airbyte.com/platform/understanding-airbyte/airbyte-protocol#configuredairbytestream
    """
    if stream_selection == "all":
        selected = streams
    else:
        wanted = {s for s in stream_selection}
        selected = [s for s in streams if s.get("name") in wanted]
        if not selected:
            raise RunnerError(
                f"None of the requested streams were discovered: {sorted(wanted)}"
            )

    configured: list[dict[str, Any]] = []
    skipped: list[str] = []
    for stream in selected:
        modes = stream.get("supported_sync_modes") or ["full_refresh"]
        if "full_refresh" not in modes:
            skipped.append(stream.get("name") or "?")
            continue
        configured.append(
            {
                "stream": stream,
                "sync_mode": "full_refresh",
                "destination_sync_mode": "overwrite",
            }
        )

    if not configured:
        raise RunnerError(
            "No streams support full_refresh. "
            f"Skipped (incremental-only): {skipped}"
        )
    return {"streams": configured}


# ---------- read ----------


@dataclass
class ReadHandlers:
    on_record: Callable[[str, dict[str, Any], int | None], None]
    on_state: Callable[[dict[str, Any]], None]
    on_log: Callable[[str, str], None] = field(
        default=lambda level, message: None
    )


def run_read(
    client: docker.DockerClient,
    image: str,
    config: dict[str, Any],
    configured_catalog: dict[str, Any],
    handlers: ReadHandlers,
) -> RunStats:
    """Run `read` and dispatch each protocol message to the right handler.

    RECORD  → handlers.on_record(stream, data, emitted_at)
    STATE   → handlers.on_state(state_payload)   (all three shapes:
              STREAM / GLOBAL / LEGACY — caller decides how to interpret)
    LOG     → handlers.on_log(level, message)
    TRACE/ERROR → captured in returned RunStats.trace_error; run considered failed

    Records are forwarded as the inner `record` object (drops the redundant
    `type:"RECORD"` envelope but keeps `stream`, `data`, `emitted_at` etc.)
    so callers can persist them verbatim.
    """
    trace_error: dict[str, Any] | None = None

    def on_line(text: str) -> None:
        nonlocal trace_error
        msg = _parse_message(text)
        if msg is None:
            return
        if msg.type == MSG_RECORD and isinstance(msg.record, dict):
            stream = msg.record.get("stream")
            data = msg.record.get("data")
            emitted = msg.record.get("emitted_at")
            if isinstance(stream, str) and isinstance(data, dict):
                handlers.on_record(
                    stream,
                    data,
                    emitted if isinstance(emitted, int) else None,
                )
        elif msg.type == MSG_STATE and isinstance(msg.state, dict):
            handlers.on_state(msg.state)
        elif msg.type == MSG_LOG and isinstance(msg.log, dict):
            level = str(msg.log.get("level") or "INFO")
            message = str(msg.log.get("message") or "")
            handlers.on_log(level, message)
        elif msg.type == MSG_TRACE and isinstance(msg.trace, dict) and msg.trace.get("type") == "ERROR":
            trace_error = msg.trace.get("error") or {"message": "unspecified TRACE error"}

    with _airbyte_workdir(config, catalog=configured_catalog) as workdir:
        stats = _run_protocol_command(
            client,
            image,
            [
                "read",
                "--config",
                "/config/config.json",
                "--catalog",
                "/config/catalog.json",
            ],
            workdir,
            on_stdout_line=on_line,
        )

    stats.trace_error = trace_error
    return stats
