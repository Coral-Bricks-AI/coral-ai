"""`coralbricks sync <source>` — run a local Airbyte sync and upload to S3.

Flow per run:

  1. POST /cli/v1/runs with {sourceId}. Backend refreshes OAuth tokens,
     assembles the source config, creates a SyncRun row, and mints STS
     session credentials scoped to exactly this run's S3 prefix.
  2. docker pull <image>
  3. docker run discover → ConfiguredCatalog built locally.
  4. docker run read → dispatch RECORD/STATE/LOG/TRACE messages.
     RECORDs buffer into a gzip stream per discovered stream.
  5. On stream-close, upload all gzipped JSONL parts to S3 under
     users/{uid}/sources/{src}/conn-{cid}/runs/{rid}/{stream}/part-NNNN.jsonl.gz.
  6. POST /cli/v1/runs/{runId}/complete with the totals (and error, if any)
     so the run history stays in step across the CLI and managed paths.

Records land in S3 as the raw Airbyte Protocol inner record —
`{stream, data, emitted_at}` — no Coral envelope wrapping.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import threading
import time
from typing import Any

import click

from .. import config as cfg_mod
from .. import tui
from ..api import ApiError, AuthError, Client
from ..runner import (
    ReadHandlers,
    RunnerError,
    build_configured_catalog,
    discover_streams,
    get_docker_client,
    pull_image,
    run_read,
)
from ..s3 import ScopedS3Writer, StsCredentials

_SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")
_ALLOWED_IMAGE_PREFIX = "airbyte/"
_DEFAULT_ALLOWED_BUCKETS = ("coralbricks-connectors",)
_ENV_ALLOWED_BUCKETS = "CORALBRICKS_ALLOWED_BUCKETS"


def _allowed_buckets() -> tuple[str, ...]:
    """Buckets the CLI will accept from the backend's /runs response.

    Defaults to the prod bucket. Overridable via env for local testing
    (e.g. `CORALBRICKS_ALLOWED_BUCKETS=coralbricks-dev-logs`).
    """
    raw = os.environ.get(_ENV_ALLOWED_BUCKETS, "").strip()
    if not raw:
        return _DEFAULT_ALLOWED_BUCKETS
    parsed = tuple(b.strip() for b in raw.split(",") if b.strip())
    return parsed or _DEFAULT_ALLOWED_BUCKETS


@click.command("sync")
@click.argument("source_id")
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Stream every Airbyte log line instead of the live per-stream table.",
)
def sync_cmd(source_id: str, verbose: bool) -> None:
    """Run a local Airbyte sync for a connected source."""
    if not _SOURCE_ID_RE.match(source_id):
        raise click.ClickException(
            f"Invalid source id: {source_id!r}. Use lowercase letters, digits, '-' or '_'."
        )

    cfg = cfg_mod.load()
    client = Client(cfg)

    run_blob = _start_run(client, source_id)
    _validate_run_blob(run_blob)

    run_id = int(run_blob["runId"])
    image = str(run_blob["dockerImage"])
    config_json = run_blob["configJson"]
    bucket = str(run_blob["s3Bucket"])
    key_prefix = str(run_blob["s3KeyPrefix"])
    stream_selection = run_blob.get("streamSelection", "all")
    sts_payload = run_blob["stsCredentials"]

    click.echo()
    click.secho(f"Syncing {source_id}", bold=True)
    tui.kv(
        [
            ("run", click.style(f"#{run_id}", fg="cyan")),
            ("image", image),
            ("destination", f"s3://{bucket}/{key_prefix}/"),
        ]
    )
    click.echo()

    try:
        writer_summary, end_cursor, trace_error, stream_errors = _execute_run(
            image=image,
            config_json=config_json,
            stream_selection=stream_selection,
            bucket=bucket,
            key_prefix=key_prefix,
            sts=StsCredentials(
                access_key_id=sts_payload["accessKeyId"],
                secret_access_key=sts_payload["secretAccessKey"],
                session_token=sts_payload["sessionToken"],
                expiration=sts_payload.get("expiration"),
            ),
            verbose=verbose,
        )
    except RunnerError as e:
        message = _short_error(str(e))
        _report_failed(client, run_id, key_prefix, None, message)
        _print_failure_summary(source_id, run_id, None, message, {}, verbose)
        raise click.exceptions.Exit(1) from e
    except Exception as e:  # noqa: BLE001 — last-resort so /complete still fires
        _report_failed(client, run_id, key_prefix, None, f"unexpected: {e}")
        raise

    # Airbyte marks the whole run failed if any stream errors, even when
    # the other streams succeeded and their records made it to S3. From
    # the user's perspective that's a success-with-warnings, not a
    # failure — treat it that way iff we actually uploaded data.
    records_written = int(writer_summary.get("records", 0)) if writer_summary else 0
    if trace_error and records_written == 0:
        message = _short_error(
            str(trace_error.get("message") or "Airbyte source emitted a TRACE/ERROR")
        )
        _report_failed(client, run_id, key_prefix, writer_summary, message)
        _print_failure_summary(source_id, run_id, writer_summary, message, stream_errors, verbose)
        raise click.exceptions.Exit(1)

    _report_success(client, run_id, key_prefix, writer_summary, end_cursor)
    _print_success(source_id, run_id, writer_summary, stream_errors if trace_error else {})


# ---------- backend calls ----------


def _start_run(client: Client, source_id: str) -> dict[str, Any]:
    try:
        resp = client.post("/cli/v1/runs", json={"sourceId": source_id})
    except AuthError as e:
        raise click.ClickException(e.message) from e
    except ApiError as e:
        raise click.ClickException(f"Failed to start run ({e.status}): {e.message}") from e
    if not isinstance(resp, dict):
        raise click.ClickException("Backend returned malformed /runs response")
    return resp


def _validate_run_blob(blob: dict[str, Any]) -> None:
    """Defensive checks before we trust the backend response.

    The CLI runs on the user's machine and will act on whatever image
    and S3 destination this blob names; clamp down on the interesting
    fields to contain a compromised / misconfigured backend response.
    """
    required = ("runId", "dockerImage", "configJson", "s3Bucket", "s3KeyPrefix", "stsCredentials")
    for key in required:
        if key not in blob:
            raise click.ClickException(f"Backend /runs response missing {key!r}")

    image = str(blob["dockerImage"])
    if not image.startswith(_ALLOWED_IMAGE_PREFIX):
        raise click.ClickException(
            f"Refusing to run non-Airbyte image: {image!r} (expected {_ALLOWED_IMAGE_PREFIX}*)"
        )

    bucket = str(blob["s3Bucket"])
    allowed = _allowed_buckets()
    if bucket not in allowed:
        raise click.ClickException(
            f"Unexpected destination bucket: {bucket!r} (allowed: {', '.join(allowed)})"
        )

    prefix = str(blob["s3KeyPrefix"])
    if not prefix.startswith("users/"):
        raise click.ClickException(f"Unexpected key prefix shape: {prefix!r}")


def _report_success(
    client: Client,
    run_id: int,
    key_prefix: str,
    summary: dict[str, Any],
    end_cursor: dict[str, Any] | None,
) -> None:
    body: dict[str, Any] = {
        "status": "success",
        "recordsWritten": int(summary.get("records", 0)),
        # `/complete` accepts bytes as string to avoid JSON number-precision loss.
        "bytesWritten": str(int(summary.get("bytes", 0))),
        "s3KeyPrefix": key_prefix,
    }
    if end_cursor is not None:
        body["endCursor"] = end_cursor
    try:
        client.post(f"/cli/v1/runs/{run_id}/complete", json=body)
    except (ApiError, AuthError) as e:
        tui.warn(f"Sync finished but /complete failed: {e}")


def _report_failed(
    client: Client,
    run_id: int,
    key_prefix: str,
    summary: dict[str, Any] | None,
    message: str,
) -> None:
    # Best effort — the run is already broken; don't swallow the
    # original cause by raising on a follow-up reporting failure.
    # We still forward records/bytes so the history row reflects the
    # partial-data we uploaded before the source aborted.
    body: dict[str, Any] = {"status": "failed", "errorMessage": message}
    if summary:
        body["recordsWritten"] = int(summary.get("records", 0))
        body["bytesWritten"] = str(int(summary.get("bytes", 0)))
        body["s3KeyPrefix"] = key_prefix
    with contextlib.suppress(ApiError, AuthError):
        client.post(f"/cli/v1/runs/{run_id}/complete", json=body)


def _short_error(raw: str) -> str:
    """Condense Airbyte's multi-line error text into one useful line.

    Source connectors love to wrap a root-cause into a traceback-ish
    blob (AirbyteTracedException + HTTP response headers + cookies).
    The signal is almost always the first ~200 chars of the first
    non-empty line; the rest is noise that makes the error unreadable
    in both the terminal and the run-history row.
    """
    if not raw:
        return "unspecified error"
    for line in raw.splitlines():
        line = line.strip()
        if line:
            break
    else:
        return "unspecified error"
    return line if len(line) <= 240 else line[:237] + "…"


# ---------- local runner orchestration ----------


# Airbyte Python CDK log patterns we parse to drive the live table:
#
#   "Finished syncing <name>"                 → stream flipped to ✓
#   "Exception while syncing stream <name>"   → stream flipped to ✗
#
# These aren't protocol-level signals, but they've been stable across
# CDK-based connectors for years. If a non-CDK connector doesn't emit
# them, streams just stay on the spinner until container exit (where
# `stop()` finalizes them) — a safe degradation.
_FINISHED_SYNCING_RE = re.compile(r"Finished syncing (\S+)")
_EXCEPTION_SYNCING_RE = re.compile(r"Exception while syncing stream (\S+)")

# Most connector-level errors worth surfacing are HTTP failures logged
# by the Python CDK in this exact shape. Extracting (status, message)
# gives us a compact reason we can tuck onto the ✗ row of the live
# table instead of dumping the 500-line traceback that follows.
_HTTP_ERR_RE = re.compile(r"status code '(\d+)'.*error message: '([^']+)'")


def _extract_short_reason(raw: str) -> str:
    """Condense a raw Airbyte error log into a ≤120-char cause.

    Prefers the HTTP status + error message when the connector logs
    in that shape; otherwise trims to the first non-empty line.
    """
    if not raw:
        return "unknown error"
    m = _HTTP_ERR_RE.search(raw)
    if m:
        reason = f"{m.group(1)}: {m.group(2)}"
        return reason if len(reason) <= 120 else reason[:117] + "…"
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= 120 else line[:117] + "…"
    return "unknown error"


def _execute_run(
    *,
    image: str,
    config_json: dict[str, Any],
    stream_selection: str | list[str],
    bucket: str,
    key_prefix: str,
    sts: StsCredentials,
    verbose: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, dict[str, str]]:
    """Pull image, discover, read, upload.

    Returns (summary, end_cursor, trace_error, stream_errors). Per-stream
    failure reasons are extracted from the log stream regardless of
    mode; only the rendering differs (live table vs. plain logs).
    """
    docker_client = get_docker_client()

    # Pull is quick and produces at most a few "status" lines; keep it
    # visible in both modes so the user sees motion during image fetch.
    tui.hint(f"Pulling {image}…")
    last_progress: list[str] = []

    def on_progress(status: str) -> None:
        if last_progress and last_progress[-1] == status:
            return
        last_progress.append(status)
        click.secho(f"  docker: {status}", dim=True)

    pull_image(docker_client, image, on_progress=on_progress)

    tui.hint("Discovering streams…")
    discover = discover_streams(docker_client, image, config_json)
    if not discover.streams:
        raise RunnerError("Source emitted an empty catalog — nothing to sync.")

    configured = build_configured_catalog(discover.streams, stream_selection)
    stream_names = [s["stream"].get("name", "?") for s in configured["streams"]]
    tui.hint(f"Reading {len(stream_names)} stream(s): {', '.join(stream_names)}")

    writer = ScopedS3Writer(bucket=bucket, key_prefix=key_prefix, sts=sts)
    latest_state: dict[str, Any] | None = None
    # Per-stream root cause, distilled to ≤120 chars (e.g. "403:
    # Insufficient permissions"). Populated by on_log when it sees
    # "Exception while syncing stream X"; the *cause* sits on the
    # previous ERROR frame, so we keep the last raw error around to
    # attach to whichever stream gets named next.
    stream_errors: dict[str, str] = {}
    last_error_raw: list[str | None] = [None]

    # Live-table mode is opt-out (via -v) and only engages when stdout
    # is a real terminal — piping or CI should fall through to the
    # plain log stream so nothing is lost in a non-interactive context.
    live = _LiveStreamTable(stream_names) if (not verbose and sys.stdout.isatty()) else None

    record_count = [0]
    last_log_print = [time.monotonic()]

    def on_record(stream: str, data: dict[str, Any], emitted_at: int | None) -> None:
        writer.write_record(stream, data, emitted_at)
        record_count[0] += 1
        if live is not None:
            live.bump(stream)
            return
        # Verbose / non-TTY path: pulse a single-line counter ~1 Hz so
        # the user can see motion without the terminal going full-gusher.
        now = time.monotonic()
        if now - last_log_print[0] > 1.0:
            last_log_print[0] = now
            click.secho(f"  · {record_count[0]} records", dim=True)

    def on_state(state: dict[str, Any]) -> None:
        nonlocal latest_state
        latest_state = state

    def on_log(level: str, message: str) -> None:
        lvl = level.upper()
        # Always harvest stream lifecycle + error reasons; the render
        # path (live vs. verbose) is the only thing that differs.
        if lvl == "INFO":
            m = _FINISHED_SYNCING_RE.search(message)
            if m and live is not None:
                live.mark_done(m.group(1))
        elif lvl in ("ERROR", "FATAL"):
            m = _EXCEPTION_SYNCING_RE.search(message)
            if m:
                stream = m.group(1)
                reason = (
                    _extract_short_reason(last_error_raw[0])
                    if last_error_raw[0]
                    else None
                )
                if reason and stream not in stream_errors:
                    stream_errors[stream] = reason
                if live is not None:
                    live.mark_failed(stream, reason)
            else:
                # Stash the raw error so the *next* "Exception while
                # syncing X" ERROR frame can pull its root cause from it.
                last_error_raw[0] = message
        # Live mode stays quiet — the per-stream table is the UI.
        # Verbose (or non-TTY) dumps every log line colored by level.
        if live is None:
            color = {"ERROR": "red", "FATAL": "red", "WARN": "yellow"}.get(lvl)
            click.secho(
                f"  [{lvl}] {_short_error(message)}",
                fg=color,
                dim=color is None,
                err=True,
            )

    if live is not None:
        live.start()
    stats = None
    try:
        stats = run_read(
            docker_client,
            image,
            config_json,
            configured,
            ReadHandlers(on_record=on_record, on_state=on_state, on_log=on_log),
        )
    finally:
        if live is not None:
            overall_failed = stats is None or (
                stats.trace_error is not None or stats.exit_code != 0
            )
            live.stop(overall_failed=overall_failed)

    # Always flush whatever we buffered. A zero-record run still writes
    # zero-record files so the prefix exists — makes the verify-step
    # directory listing meaningful even for empty syncs.
    summary = writer.close()

    if stats.trace_error is not None:
        return summary, latest_state, stats.trace_error, stream_errors
    if stats.exit_code != 0:
        return (
            summary,
            latest_state,
            {"message": f"read exited {stats.exit_code}\n{stats.stderr_tail}"},
            stream_errors,
        )
    return summary, latest_state, None, stream_errors


class _LiveStreamTable:
    """Per-stream progress pane that updates in place during `read`.

    Renders a block of N lines (one per stream) using ANSI cursor-up
    (`\\x1b[NF`) to overwrite itself on every refresh. A daemon ticker
    thread advances the spinner at 10 Hz so quiet streams still look
    alive; RECORD arrivals bump the per-stream counter synchronously.

    State transitions:
      pending → running  on first RECORD for that stream
      running → done     on INFO "Finished syncing <name>", or on stop()
      *       → failed   on ERROR "Exception while syncing stream <name>"
    """

    _SPINNER = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, streams: list[str]) -> None:
        self._streams = list(streams)
        self._counts: dict[str, int] = {s: 0 for s in self._streams}
        self._status: dict[str, str] = {s: "pending" for s in self._streams}
        self._reasons: dict[str, str] = {}
        self._frame = 0
        self._rendered_lines = 0
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._render()
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()

    def bump(self, stream: str) -> None:
        with self._lock:
            if stream in self._counts:
                self._counts[stream] += 1
                if self._status[stream] == "pending":
                    self._status[stream] = "running"

    def mark_done(self, stream: str) -> None:
        with self._lock:
            if stream in self._status and self._status[stream] != "failed":
                self._status[stream] = "done"

    def mark_failed(self, stream: str, reason: str | None = None) -> None:
        with self._lock:
            if stream in self._status:
                self._status[stream] = "failed"
                # Keep the *first* reason — later ERROR frames are
                # usually retries of the same root cause.
                if reason and stream not in self._reasons:
                    self._reasons[stream] = reason

    def stop(self, overall_failed: bool = False) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        # Container closed stdout → anything still animating is resolved.
        # If the run itself failed (top-level trace error or non-zero
        # exit) streams that never got a per-stream error frame are
        # marked failed without a reason — honest "we don't know why
        # this one didn't finish" rather than a misleading ✓.
        with self._lock:
            for name in self._streams:
                if self._status[name] in ("pending", "running"):
                    self._status[name] = "failed" if overall_failed else "done"
        self._render()

    def _tick(self) -> None:
        while not self._stop_evt.wait(0.1):
            with self._lock:
                self._frame = (self._frame + 1) % len(self._SPINNER)
            self._render()

    def _render(self) -> None:
        if not self._streams:
            return
        with self._lock:
            out: list[str] = []
            if self._rendered_lines:
                # CPL — cursor previous line N times (column 1).
                out.append(f"\x1b[{self._rendered_lines}F")
            name_w = max(len(s) for s in self._streams)
            for name in self._streams:
                status = self._status[name]
                count = self._counts[name]
                if status == "pending":
                    icon = click.style("·", dim=True)
                    count_text = click.style("pending", dim=True)
                elif status == "running":
                    icon = click.style(self._SPINNER[self._frame], fg="cyan")
                    count_text = f"{count} records"
                elif status == "failed":
                    icon = click.style("✗", fg="red", bold=True)
                    reason = self._reasons.get(name)
                    suffix = reason if reason else "failed"
                    count_text = click.style(
                        f"{count} records · {suffix}", fg="red"
                    )
                else:
                    icon = click.style("✓", fg="green", bold=True)
                    count_text = f"{count} records"
                # `\x1b[K` clears any leftover chars from a previous
                # longer render on this line.
                out.append(f"  {icon} {name.ljust(name_w)}  {count_text}\x1b[K\n")
            sys.stdout.write("".join(out))
            sys.stdout.flush()
            self._rendered_lines = len(self._streams)


# ---------- terminal output ----------


def _print_failure_summary(
    source_id: str,
    run_id: int,
    summary: dict[str, Any] | None,
    message: str,
    stream_errors: dict[str, str],
    verbose: bool,
) -> None:
    """Pretty single block that replaces the raw Airbyte error dump.

    On failure the CLI shows: the top-level reason, the per-stream
    reasons we distilled during the run, and — if any records landed
    before the source aborted — a table of what made it to S3. That's
    enough to diagnose the common failures (auth scope, rate limit,
    transient 5xx) without exposing the Python traceback the source
    connector produces. Users who want the firehose re-run with `-v`.
    """
    # When we've already distilled a reason for each failed stream, the
    # top-line message is almost always a duplicate of Airbyte's noisy
    # "During the sync, the following streams did not sync successfully"
    # envelope. Compact it to "N stream(s) failed" and let the per-stream
    # rows carry the signal; fall back to the raw message otherwise so a
    # top-level TRACE/ERROR (auth, config) still reaches the user.
    headline = (
        f"{len(stream_errors)} stream{'s' if len(stream_errors) != 1 else ''} failed"
        if stream_errors
        else message
    )
    click.echo()
    tui.warn(
        f"Sync failed for {click.style(source_id, fg='cyan', bold=True)} "
        f"(run #{run_id}): {headline}"
    )
    if stream_errors:
        click.echo()
        for name, reason in stream_errors.items():
            click.secho(
                f"  ✗ {name}: {reason}",
                fg="red",
            )

    records = int(summary.get("records", 0)) if summary else 0
    streams: dict[str, dict[str, Any]] = summary.get("streams", {}) if summary else {}
    if records > 0 and streams:
        click.echo()
        tui.hint(
            f"Partial data uploaded: {records} records "
            f"({_human_bytes(summary.get('bytes', 0) if summary else 0)})"
        )
        rows = [
            (
                name,
                str(info.get("records", 0)),
                _human_bytes(info.get("bytes", 0)),
                _human_bytes(info.get("gzippedBytes", 0)),
            )
            for name, info in streams.items()
        ]
        click.echo()
        tui.table(rows, headers=("stream", "records", "raw", "gzipped"))

    if not verbose:
        click.echo()
        tui.hint("Rerun with --verbose to see the full Airbyte log.")
    click.echo()


def _print_success(
    source_id: str,
    run_id: int,
    summary: dict[str, Any],
    stream_warnings: dict[str, str] | None = None,
) -> None:
    click.echo()
    headline = (
        f"Synced {click.style(source_id, fg='cyan', bold=True)}  "
        f"run #{run_id} — "
        f"{summary.get('records', 0)} records "
        f"({_human_bytes(summary.get('bytes', 0))})"
    )
    if stream_warnings:
        tui.ok(f"{headline}  ({len(stream_warnings)} skipped)")
        for name, reason in stream_warnings.items():
            click.secho(f"  ! {name}: {reason}", fg="yellow")
    else:
        tui.ok(headline)

    streams: dict[str, dict[str, Any]] = summary.get("streams", {})
    if streams:
        rows = [
            (
                name,
                str(info.get("records", 0)),
                _human_bytes(info.get("bytes", 0)),
                _human_bytes(info.get("gzippedBytes", 0)),
            )
            for name, info in streams.items()
        ]
        click.echo()
        tui.table(rows, headers=("stream", "records", "raw", "gzipped"))
    click.echo()


def _human_bytes(n: Any) -> str:
    try:
        n_int = int(n)
    except (TypeError, ValueError):
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n_int < 1024:
            return f"{n_int} {unit}" if unit == "B" else f"{n_int:.1f} {unit}"
        n_int = n_int / 1024  # type: ignore[assignment]
    return f"{n_int:.1f} TB"
