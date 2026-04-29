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
     airbyte/users/{uid}/sources/{src}/conn-{cid}/runs/{rid}/{stream}/part-NNNN.jsonl.gz.
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
from rich.console import Group
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from .. import config as cfg_mod
from .. import tui
from ..api import ApiError, AuthError, Client
from coralbricks.sync_core.runner import (
    ReadHandlers,
    RunnerError,
    build_configured_catalog,
    discover_streams,
    get_docker_client,
    pull_image,
    run_read,
)
from coralbricks.sync_core.s3 import ScopedS3Writer, StsCredentials

_SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")
_ALLOWED_IMAGE_PREFIX = "airbyte/"
_DEFAULT_ALLOWED_BUCKETS = ("coralbricks-connectors",)
_ENV_ALLOWED_BUCKETS = "CORALBRICKS_ALLOWED_BUCKETS"
_TOTAL_PHASES = 4


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
    help="Stream every Airbyte log line instead of the live per-stream panel.",
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

    started_at = time.monotonic()

    _render_run_header(source_id, run_id, image, bucket, key_prefix)

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
        _print_failure_summary(source_id, run_id, None, message, {}, verbose, started_at)
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
        _print_failure_summary(
            source_id, run_id, writer_summary, message, stream_errors, verbose, started_at
        )
        raise click.exceptions.Exit(1)

    _report_success(client, run_id, key_prefix, writer_summary, end_cursor)
    _print_success(
        source_id,
        run_id,
        bucket,
        key_prefix,
        writer_summary,
        stream_errors if trace_error else {},
        started_at,
    )


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
    if not prefix.startswith("airbyte/users/"):
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
# panel instead of dumping the 500-line traceback that follows.
_HTTP_ERR_RE = re.compile(r"status code '(\d+)'.*error message: '([^']+)'")


def _extract_short_reason(raw: str) -> str:
    """Condense a raw Airbyte error log into a ≤120-char cause."""
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
    mode; only the rendering differs (live panel vs. plain logs).
    """
    docker_client = get_docker_client()

    # --- Phase 1: pull
    t0 = time.monotonic()
    tui.phase(1, _TOTAL_PHASES, f"Pulling {image}")
    seen_progress: set[str] = set()

    def on_progress(status: str) -> None:
        if status in seen_progress:
            return
        seen_progress.add(status)
        tui.console.print(Text("    " + status, style="dim"))

    pull_image(docker_client, image, on_progress=on_progress)
    tui.console.print(
        Text("    " + tui.CHECK + " image ready  ", style="dim").append(
            f"{time.monotonic() - t0:.1f}s", style="dim"
        )
    )

    # --- Phase 2: discover
    t1 = time.monotonic()
    tui.phase(2, _TOTAL_PHASES, "Discovering streams")
    discover = discover_streams(docker_client, image, config_json)
    if not discover.streams:
        raise RunnerError("Source emitted an empty catalog — nothing to sync.")
    configured = build_configured_catalog(discover.streams, stream_selection)
    stream_names = [s["stream"].get("name", "?") for s in configured["streams"]]
    tui.console.print(
        Text(f"    {tui.CHECK} {len(stream_names)} stream(s) discovered  ", style="dim").append(
            f"{time.monotonic() - t1:.1f}s", style="dim"
        )
    )

    # --- Phase 3: read
    tui.phase(3, _TOTAL_PHASES, "Reading records")

    writer = ScopedS3Writer(bucket=bucket, key_prefix=key_prefix, sts=sts)
    latest_state: dict[str, Any] | None = None
    stream_errors: dict[str, str] = {}
    last_error_raw: list[str | None] = [None]

    # Live panel is opt-out (-v) and only engages when stdout is a real
    # terminal — piping or CI falls through to the plain log stream so
    # nothing is lost in a non-interactive context.
    live = _LivePanel(stream_names) if (not verbose and sys.stdout.isatty()) else None

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
            tui.console.print(Text(f"    · {record_count[0]:,} records", style="dim"))

    def on_state(state: dict[str, Any]) -> None:
        nonlocal latest_state
        latest_state = state

    def on_log(level: str, message: str) -> None:
        lvl = level.upper()
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
        if live is None:
            color = {"ERROR": "red", "FATAL": "red", "WARN": "yellow"}.get(lvl)
            tui.console.print(
                Text("    [" + lvl + "] " + _short_error(message), style=color or "dim")
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

    # --- Phase 4: upload (writer.close flushes any pending parts)
    tui.phase(4, _TOTAL_PHASES, "Uploading to S3")
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


class _LivePanel:
    """Live per-stream progress panel powered by rich.Live.

    Renders a Panel containing a Table with one row per stream:
    icon · name · records · bytes · note. The icon column animates a
    rich Spinner for running streams; rich.Live re-renders the panel
    at 10 Hz so the spinner advances on its own.

    State transitions:
      pending → running  on first RECORD for that stream
      running → done     on INFO "Finished syncing <name>", or on stop()
      *       → failed   on ERROR "Exception while syncing stream <name>"
    """

    def __init__(self, streams: list[str]) -> None:
        self._streams = list(streams)
        self._counts: dict[str, int] = {s: 0 for s in self._streams}
        self._status: dict[str, str] = {s: "pending" for s in self._streams}
        self._reasons: dict[str, str] = {}
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._live: Live | None = None
        # Reused so rich.Spinner can animate uniformly across all running rows.
        self._spinner = Spinner("dots", style="cyan")

    # ----- lifecycle -----

    def start(self) -> None:
        self._live = Live(
            self._render(),
            console=tui.console,
            refresh_per_second=10,
            transient=False,
        )
        self._live.start()

    def stop(self, overall_failed: bool = False) -> None:
        with self._lock:
            for name in self._streams:
                if self._status[name] in ("pending", "running"):
                    self._status[name] = "failed" if overall_failed else "done"
        if self._live is not None:
            # One last render so the final state lands in the buffer.
            self._live.update(self._render(), refresh=True)
            self._live.stop()
            self._live = None

    # ----- updates -----

    def bump(self, stream: str) -> None:
        with self._lock:
            if stream in self._counts:
                self._counts[stream] += 1
                if self._status[stream] == "pending":
                    self._status[stream] = "running"
        self._refresh()

    def mark_done(self, stream: str) -> None:
        with self._lock:
            if stream in self._status and self._status[stream] != "failed":
                self._status[stream] = "done"
        self._refresh()

    def mark_failed(self, stream: str, reason: str | None = None) -> None:
        with self._lock:
            if stream in self._status:
                self._status[stream] = "failed"
                if reason and stream not in self._reasons:
                    self._reasons[stream] = reason
        self._refresh()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    # ----- rendering -----

    def _render(self) -> Group:
        with self._lock:
            elapsed = time.monotonic() - self._started
            total = sum(self._counts.values())
            rate = total / elapsed if elapsed > 0.05 else 0.0
            done = sum(1 for s in self._streams if self._status[s] == "done")
            failed = sum(1 for s in self._streams if self._status[s] == "failed")
            running = sum(1 for s in self._streams if self._status[s] == "running")

            header = Text()
            header.append(f"{done}/{len(self._streams)} done", style="dim")
            header.append("   ")
            header.append(f"{running} running", style="cyan")
            if failed:
                header.append("   ")
                header.append(f"{failed} failed", style="red")
            header.append("   ")
            header.append(f"{total:,} records", style="white")
            header.append("   ")
            header.append(f"{rate:,.0f}/s", style="dim")
            header.append("   ")
            header.append(f"{elapsed:.1f}s", style="dim")

            t = Table(box=None, show_edge=False, pad_edge=False, padding=(0, 2), show_header=False)
            t.add_column("icon", width=2)
            t.add_column("name", overflow="ellipsis")
            t.add_column("count", justify="right")
            t.add_column("note", overflow="ellipsis")

            for name in self._streams:
                status = self._status[name]
                count = self._counts[name]
                if status == "pending":
                    icon: Any = Text("·", style="dim")
                    count_t = Text("pending", style="dim")
                    note_t: Any = Text("")
                elif status == "running":
                    icon = self._spinner
                    count_t = Text(f"{count:,}", style="white")
                    note_t = Text("syncing", style="cyan")
                elif status == "failed":
                    icon = Text(tui.CROSS, style="bold red")
                    count_t = Text(f"{count:,}", style="white")
                    note_t = Text(self._reasons.get(name, "failed"), style="red")
                else:  # done
                    icon = Text(tui.CHECK, style="bold green")
                    count_t = Text(f"{count:,}", style="white")
                    note_t = Text("complete", style="green")
                t.add_row(icon, Text(name, style="white"), count_t, note_t)

            return Group(header, Text(""), t)


# ---------- terminal output ----------


def _render_run_header(
    source_id: str, run_id: int, image: str, bucket: str, key_prefix: str
) -> None:
    body = tui.kv_renderable(
        [
            ("run", Text(f"#{run_id}", style="cyan")),
            ("image", Text(image, style="white")),
            ("destination", Text(f"s3://{bucket}/{key_prefix}/", style="dim")),
        ]
    )
    tui.blank()
    tui.panel(
        body,
        title=f"Sync · {source_id}",
        title_extra=tui.pill("RUNNING", "running"),
    )
    tui.blank()


def _print_failure_summary(
    source_id: str,
    run_id: int,
    summary: dict[str, Any] | None,
    message: str,
    stream_errors: dict[str, str],
    verbose: bool,
    started_at: float,
) -> None:
    """Branded failure panel — replaces the raw Airbyte traceback dump."""
    duration = time.monotonic() - started_at
    headline = (
        f"{len(stream_errors)} stream{'s' if len(stream_errors) != 1 else ''} failed"
        if stream_errors
        else message
    )

    lines: list[Any] = []
    lines.append(
        Text("✗ ", style="err").append(Text(headline, style="bold")).append(
            f"   {duration:.1f}s", style="dim"
        )
    )

    if stream_errors:
        lines.append(Text(""))
        for name, reason in stream_errors.items():
            line = Text("  ")
            line.append(tui.CROSS + " ", style="bold red")
            line.append(name, style="bold")
            line.append(": ", style="dim")
            line.append(reason, style="red")
            lines.append(line)

    records = int(summary.get("records", 0)) if summary else 0
    streams: dict[str, dict[str, Any]] = summary.get("streams", {}) if summary else {}
    if records > 0 and streams:
        lines.append(Text(""))
        lines.append(
            Text(
                f"Partial data uploaded: {records:,} records "
                f"({_human_bytes(summary.get('bytes', 0) if summary else 0)})",
                style="dim",
            )
        )
        rows = [
            (
                Text(name, style=f"bold {tui.CORAL}"),
                Text(f"{int(info.get('records', 0)):,}"),
                Text(_human_bytes(info.get("bytes", 0))),
                Text(_human_bytes(info.get("gzippedBytes", 0))),
            )
            for name, info in streams.items()
        ]
        lines.append(Text(""))
        lines.append(tui.table_renderable(rows, headers=("stream", "records", "raw", "gzipped")))

    if not verbose:
        lines.append(Text(""))
        lines.append(Text("rerun with --verbose for the full Airbyte log", style="dim"))

    tui.blank()
    tui.panel(
        Group(*lines),
        title=f"Sync failed · {source_id} · run #{run_id}",
        title_extra=tui.pill("FAILED", "fail"),
        accent="red",
    )
    tui.blank()


def _print_success(
    source_id: str,
    run_id: int,
    bucket: str,
    key_prefix: str,
    summary: dict[str, Any],
    stream_warnings: dict[str, str] | None,
    started_at: float,
) -> None:
    duration = time.monotonic() - started_at
    records = int(summary.get("records", 0))
    bytes_total = int(summary.get("bytes", 0))
    rate = records / duration if duration > 0.05 else 0.0
    streams: dict[str, dict[str, Any]] = summary.get("streams", {})

    # A run that completes the protocol but uploads zero records is the
    # classic restricted-API-key signature: the connector authenticates
    # fine but every list endpoint silently returns []. Surface it as a
    # warning instead of a green checkmark so the user actually notices.
    is_empty = records == 0 and not stream_warnings

    lines: list[Any] = []
    headline = Text()
    if is_empty:
        headline.append("! ", style="warn")
        headline.append("Synced ", style="white")
        headline.append(source_id, style=f"bold {tui.CORAL}")
        headline.append("  ")
        headline.append("0 records", style="bold yellow")
        headline.append("  ·  ", style="dim")
        headline.append(f"{duration:.1f}s", style="white")
    else:
        headline.append(tui.CHECK + " ", style="ok")
        headline.append("Synced ", style="white")
        headline.append(source_id, style=f"bold {tui.CORAL}")
        headline.append("  ")
        headline.append(f"{records:,} records", style="white")
        headline.append("  ·  ", style="dim")
        headline.append(_human_bytes(bytes_total), style="white")
        headline.append("  ·  ", style="dim")
        headline.append(f"{duration:.1f}s", style="white")
        if rate:
            headline.append("  ·  ", style="dim")
            headline.append(f"{rate:,.0f}/s", style="dim")
    lines.append(headline)

    if is_empty:
        lines.append(Text(""))
        lines.append(
            Text(
                "No data was returned by any stream. Common causes:",
                style="yellow",
            )
        )
        lines.append(Text("  · the account is genuinely empty", style="dim"))
        lines.append(
            Text(
                "  · the credentials are too restricted (e.g. a Stripe rk_live_… key with no read scopes)",
                style="dim",
            )
        )
        lines.append(
            Text(
                "  · you connected a sandbox/test account that has no data",
                style="dim",
            )
        )
        lines.append(Text(""))
        line = Text("  Try ", style="dim")
        line.append(f"coralbricks connect {source_id}", style=f"bold {tui.CORAL}")
        line.append(" again with a full-access key.", style="dim")
        lines.append(line)

    if stream_warnings:
        lines.append(Text(""))
        lines.append(
            Text(f"{len(stream_warnings)} stream(s) skipped:", style="yellow")
        )
        for name, reason in stream_warnings.items():
            line = Text("  ! ", style="warn")
            line.append(name, style="bold")
            line.append(": ", style="dim")
            line.append(reason, style="yellow")
            lines.append(line)

    if streams:
        rows = [
            (
                Text(name, style=f"bold {tui.CORAL}"),
                Text(f"{int(info.get('records', 0)):,}"),
                Text(_human_bytes(info.get("bytes", 0))),
                Text(_human_bytes(info.get("gzippedBytes", 0))),
            )
            for name, info in streams.items()
        ]
        lines.append(Text(""))
        lines.append(tui.table_renderable(rows, headers=("stream", "records", "raw", "gzipped")))

    lines.append(Text(""))
    lines.append(
        Text("uploaded to ", style="dim").append(
            f"s3://{bucket}/{key_prefix}/", style="white"
        )
    )

    tui.blank()
    tui.panel(
        Group(*lines),
        title=f"Sync · {source_id} · run #{run_id}",
        title_extra=tui.pill("EMPTY" if is_empty else "SUCCESS", "warn" if is_empty else "success"),
        accent="yellow" if is_empty else "green",
    )
    tui.blank()


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
