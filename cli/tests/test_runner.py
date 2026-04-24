"""Tests for runner.py — pull, discover, read against a fake docker client.

We fake the docker-py client instead of the real daemon so CI doesn't
need Docker installed. The fakes model just enough of the docker-py
surface for runner.py's call sites: `containers.create`, `container.start`,
`container.logs(stream=True, follow=True, stdout=True, stderr=False)`,
`container.wait`, and stderr `container.logs(stdout=False, stderr=True)`.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from coralbricks.cli import runner
from coralbricks.cli.runner import (
    ReadHandlers,
    RunnerError,
    build_configured_catalog,
    discover_streams,
    run_read,
)

# ---------- fake docker surface ----------


class FakeContainer:
    def __init__(self, stdout_lines: list[bytes], stderr: bytes = b"", exit_code: int = 0):
        self._stdout_lines = stdout_lines
        self._stderr = stderr
        self._exit = exit_code
        self.start = MagicMock()
        self.remove = MagicMock()

    def logs(self, *, stream: bool = False, follow: bool = False, stdout: bool = True, stderr: bool = False):
        if stream:
            # Simulate stdout stream — one chunk per line, newline preserved
            # so runner.py's buffering splits them correctly.
            def _gen():
                for line in self._stdout_lines:
                    yield line + b"\n"
            return _gen()
        if stderr and not stdout:
            return self._stderr
        return b""

    def wait(self) -> dict[str, int]:
        return {"StatusCode": self._exit}


class FakeContainerFactory:
    def __init__(self, containers: list[FakeContainer]):
        self._containers = list(containers)
        self.create_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeContainer:
        self.create_calls.append(kwargs)
        return self._containers.pop(0)


class FakeDockerClient:
    def __init__(self, containers: list[FakeContainer]):
        self.containers = FakeContainerFactory(containers)


def _encode_msg(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj).encode("utf-8")


# ---------- discover ----------


def test_discover_parses_catalog():
    catalog_msg = {
        "type": "CATALOG",
        "catalog": {
            "streams": [
                {"name": "pages", "supported_sync_modes": ["full_refresh", "incremental"]},
                {"name": "databases", "supported_sync_modes": ["full_refresh"]},
            ]
        },
    }
    fake = FakeDockerClient([FakeContainer([_encode_msg(catalog_msg)])])
    result = discover_streams(fake, "airbyte/source-notion:latest", {"start_date": "2020-01-01"})
    assert [s["name"] for s in result.streams] == ["pages", "databases"]


def test_discover_surfaces_trace_error():
    trace = {"type": "TRACE", "trace": {"type": "ERROR", "error": {"message": "401 Unauthorized"}}}
    fake = FakeDockerClient([FakeContainer([_encode_msg(trace)], exit_code=1)])
    with pytest.raises(RunnerError, match="401 Unauthorized"):
        discover_streams(fake, "airbyte/source-notion:latest", {"start_date": "2020-01-01"})


def test_discover_nonzero_exit_no_catalog():
    fake = FakeDockerClient([FakeContainer([], stderr=b"boom\n", exit_code=2)])
    with pytest.raises(RunnerError, match="discover exited 2"):
        discover_streams(fake, "airbyte/source-notion:latest", {})


# ---------- ConfiguredCatalog ----------


def test_build_configured_catalog_all():
    streams = [
        {"name": "pages", "supported_sync_modes": ["full_refresh"]},
        {"name": "databases", "supported_sync_modes": ["full_refresh"]},
    ]
    out = build_configured_catalog(streams, "all")
    assert len(out["streams"]) == 2
    for s in out["streams"]:
        assert s["sync_mode"] == "full_refresh"
        assert s["destination_sync_mode"] == "overwrite"


def test_build_configured_catalog_selection():
    streams = [
        {"name": "pages", "supported_sync_modes": ["full_refresh"]},
        {"name": "databases", "supported_sync_modes": ["full_refresh"]},
    ]
    out = build_configured_catalog(streams, ["pages"])
    assert [s["stream"]["name"] for s in out["streams"]] == ["pages"]


def test_build_configured_catalog_skips_incremental_only():
    streams = [
        {"name": "pages", "supported_sync_modes": ["full_refresh"]},
        {"name": "events", "supported_sync_modes": ["incremental"]},
    ]
    out = build_configured_catalog(streams, "all")
    names = [s["stream"]["name"] for s in out["streams"]]
    assert names == ["pages"]


def test_build_configured_catalog_empty_selection_raises():
    streams = [{"name": "pages", "supported_sync_modes": ["full_refresh"]}]
    with pytest.raises(RunnerError, match="None of the requested streams"):
        build_configured_catalog(streams, ["orphan"])


# ---------- read ----------


def test_run_read_dispatches_records_and_state():
    msgs = [
        {"type": "LOG", "log": {"level": "INFO", "message": "Starting"}},
        {"type": "RECORD", "record": {"stream": "pages", "data": {"id": "a"}, "emitted_at": 1}},
        {"type": "RECORD", "record": {"stream": "pages", "data": {"id": "b"}, "emitted_at": 2}},
        {"type": "STATE", "state": {"type": "STREAM", "stream": {"stream_descriptor": {"name": "pages"}}}},
    ]
    fake = FakeDockerClient([FakeContainer([_encode_msg(m) for m in msgs])])

    records: list[tuple[str, dict[str, Any], int | None]] = []
    states: list[dict[str, Any]] = []
    logs: list[tuple[str, str]] = []

    stats = run_read(
        fake,
        "airbyte/source-notion:latest",
        {"start_date": "2020-01-01"},
        {"streams": [{"stream": {"name": "pages"}, "sync_mode": "full_refresh", "destination_sync_mode": "overwrite"}]},
        ReadHandlers(
            on_record=lambda s, d, e: records.append((s, d, e)),
            on_state=lambda s: states.append(s),
            on_log=lambda lvl, msg: logs.append((lvl, msg)),
        ),
    )
    assert stats.exit_code == 0
    assert stats.trace_error is None
    assert records == [("pages", {"id": "a"}, 1), ("pages", {"id": "b"}, 2)]
    assert states == [{"type": "STREAM", "stream": {"stream_descriptor": {"name": "pages"}}}]
    assert logs == [("INFO", "Starting")]


def test_run_read_captures_trace_error():
    msgs = [
        {"type": "RECORD", "record": {"stream": "pages", "data": {"id": "a"}, "emitted_at": 1}},
        {"type": "TRACE", "trace": {"type": "ERROR", "error": {"message": "quota"}}},
    ]
    fake = FakeDockerClient([FakeContainer([_encode_msg(m) for m in msgs], exit_code=1)])
    stats = run_read(
        fake,
        "airbyte/source-x:latest",
        {},
        {"streams": []},
        ReadHandlers(on_record=lambda s, d, e: None, on_state=lambda s: None, on_log=lambda lvl, msg: None),
    )
    assert stats.trace_error is not None
    assert stats.trace_error.get("message") == "quota"


def test_parse_skips_non_json_lines():
    # Real sources occasionally print non-JSON banners to stdout before
    # the protocol kicks in; runner should silently skip them.
    msgs = [
        b"Booting source-foo v1.2",
        _encode_msg({"type": "RECORD", "record": {"stream": "s", "data": {"k": 1}, "emitted_at": 1}}),
    ]
    fake = FakeDockerClient([FakeContainer(msgs)])
    records: list[Any] = []
    run_read(
        fake,
        "airbyte/source-x:latest",
        {},
        {"streams": []},
        ReadHandlers(
            on_record=lambda s, d, e: records.append((s, d, e)),
            on_state=lambda s: None,
            on_log=lambda lvl, msg: None,
        ),
    )
    assert records == [("s", {"k": 1}, 1)]


def test_get_docker_client_unreachable(monkeypatch):
    # Simulate docker.from_env() raising — e.g. daemon not running.
    class _Exc(runner.DockerException):
        pass

    monkeypatch.setattr(runner.docker, "from_env", lambda: (_ for _ in ()).throw(_Exc("nope")))
    with pytest.raises(RunnerError, match="Docker Desktop"):
        runner.get_docker_client()
