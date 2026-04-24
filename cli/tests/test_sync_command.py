"""End-to-end tests for `coralbricks sync <source>`.

The real work — docker pull + container run — is mocked out. What we
care about here is the orchestration: did we call /runs, hand the runner
the right image/config, forward records through the S3 writer, and
report success/failure to /complete with the expected payload.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
import responses
from click.testing import CliRunner
from coralbricks.cli import config as cfg_mod
from coralbricks.cli.app import cli
from coralbricks.cli.commands import sync as sync_mod
from coralbricks.cli.runner import DiscoverResult, RunStats

BACKEND = "http://backend.test"


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg_mod, "config_dir", lambda: tmp_path / "coralbricks")
    monkeypatch.setattr(cfg_mod, "config_path", lambda: tmp_path / "coralbricks" / "config.json")
    monkeypatch.delenv(cfg_mod.ENV_API_KEY, raising=False)
    monkeypatch.delenv("CORALBRICKS_ALLOWED_BUCKETS", raising=False)
    monkeypatch.setenv(cfg_mod.ENV_SERVER_URL, BACKEND)
    cfg_mod.save(
        cfg_mod.Config(
            api_key="ak_test",
            server_url=BACKEND,
            user_id=1,
            email="test@coralbricks.ai",
        )
    )


@pytest.fixture
def mocked_responses():
    with responses.RequestsMock() as r:
        yield r


def _run_blob(**overrides: Any) -> dict[str, Any]:
    base = {
        "runId": 42,
        "dockerImage": "airbyte/source-notion:latest",
        "configJson": {"start_date": "2020-01-01", "credentials": {"access_token": "x"}},
        "s3Bucket": "coralbricks-connectors",
        "s3KeyPrefix": "users/1/sources/notion/conn-7/runs/42",
        "stsCredentials": {
            "accessKeyId": "AKIA",
            "secretAccessKey": "SECRET",
            "sessionToken": "TOKEN",
            "expiration": "2026-01-01T00:00:00Z",
        },
        "streamSelection": "all",
    }
    base.update(overrides)
    return base


def _stub_runner(monkeypatch, *, records: list[tuple[str, dict[str, Any], int]], trace_error=None):
    """Replace runner + s3 internals with in-memory fakes."""
    monkeypatch.setattr(sync_mod, "get_docker_client", lambda: MagicMock())
    monkeypatch.setattr(sync_mod, "pull_image", lambda *a, **kw: None)
    monkeypatch.setattr(
        sync_mod,
        "discover_streams",
        lambda *a, **kw: DiscoverResult(
            streams=[{"name": "pages", "supported_sync_modes": ["full_refresh"]}],
            raw_catalog={"streams": [{"name": "pages"}]},
        ),
    )

    def fake_run_read(_client, _image, _config, _catalog, handlers):
        for stream, data, emitted_at in records:
            handlers.on_record(stream, data, emitted_at)
        handlers.on_state({"type": "STREAM", "stream": {"stream_descriptor": {"name": "pages"}}})
        return RunStats(exit_code=0 if trace_error is None else 1, trace_error=trace_error)

    monkeypatch.setattr(sync_mod, "run_read", fake_run_read)

    # Mock boto3 so the S3 writer captures uploads in memory.
    uploaded: dict[str, bytes] = {}

    class _FakeS3:
        def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_kw: Any) -> dict[str, Any]:  # noqa: N803 — boto3 kwargs
            uploaded[f"{Bucket}/{Key}"] = Body
            return {"ETag": '"fake"'}

    import coralbricks.cli.s3 as s3_mod

    monkeypatch.setattr(s3_mod, "boto3", MagicMock(client=MagicMock(return_value=_FakeS3())))
    return uploaded


# ---------- happy path ----------


def test_sync_success_reports_complete(monkeypatch, mocked_responses):
    mocked_responses.add(
        responses.POST,
        f"{BACKEND}/cli/v1/runs",
        json={"success": True, "data": _run_blob()},
        status=201,
    )
    complete_req: dict[str, Any] = {}

    def _complete_cb(req):
        complete_req["body"] = json.loads(req.body)
        return (200, {}, json.dumps({"success": True, "data": {"runId": 42, "status": "success"}}))

    mocked_responses.add_callback(
        responses.POST, f"{BACKEND}/cli/v1/runs/42/complete", callback=_complete_cb
    )

    uploaded = _stub_runner(
        monkeypatch,
        records=[
            ("pages", {"id": "a"}, 1),
            ("pages", {"id": "b"}, 2),
        ],
    )

    result = CliRunner().invoke(cli, ["sync", "notion"])
    assert result.exit_code == 0, result.output
    assert "Synced" in result.output
    body = complete_req["body"]
    assert body["status"] == "success"
    assert body["recordsWritten"] == 2
    assert body["s3KeyPrefix"] == "users/1/sources/notion/conn-7/runs/42"
    assert "endCursor" in body

    expected_key = "coralbricks-connectors/users/1/sources/notion/conn-7/runs/42/pages/part-0000.jsonl.gz"
    assert expected_key in uploaded


def test_sync_trace_error_reports_failed(monkeypatch, mocked_responses):
    """Top-level trace error with zero records → hard failure."""
    mocked_responses.add(
        responses.POST,
        f"{BACKEND}/cli/v1/runs",
        json={"success": True, "data": _run_blob()},
        status=201,
    )
    complete_req: dict[str, Any] = {}

    def _complete_cb(req):
        complete_req["body"] = json.loads(req.body)
        return (200, {}, json.dumps({"success": True, "data": {"runId": 42, "status": "failed"}}))

    mocked_responses.add_callback(
        responses.POST, f"{BACKEND}/cli/v1/runs/42/complete", callback=_complete_cb
    )

    _stub_runner(
        monkeypatch,
        records=[],
        trace_error={"message": "401 Unauthorized"},
    )

    result = CliRunner().invoke(cli, ["sync", "notion"])
    assert result.exit_code != 0
    assert "401 Unauthorized" in result.output
    assert complete_req["body"]["status"] == "failed"
    assert "401 Unauthorized" in complete_req["body"]["errorMessage"]


def test_sync_trace_error_with_records_reports_success(monkeypatch, mocked_responses):
    """Per-stream failure but other streams uploaded data → success-with-warnings."""
    mocked_responses.add(
        responses.POST,
        f"{BACKEND}/cli/v1/runs",
        json={"success": True, "data": _run_blob()},
        status=201,
    )
    complete_req: dict[str, Any] = {}

    def _complete_cb(req):
        complete_req["body"] = json.loads(req.body)
        return (200, {}, json.dumps({"success": True, "data": {"runId": 42, "status": "success"}}))

    mocked_responses.add_callback(
        responses.POST, f"{BACKEND}/cli/v1/runs/42/complete", callback=_complete_cb
    )

    _stub_runner(
        monkeypatch,
        records=[("pages", {"id": "a"}, 1)],
        trace_error={"message": "1 stream failed"},
    )

    result = CliRunner().invoke(cli, ["sync", "notion"])
    assert result.exit_code == 0, result.output
    assert "Synced" in result.output
    assert complete_req["body"]["status"] == "success"
    assert complete_req["body"]["recordsWritten"] == 1


# ---------- defensive checks ----------


def test_sync_refuses_non_airbyte_image(monkeypatch, mocked_responses):
    mocked_responses.add(
        responses.POST,
        f"{BACKEND}/cli/v1/runs",
        json={"success": True, "data": _run_blob(dockerImage="malicious/source:latest")},
        status=201,
    )
    result = CliRunner().invoke(cli, ["sync", "notion"])
    assert result.exit_code != 0
    assert "non-Airbyte image" in result.output


def test_sync_refuses_unexpected_bucket(monkeypatch, mocked_responses):
    mocked_responses.add(
        responses.POST,
        f"{BACKEND}/cli/v1/runs",
        json={"success": True, "data": _run_blob(s3Bucket="some-other-bucket")},
        status=201,
    )
    result = CliRunner().invoke(cli, ["sync", "notion"])
    assert result.exit_code != 0
    assert "Unexpected destination bucket" in result.output


def test_sync_env_override_allows_dev_bucket(monkeypatch, mocked_responses):
    # Local-testing path: operator opts into an alternate bucket via env.
    monkeypatch.setenv("CORALBRICKS_ALLOWED_BUCKETS", "coralbricks-dev-logs")
    mocked_responses.add(
        responses.POST,
        f"{BACKEND}/cli/v1/runs",
        json={"success": True, "data": _run_blob(s3Bucket="coralbricks-dev-logs")},
        status=201,
    )
    mocked_responses.add(
        responses.POST,
        f"{BACKEND}/cli/v1/runs/42/complete",
        json={"success": True, "data": {"runId": 42, "status": "success"}},
        status=200,
    )
    _stub_runner(monkeypatch, records=[("pages", {"id": "a"}, 1)])

    result = CliRunner().invoke(cli, ["sync", "notion"])
    assert result.exit_code == 0, result.output
    assert "Unexpected destination bucket" not in result.output


def test_sync_refuses_non_user_prefix(monkeypatch, mocked_responses):
    mocked_responses.add(
        responses.POST,
        f"{BACKEND}/cli/v1/runs",
        json={"success": True, "data": _run_blob(s3KeyPrefix="global/bad")},
        status=201,
    )
    result = CliRunner().invoke(cli, ["sync", "notion"])
    assert result.exit_code != 0
    assert "Unexpected key prefix" in result.output


def test_sync_rejects_invalid_source_id(mocked_responses):
    # No backend calls should fire — local regex rejects it.
    result = CliRunner().invoke(cli, ["sync", "../admin"])
    assert result.exit_code != 0
    assert "Invalid source id" in result.output
