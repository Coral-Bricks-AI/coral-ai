"""Tests for ``coralbricks.sandbox.tools.py`` (the user-facing shim).

The executor itself is tested in ``test_py_executor.py``; this file
focuses on the wrapper:

- env-var-driven configuration of the process-singleton executor
  (``CORAL_PY_LIBRARIES`` etc.) so what the runner sets at startup
  actually reaches the AST validator,
- the JSON-friendly response envelope (``ok`` / ``result`` /
  ``stdout`` / ``error``) which the wire shape callers expect,
- the ``PyValidationError`` propagation rule (validation failures
  bubble; runtime failures fold into ``ok=False``).
"""

from __future__ import annotations

import os

import pytest

from coralbricks.sandbox import tools
from coralbricks.sandbox.py_executor import PyValidationError


@pytest.fixture(autouse=True)
def _reset_executor_between_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets a fresh process-singleton executor.

    The env-driven lazy init is by-design sticky for the lifetime of
    the runner subprocess; tests need a clean slate so library
    allowlists don't bleed across cases.
    """
    monkeypatch.delenv(tools.PY_LIBRARIES_ENV_VAR, raising=False)
    monkeypatch.delenv(tools.PY_TIMEOUT_MS_ENV_VAR, raising=False)
    monkeypatch.delenv(tools.PY_MAX_OUTPUT_BYTES_ENV_VAR, raising=False)
    tools.reset_py_executor()
    yield
    tools.reset_py_executor()


def test_simple_snippet_returns_ok_envelope() -> None:
    out = tools.py("result = 1 + 2\n")
    assert out["ok"] is True
    assert out["result"] == 3
    assert out["stdout"] == ""
    assert out["truncated"] is False
    assert out["globals_added"] == []
    assert isinstance(out["took_ms"], int)
    assert "error" not in out


def test_runtime_error_returns_failure_envelope() -> None:
    out = tools.py("x = 1 / 0\n")
    assert out["ok"] is False
    assert out["result"] is None
    assert out["error"]["type"] == "ZeroDivisionError"
    assert "ZeroDivisionError" in out["error"]["traceback"]


def test_validation_error_propagates() -> None:
    with pytest.raises(PyValidationError):
        tools.py("eval('1+1')\n")


def test_env_libraries_allow_third_party_import() -> None:
    os.environ[tools.PY_LIBRARIES_ENV_VAR] = "json_helpers"
    tools.reset_py_executor()
    # Won't actually be importable, but the AST validator must let
    # it through; the failure must come from the runtime importer.
    out = tools.py("import json_helpers\n")
    assert out["ok"] is False
    assert out["error"]["type"] == "ModuleNotFoundError"


def test_env_libraries_unset_blocks_third_party_import() -> None:
    # No CORAL_PY_LIBRARIES => stdlib basics only. Validation
    # rejections raise (they're a programming error, not a
    # ok=False runtime failure) so callers see a clean traceback.
    with pytest.raises(PyValidationError, match="json_helpers"):
        tools.py("import json_helpers\n")


def test_inputs_are_threaded_into_executor() -> None:
    out = tools.py(
        "result = bars['n'] * 2\n",
        inputs={"bars": {"n": 21}},
    )
    assert out["ok"] is True
    assert out["result"] == 42


def test_env_timeout_applied() -> None:
    # 50ms timeout: ``time.sleep(2)`` will trip it.
    os.environ[tools.PY_TIMEOUT_MS_ENV_VAR] = "50"
    tools.reset_py_executor()
    out = tools.py("import time\ntime.sleep(2)\n")
    assert out["ok"] is False
    assert out["error"]["type"] == "PyTimeoutError"


def test_env_max_output_truncation() -> None:
    os.environ[tools.PY_MAX_OUTPUT_BYTES_ENV_VAR] = "32"
    tools.reset_py_executor()
    out = tools.py("print('x' * 1000)\nresult = 'done'\n")
    assert out["ok"] is True
    assert out["truncated"] is True
    assert out["result"] == "done"


def test_state_persists_across_shim_calls() -> None:
    # The shim returns a new envelope each time but the process-
    # singleton executor accumulates globals.
    tools.py("count = 0\n")
    tools.py("count += 5\n")
    out = tools.py("result = count\n")
    assert out["result"] == 5


def test_reset_py_executor_drops_state() -> None:
    tools.py("x = 99\n")
    tools.reset_py_executor()
    out = tools.py(
        "try:\n    result = x\nexcept NameError:\n    result = 'gone'\n"
    )
    assert out["result"] == "gone"


def test_garbage_env_timeout_falls_back_to_default() -> None:
    os.environ[tools.PY_TIMEOUT_MS_ENV_VAR] = "not-a-number"
    tools.reset_py_executor()
    # Default is 30s; the snippet should run fine.
    out = tools.py("result = 'ok'\n")
    assert out["result"] == "ok"


# ---------------------------------------------------------------------------
# bind_py_global / unbind_py_global -- the "tool result -> python var"
# affordance used by upstream pipeline tools (e.g. cb_ia's
# bm25_gdelt(bind_as=...)).
# ---------------------------------------------------------------------------


def test_bind_py_global_makes_value_visible_to_later_py_call() -> None:
    # The whole flow in one test: bind a payload, read it from a
    # later snippet without passing it through inputs=.
    payload = {"hits": [{"id": "a", "score": 1.0}, {"id": "b", "score": 0.5}]}
    tools.bind_py_global("scraped_articles", payload)
    out = tools.py("result = [h['id'] for h in scraped_articles['hits']]\n")
    assert out["ok"] is True
    assert out["result"] == ["a", "b"]


def test_bind_py_global_persists_across_multiple_py_calls() -> None:
    tools.bind_py_global("counter_seed", 100)
    tools.py("acc = counter_seed * 2\n")
    out = tools.py("result = acc + counter_seed\n")
    assert out["result"] == 300


def test_unbind_py_global_drops_value() -> None:
    tools.bind_py_global("temp", "hello")
    assert tools.unbind_py_global("temp") is True
    out = tools.py(
        "try:\n    result = temp\nexcept NameError:\n    result = 'gone'\n"
    )
    assert out["result"] == "gone"
    # Idempotent.
    assert tools.unbind_py_global("temp") is False


def test_bind_py_global_rejects_bad_name() -> None:
    from coralbricks.sandbox.py_executor import PyValidationError

    with pytest.raises(PyValidationError):
        tools.bind_py_global("not an identifier", 1)
    with pytest.raises(PyValidationError):
        tools.bind_py_global("__class__", 1)


def test_bound_value_survives_inputs_shadow() -> None:
    # If a snippet's inputs={...} shadows a bound name, the
    # original bound value must be restored after exec.
    tools.bind_py_global("hits", ["original"])

    inside = tools.py("result = hits\n", inputs={"hits": ["shadowed"]})
    assert inside["result"] == ["shadowed"]

    after = tools.py("result = hits\n")
    assert after["result"] == ["original"]
