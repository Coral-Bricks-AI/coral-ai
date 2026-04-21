"""Tests for ``coralbricks.sandbox.py_executor``.

The executor sits squarely on the trust boundary between the model's
generated code and the runner subprocess. We exercise three layers:

1. **AST validation.** Every banned shape (forbidden imports,
   ``eval`` / ``exec`` / ``compile`` calls, dunder attribute access,
   relative imports, async constructs) must be rejected at parse
   time, before the snippet ever gets ``compile()``-d let alone
   executed. We test each rejection class with at least one
   positive ("this exact pattern is rejected") and one negative
   ("this almost-bad-but-actually-fine pattern is accepted") case
   so the regex-style allowlists don't drift into being either too
   loose or too strict.

2. **Stateful execution.** Globals defined by snippet N are visible
   to snippet N+1; the ``result`` variable is read out and reset
   between calls; ``inputs`` are bound transiently (don't leak into
   ``globals_added`` or persist across calls).

3. **Resource enforcement.** Timeout via ``SIGALRM`` actually fires
   (we test with a tight ``time.sleep`` rather than a busy loop so
   the test stays fast). Stdout truncation respects ``max_output_bytes``
   on a UTF-8 byte boundary.

The shim wrapper (``coralbricks.sandbox.tools.py``) gets its own
test file -- this one stays focused on the executor.
"""

from __future__ import annotations

import pytest

from coralbricks.sandbox.py_executor import (
    PyExecutionError,
    PyExecutor,
    PyResult,
    PyTimeoutError,
    PyValidationError,
    validate_code,
)


# ---------------------------------------------------------------------------
# AST validator
# ---------------------------------------------------------------------------


class TestValidateCode:
    def test_empty_snippet_is_valid(self) -> None:
        validate_code("", allowed_imports=())

    def test_stdlib_imports_always_allowed(self) -> None:
        # math / json / re / datetime are in the default stdlib
        # allowlist; the manifest doesn't need to opt in.
        validate_code(
            "import math\nimport json\nfrom re import compile as _c\n",
            allowed_imports=(),
        )

    def test_third_party_import_requires_allowlist(self) -> None:
        with pytest.raises(PyValidationError, match="pandas"):
            validate_code(
                "import pandas as pd\n",
                allowed_imports=(),
            )

    def test_third_party_import_when_allowlisted(self) -> None:
        validate_code(
            "import pandas as pd\n",
            allowed_imports=("pandas",),
        )

    def test_dotted_import_checks_top_level_name(self) -> None:
        # `from pandas.io.sql import read_sql` should pass when
        # only the top-level "pandas" is allowed (matches how the
        # AST validator says it works).
        validate_code(
            "from pandas.io.sql import read_sql\n",
            allowed_imports=("pandas",),
        )

    def test_relative_import_always_rejected(self) -> None:
        with pytest.raises(PyValidationError, match="relative imports"):
            validate_code(
                "from . import helpers\n",
                allowed_imports=("helpers",),
            )

    def test_os_import_rejected_by_default(self) -> None:
        with pytest.raises(PyValidationError, match="not in library allowlist"):
            validate_code("import os\n", allowed_imports=())

    def test_subprocess_import_rejected_by_default(self) -> None:
        with pytest.raises(PyValidationError, match="not in library allowlist"):
            validate_code("import subprocess\n", allowed_imports=())

    def test_eval_call_rejected(self) -> None:
        with pytest.raises(PyValidationError, match="eval"):
            validate_code("x = eval('1+1')\n", allowed_imports=())

    def test_exec_call_rejected(self) -> None:
        with pytest.raises(PyValidationError, match="exec"):
            validate_code("exec('print(1)')\n", allowed_imports=())

    def test_compile_call_rejected(self) -> None:
        with pytest.raises(PyValidationError, match="compile"):
            validate_code(
                "co = compile('1+1', '<x>', 'eval')\n",
                allowed_imports=(),
            )

    def test_dunder_import_call_rejected(self) -> None:
        with pytest.raises(PyValidationError, match="__import__"):
            validate_code(
                "m = __import__('os')\n",
                allowed_imports=(),
            )

    def test_globals_call_rejected(self) -> None:
        with pytest.raises(PyValidationError, match="globals"):
            validate_code("g = globals()\n", allowed_imports=())

    def test_class_dunder_attr_rejected(self) -> None:
        with pytest.raises(PyValidationError, match="__class__"):
            validate_code(
                "x = (1).__class__\n",
                allowed_imports=(),
            )

    def test_subclasses_chain_rejected(self) -> None:
        with pytest.raises(
            PyValidationError, match="(__subclasses__|__class__|__bases__)"
        ):
            validate_code(
                "x = ().__class__.__bases__[0].__subclasses__()\n",
                allowed_imports=(),
            )

    def test_builtins_attr_rejected(self) -> None:
        with pytest.raises(PyValidationError, match="__builtins__"):
            validate_code(
                "b = (1).__builtins__\n",
                allowed_imports=(),
            )

    def test_async_def_rejected(self) -> None:
        with pytest.raises(PyValidationError, match="async"):
            validate_code(
                "async def f():\n    return 1\n",
                allowed_imports=(),
            )

    def test_await_rejected(self) -> None:
        with pytest.raises(PyValidationError, match="async"):
            # await sits inside an async def, which itself is rejected
            # first; either error is acceptable.
            validate_code(
                "async def f():\n    await x\n",
                allowed_imports=(),
            )

    def test_syntax_error_surfaces_as_validation_error(self) -> None:
        with pytest.raises(PyValidationError, match="syntax error"):
            validate_code("def f(\n", allowed_imports=())

    def test_non_string_code_rejected(self) -> None:
        with pytest.raises(PyValidationError, match="must be a string"):
            validate_code(b"x = 1", allowed_imports=())  # type: ignore[arg-type]

    def test_dict_attribute_access_rejected(self) -> None:
        # Even harmless-looking ``obj.__dict__`` can be used to walk
        # into closure cells; ban it.
        with pytest.raises(PyValidationError, match="__dict__"):
            validate_code(
                "class C: pass\nx = C.__dict__\n",
                allowed_imports=(),
            )

    def test_attr_lookalike_passes(self) -> None:
        # ``something.dict_thing`` is fine -- only the literal
        # ``__dict__`` is banned.
        validate_code(
            "class C:\n    dict_thing = 1\nx = C.dict_thing\n",
            allowed_imports=(),
        )


# ---------------------------------------------------------------------------
# Executor: stateful execution
# ---------------------------------------------------------------------------


class TestPyExecutorState:
    def test_simple_arithmetic_returns_via_result(self) -> None:
        ex = PyExecutor()
        out = ex.exec("result = 1 + 2 + 3\n")
        assert isinstance(out, PyResult)
        assert out.result == 6
        assert out.stdout == ""
        assert out.truncated is False
        assert out.took_ms >= 0
        assert "result" not in out.globals_added

    def test_no_result_returns_none(self) -> None:
        ex = PyExecutor()
        out = ex.exec("x = 1\n")
        assert out.result is None
        assert out.globals_added == ["x"]

    def test_globals_persist_across_calls(self) -> None:
        ex = PyExecutor()
        ex.exec("counter = 0\n")
        ex.exec("counter += 5\n")
        out = ex.exec("result = counter\n")
        assert out.result == 5

    def test_function_def_persists_across_calls(self) -> None:
        ex = PyExecutor()
        ex.exec("def add(a, b):\n    return a + b\n")
        out = ex.exec("result = add(2, 3)\n")
        assert out.result == 5

    def test_inputs_bind_into_globals(self) -> None:
        ex = PyExecutor()
        out = ex.exec(
            "result = bars['n'] * 2\n",
            inputs={"bars": {"n": 21}},
        )
        assert out.result == 42

    def test_inputs_do_not_persist_across_calls(self) -> None:
        ex = PyExecutor()
        ex.exec("y = 1\n", inputs={"transient": 99})
        out = ex.exec(
            "try:\n    result = transient\nexcept NameError:\n    "
            "result = 'gone'\n"
        )
        assert out.result == "gone"

    def test_inputs_do_not_appear_as_globals_added(self) -> None:
        ex = PyExecutor()
        out = ex.exec(
            "z = bars['n']\n",
            inputs={"bars": {"n": 1}},
        )
        # Only `z` is added -- `bars` was caller-provided, not
        # snippet-provided.
        assert "z" in out.globals_added
        assert "bars" not in out.globals_added

    def test_inputs_with_invalid_key_rejected(self) -> None:
        ex = PyExecutor()
        with pytest.raises(PyValidationError, match="identifier"):
            ex.exec("pass\n", inputs={"not a name": 1})

    def test_inputs_with_dunder_key_rejected(self) -> None:
        ex = PyExecutor()
        with pytest.raises(PyValidationError, match="dunder"):
            ex.exec("pass\n", inputs={"__class__": 1})

    def test_globals_added_excludes_dunder(self) -> None:
        # __annotations__ etc. show up in the globals dict after
        # ``exec`` even on tiny snippets; the executor filters them.
        ex = PyExecutor()
        out = ex.exec("x = 1\n")
        assert all(not g.startswith("__") for g in out.globals_added)

    def test_globals_snapshot_is_defensive_copy(self) -> None:
        ex = PyExecutor()
        ex.exec("x = 1\n")
        snap = ex.globals_snapshot
        snap["x"] = 999
        out = ex.exec("result = x\n")
        assert out.result == 1


# ---------------------------------------------------------------------------
# Executor: stdout / IO
# ---------------------------------------------------------------------------


class TestPyExecutorStdout:
    def test_print_captured(self) -> None:
        ex = PyExecutor()
        out = ex.exec("print('hello')\nresult = 1\n")
        assert "hello" in out.stdout
        assert out.result == 1
        assert out.truncated is False

    def test_stdout_truncation_fires_on_byte_boundary(self) -> None:
        ex = PyExecutor(max_output_bytes=64)
        out = ex.exec("print('x' * 1000)\n")
        assert out.truncated is True
        # Truncation marker is appended.
        assert "truncated" in out.stdout
        # The kept payload + marker can be a bit over the cap, but
        # the kept payload itself is at most max_output_bytes.
        kept_until_marker = out.stdout.split("\n... [truncated", 1)[0]
        assert len(kept_until_marker.encode("utf-8")) <= 64

    def test_stdout_just_under_cap_not_truncated(self) -> None:
        ex = PyExecutor(max_output_bytes=64)
        out = ex.exec("print('x' * 10)\n")
        assert out.truncated is False
        assert "truncated" not in out.stdout


# ---------------------------------------------------------------------------
# Executor: errors + timeout
# ---------------------------------------------------------------------------


class TestPyExecutorErrors:
    def test_runtime_error_wrapped(self) -> None:
        ex = PyExecutor()
        with pytest.raises(PyExecutionError) as ei:
            ex.exec("x = 1 / 0\n")
        assert ei.value.original_type == "ZeroDivisionError"
        assert "ZeroDivisionError" in ei.value.traceback_str

    def test_validation_error_propagates_not_wrapped(self) -> None:
        ex = PyExecutor()
        with pytest.raises(PyValidationError):
            ex.exec("eval('1+1')\n")

    def test_module_not_found_at_runtime_wrapped(self) -> None:
        # "calendar" is in the stdlib allowlist but if a snippet
        # tries to import a stdlib module that doesn't exist on this
        # interpreter (artificial -- use a typo), the import error
        # comes from runtime, not the validator.
        ex = PyExecutor(allowed_imports=("definitely_not_a_real_pkg",))
        with pytest.raises(PyExecutionError) as ei:
            ex.exec("import definitely_not_a_real_pkg\n")
        assert ei.value.original_type == "ModuleNotFoundError"

    def test_timeout_fires(self) -> None:
        # SIGALRM has 1s minimum granularity via setitimer in some
        # very old environments; we use 100ms here -- POSIX setitimer
        # supports microsecond granularity.
        ex = PyExecutor(timeout_ms=100)
        with pytest.raises(PyTimeoutError) as ei:
            ex.exec("import time\ntime.sleep(2)\n")
        assert ei.value.timeout_ms == 100

    def test_timeout_does_not_fire_for_quick_code(self) -> None:
        ex = PyExecutor(timeout_ms=1000)
        out = ex.exec("result = 42\n")
        assert out.result == 42

    def test_executor_recovers_from_timeout(self) -> None:
        # After a timeout, the executor's globals_dict should still
        # be usable for the next call (state from before the bad
        # snippet survives; the alarm handler tears down cleanly).
        ex = PyExecutor(timeout_ms=100)
        ex.exec("seed = 7\n")
        with pytest.raises(PyTimeoutError):
            ex.exec("import time\ntime.sleep(2)\n")
        out = ex.exec("result = seed * 6\n")
        assert out.result == 42


# ---------------------------------------------------------------------------
# Executor: configuration
# ---------------------------------------------------------------------------


class TestPyExecutorConfig:
    def test_zero_timeout_clamped_to_minimum(self) -> None:
        ex = PyExecutor(timeout_ms=0)
        assert ex.timeout_ms >= 1

    def test_negative_max_output_clamped(self) -> None:
        ex = PyExecutor(max_output_bytes=-100)
        assert ex.max_output_bytes >= 1

    def test_allowed_imports_property_is_frozenset(self) -> None:
        ex = PyExecutor(allowed_imports=("pandas", "numpy", "pandas"))
        assert ex.allowed_imports == frozenset({"pandas", "numpy"})


# ---------------------------------------------------------------------------
# Executor: persistent bindings (bind_global / unbind_global)
# ---------------------------------------------------------------------------


class TestPyExecutorBindGlobal:
    """Tests for the ``bind_global`` / ``unbind_global`` affordance.

    The interesting properties: bound names PERSIST across exec()
    calls (unlike ``inputs=`` which are dropped), they are PER-THREAD
    (so parallel cb_ia specialists can't clobber each other), and a
    transient ``inputs=`` shadow correctly RESTORES the bound value
    after the snippet returns.
    """

    def test_bound_name_visible_in_snippet(self) -> None:
        ex = PyExecutor()
        ex.bind_global("hits", [{"id": "a"}, {"id": "b"}])
        out = ex.exec("result = [h['id'] for h in hits]\n")
        assert out.result == ["a", "b"]

    def test_bound_name_persists_across_exec_calls(self) -> None:
        # The whole point of bind_global vs inputs=: persistence.
        ex = PyExecutor()
        ex.bind_global("scraped_articles", {"hits": [1, 2, 3]})
        ex.exec("count_first = len(scraped_articles['hits'])\n")
        out = ex.exec("result = count_first * 10\n")
        assert out.result == 30

    def test_inputs_does_not_persist_for_comparison(self) -> None:
        # Sanity check: inputs= is the OLD ephemeral pattern. Bound
        # globals are the new persistent pattern.
        ex = PyExecutor()
        ex.exec("result = x", inputs={"x": 7})
        out = ex.exec("try:\n    result = x\nexcept NameError:\n    result = 'gone'\n")
        assert out.result == "gone"

    def test_bound_name_does_not_appear_in_globals_added(self) -> None:
        # bind_global is a "caller-provided" name -- the snippet did
        # not create it -- so it must not show up in globals_added.
        ex = PyExecutor()
        ex.bind_global("preset", 42)
        out = ex.exec("derived = preset + 1\n")
        assert "derived" in out.globals_added
        assert "preset" not in out.globals_added

    def test_bound_name_can_be_unbound(self) -> None:
        ex = PyExecutor()
        ex.bind_global("hits", [1, 2, 3])
        assert ex.unbind_global("hits") is True
        out = ex.exec(
            "try:\n    result = hits\nexcept NameError:\n    result = 'gone'\n"
        )
        assert out.result == "gone"
        # Idempotent: second unbind reports False.
        assert ex.unbind_global("hits") is False

    def test_bind_global_rejects_non_identifier(self) -> None:
        ex = PyExecutor()
        with pytest.raises(PyValidationError):
            ex.bind_global("not an identifier", 1)

    def test_bind_global_rejects_dunder_name(self) -> None:
        ex = PyExecutor()
        with pytest.raises(PyValidationError):
            ex.bind_global("__class__", 1)

    def test_inputs_temporarily_shadows_bound_global(self) -> None:
        # An inputs= entry with the same name as a previously bound
        # global is a TRANSIENT shadow: the snippet sees the input
        # value, but after exec returns the original bound value is
        # RESTORED (the input was meant to be ephemeral, not destructive).
        ex = PyExecutor()
        ex.bind_global("hits", ["original"])
        out_inside = ex.exec(
            "result = hits\n", inputs={"hits": ["shadowed"]},
        )
        assert out_inside.result == ["shadowed"]

        out_after = ex.exec("result = hits\n")
        assert out_after.result == ["original"]

    def test_bindings_are_per_thread(self) -> None:
        # The cb_ia swarm runs each specialist on its own thread.
        # If bindings leaked across threads, vc_analyst binding
        # `bm25_hits` would clobber risk_analyst's `bm25_hits`.
        # This is the central isolation guarantee.
        import threading

        ex = PyExecutor()

        results: dict[str, Any] = {}

        def worker(name: str, value: Any) -> None:
            ex.bind_global("hits", value)
            # Read it back from the same thread.
            r = ex.exec("result = hits\n")
            results[name] = r.result

        t1 = threading.Thread(target=worker, args=("vc", "vc-bound"))
        t2 = threading.Thread(target=worker, args=("risk", "risk-bound"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["vc"] == "vc-bound"
        assert results["risk"] == "risk-bound"

        # The main thread never bound `hits`; it must see NameError.
        out = ex.exec(
            "try:\n    result = hits\nexcept NameError:\n    result = 'gone'\n"
        )
        assert out.result == "gone"

    def test_globals_snapshot_is_per_thread(self) -> None:
        import threading

        ex = PyExecutor()
        ex.bind_global("main_only", 1)

        snapshot_from_other: dict[str, Any] = {}

        def worker() -> None:
            snapshot_from_other.update(ex.globals_snapshot)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        # Main thread sees its binding...
        assert "main_only" in ex.globals_snapshot
        # ...the worker thread does not.
        assert "main_only" not in snapshot_from_other
