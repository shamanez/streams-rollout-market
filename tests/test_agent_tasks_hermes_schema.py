"""Schema + coverage tests for scripts/live/agent_tasks_hermes.json.

The acceptance gate in STEER.md is operational: each task in the
hermes task suite drives one trajectory per (engine, precision)
combination, so these schema checks are load-bearing — a malformed
task file will silently produce empty trajectories instead of failing
the run.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TASKS_PATH = _REPO_ROOT / "scripts" / "live" / "agent_tasks_hermes.json"


def _load_agent_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "agent_runner_module",
        _REPO_ROOT / "scripts" / "live" / "agent_runner.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_tasks() -> list[dict]:
    return json.loads(_TASKS_PATH.read_text(encoding="utf-8"))


# --- schema ------------------------------------------------------------------


def test_tasks_file_exists() -> None:
    assert _TASKS_PATH.exists(), f"missing {_TASKS_PATH}"


def test_tasks_count_is_exactly_12() -> None:
    tasks = _load_tasks()
    assert len(tasks) == 12, f"expected 12 hermes tasks, got {len(tasks)}"


def test_task_ids_unique() -> None:
    ids = [t["task_id"] for t in _load_tasks()]
    assert len(ids) == len(set(ids)), f"duplicate task_ids: {ids}"


def test_each_task_has_required_fields() -> None:
    for t in _load_tasks():
        assert "task_id" in t and isinstance(t["task_id"], str) and t["task_id"]
        assert "task_text" in t and isinstance(t["task_text"], str) and t["task_text"]
        # expected_tools is optional; when present must be a list of strings.
        if "expected_tools" in t:
            assert isinstance(t["expected_tools"], list)
            for tool in t["expected_tools"]:
                assert isinstance(tool, str) and tool


# --- coverage against TOOL_SCHEMAS -------------------------------------------


def test_expected_tools_subset_of_tool_schemas() -> None:
    mod = _load_agent_runner()
    schema_names = {t["function"]["name"] for t in mod.TOOL_SCHEMAS}
    for t in _load_tasks():
        for tool in t.get("expected_tools", []):
            assert tool in schema_names, (
                f"task {t['task_id']!r} references unknown tool {tool!r}; "
                f"known tools: {sorted(schema_names)}"
            )


def test_at_least_four_tasks_use_multiple_distinct_tools() -> None:
    multi = [
        t for t in _load_tasks() if len(set(t.get("expected_tools", []))) >= 2
    ]
    assert len(multi) >= 4, (
        f"need >=4 multi-tool tasks (>=2 distinct expected_tools), got {len(multi)}: "
        f"{[t['task_id'] for t in multi]}"
    )


def test_at_least_one_task_has_empty_or_absent_expected_tools() -> None:
    """The no-op-trivia case must be discoverable from the schema alone."""
    no_tool_tasks = [
        t for t in _load_tasks()
        if "expected_tools" not in t or not t["expected_tools"]
    ]
    assert no_tool_tasks, (
        "need >=1 task with empty or absent expected_tools (the no-op-trivia case)"
    )


# --- _SIM_FILES new entries --------------------------------------------------


def test_sim_files_contains_fib_and_clip() -> None:
    """Tasks 5 and 9 reference /tmp/fib.py and /tmp/clip.py — both must be
    canned in agent_runner.py::_SIM_FILES so the simulated read_file tool
    returns deterministic content."""
    mod = _load_agent_runner()
    assert "/tmp/fib.py" in mod._SIM_FILES, "_SIM_FILES missing /tmp/fib.py"
    assert "/tmp/clip.py" in mod._SIM_FILES, "_SIM_FILES missing /tmp/clip.py"
    # The fib fixture must contain the off-by-one bug the task asks about.
    fib_src = mod._SIM_FILES["/tmp/fib.py"]
    assert "def fib" in fib_src and "return 1" in fib_src
    clip_src = mod._SIM_FILES["/tmp/clip.py"]
    assert "def clip" in clip_src and "return lo" in clip_src and "return hi" in clip_src


def test_sim_files_factorial_still_present() -> None:
    """Non-regression: don't accidentally drop the original factorial fixture."""
    mod = _load_agent_runner()
    assert "/tmp/factorial.py" in mod._SIM_FILES


# --- specific task contracts (light) -----------------------------------------


def test_no_op_trivia_task_id_is_present() -> None:
    """The plan calls out a 'no-op-trivia' task that exercises whether the
    agent over-uses tools. Its task_id is part of the contract."""
    ids = {t["task_id"] for t in _load_tasks()}
    assert "no-op-trivia" in ids


def test_four_tool_sequence_task_lists_all_four_tools() -> None:
    """The long-horizon task is the principal divergence-generator; it
    must list all four tools so first_divergence_step has a meaningful range."""
    tasks = {t["task_id"]: t for t in _load_tasks()}
    assert "four-tool-sequence" in tasks
    expected = set(tasks["four-tool-sequence"].get("expected_tools", []))
    assert expected == {"web_search", "read_file", "python_eval", "calculator"}, (
        f"four-tool-sequence expected_tools must list all four tools, got {expected}"
    )
