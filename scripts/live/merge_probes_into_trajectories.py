"""Stitch a vLLM-side probe sidecar into hermes-agent ShareGPT output.

Cycle-3 v2 (probe-at-rollout) needs every assistant turn in a Hermes
Agent trajectory to carry the rollout-time probes (``prompt_token_ids``,
``response_token_ids``, ``response_logprobs``,
``response_routed_experts``) that vLLM emitted at generation time.

Hermes-agent's batch_runner does not currently expose those signals.
The lowest-friction integration path is a side-channel JSONL emitted
by whatever component does intercept the chat-completions traffic (a
patched batch_runner, a local proxy, or a vLLM logging hook) — keyed
by ``(prompt_index, turn_idx)``::

    {"prompt_index": 0, "turn_idx": 0, "prompt_token_ids": [...],
     "response_token_ids": [...], "response_logprobs": [...]}
    {"prompt_index": 0, "turn_idx": 1, "prompt_token_ids": [...], ...}
    {"prompt_index": 1, "turn_idx": 0, ...}

This script:

  1. Reads the hermes-agent ShareGPT JSONL (one record per task,
     conversations under ``conversations``).
  2. Reads the probe sidecar JSONL and indexes it by
     ``(prompt_index, turn_idx)``.
  3. For each task, walks the ``from=gpt`` conv entries in order and
     injects matching probe fields as sibling keys on the entry.
  4. Calls ``adapt_hermes_sharegpt_trajectory`` (which already threads
     those sibling fields onto each ``AgentStep``) to produce the
     final probe-bearing ``AgentTrajectory`` JSON, one per task.

Probes are optional — turns without a matching sidecar entry simply
do not carry probes (the AgentStep validator already tolerates this
on assistant-role steps).

This script does not touch the spot or hermes-agent; it is purely a
stitcher and is testable offline against synthetic fixtures.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from rollout_market.observatory.agent_trajectory_lab import AgentTrajectory


def _load_runner() -> object:
    """Lazy-import hermes_agent_runner.py for adapt_hermes_sharegpt_trajectory."""
    spec_path = Path(__file__).resolve().parent / "hermes_agent_runner.py"
    spec = importlib.util.spec_from_file_location("hermes_agent_runner_for_merge", spec_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PROBE_KEYS = (
    "prompt_token_ids",
    "response_token_ids",
    "response_logprobs",
    "response_routed_experts",
)


def index_probes(probe_records: list[dict]) -> dict[tuple[int, int], dict]:
    """Group probe records by ``(prompt_index, turn_idx)``.

    Each value contains only the load-bearing probe fields — extra
    keys on the input record (timestamps, model id) are dropped.
    Raises ``ValueError`` if a record is missing the required keys
    OR if two records collide on the same ``(prompt_index, turn_idx)``
    pair (would indicate a buggy upstream capture).
    """
    out: dict[tuple[int, int], dict] = {}
    for i, rec in enumerate(probe_records):
        if not isinstance(rec, dict):
            raise ValueError(f"probe record {i} is not a dict: {rec!r}")
        if "prompt_index" not in rec or "turn_idx" not in rec:
            raise ValueError(f"probe record {i} missing prompt_index or turn_idx: {rec!r}")
        key = (int(rec["prompt_index"]), int(rec["turn_idx"]))
        if key in out:
            raise ValueError(f"duplicate probe record for prompt_index={key[0]}, turn_idx={key[1]}")
        out[key] = {k: rec[k] for k in _PROBE_KEYS if k in rec}
    return out


def inject_probes(
    sharegpt_record: dict,
    probes_by_turn: dict[int, dict],
) -> dict:
    """Return a copy of a ShareGPT record with probes stamped onto each ``from=gpt``.

    ``probes_by_turn`` maps assistant-turn index (0-based, gpt-only) to
    a probe dict (subset of ``_PROBE_KEYS``). Conv entries the probes
    dict does not mention are left unmodified.
    """
    convs_in = sharegpt_record.get("conversations") or []
    convs_out: list[dict] = []
    gpt_idx = 0
    for conv in convs_in:
        if not isinstance(conv, dict):
            convs_out.append(conv)
            continue
        if conv.get("from") == "gpt":
            probes = probes_by_turn.get(gpt_idx)
            if probes:
                merged = {**conv, **probes}
                convs_out.append(merged)
            else:
                convs_out.append(conv)
            gpt_idx += 1
        else:
            convs_out.append(conv)
    return {**sharegpt_record, "conversations": convs_out}


def merge_to_trajectories(
    *,
    sharegpt_records: list[dict],
    probe_records: list[dict],
    tasks: list[dict],
    model_id: str,
    engine_label: str,
    engine_fingerprint: str,
    engine_version: str = "0.20.1",
) -> list[AgentTrajectory]:
    """Stitch probes onto ShareGPT records and adapt each to an AgentTrajectory.

    Pure function. ``tasks`` is the in-order list of
    ``agent_tasks_hermes.json`` entries — one per ``prompt_index``
    expected on the sharegpt side; sharegpt records that disagree on
    ``prompt_index`` use positional matching as a fallback. Raises
    ``ValueError`` on length mismatch between ``sharegpt_records`` and
    ``tasks``.
    """
    if len(sharegpt_records) != len(tasks):
        raise ValueError(
            f"sharegpt has {len(sharegpt_records)} records but tasks has "
            f"{len(tasks)} — these must align 1:1 on prompt_index"
        )
    runner = _load_runner()
    probe_index = index_probes(probe_records)
    out: list[AgentTrajectory] = []
    for i, (sg, task) in enumerate(zip(sharegpt_records, tasks)):
        # Some upstream pipelines stamp prompt_index on the record;
        # prefer that, else fall back to positional index.
        prompt_index = int(sg.get("prompt_index", i))
        probes_by_turn = {
            turn_idx: probes
            for (pidx, turn_idx), probes in probe_index.items()
            if pidx == prompt_index
        }
        stitched = inject_probes(sg, probes_by_turn)
        traj = runner.adapt_hermes_sharegpt_trajectory(  # type: ignore[attr-defined]
            stitched,
            task=task,
            model_id=model_id,
            engine_label=engine_label,
            engine_fingerprint=engine_fingerprint,
            engine_version=engine_version,
            available_tools=task.get("available_tools", []),
        )
        out.append(traj)
    return out


def _read_jsonl(path: Path) -> list[dict]:
    """Read a ``.jsonl`` file; tolerate trailing whitespace + blank lines."""
    out: list[dict] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {i + 1} of {path} is not valid JSON: {exc}") from exc
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a vLLM probe sidecar JSONL into a hermes-agent ShareGPT "
            "JSONL and emit one probe-bearing AgentTrajectory JSON per task."
        )
    )
    parser.add_argument("--sharegpt", required=True, help="hermes-agent batch_runner output JSONL.")
    parser.add_argument(
        "--probes",
        required=True,
        help="Probe sidecar JSONL (keyed by prompt_index + turn_idx).",
    )
    parser.add_argument(
        "--tasks",
        required=True,
        help="Path to agent_tasks_hermes.json (the task list driving the run).",
    )
    parser.add_argument("--model-id", required=True, help="HF model id.")
    parser.add_argument(
        "--engine-label",
        required=True,
        help="Rollout engine label (e.g. hermes-qwen3-32b-bf16).",
    )
    parser.add_argument(
        "--engine-fingerprint",
        required=True,
        help="Rollout engine fingerprint (sha256:...).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Where to write <task_id>.json AgentTrajectory files.",
    )
    args = parser.parse_args(argv)

    sharegpt_path = Path(args.sharegpt)
    probes_path = Path(args.probes)
    tasks_path = Path(args.tasks)
    out_dir = Path(args.out_dir)
    for p in (sharegpt_path, probes_path, tasks_path):
        if not p.exists():
            print(f"[fatal] not found: {p}", file=sys.stderr)
            return 2

    sharegpt_records = _read_jsonl(sharegpt_path)
    probe_records = _read_jsonl(probes_path)
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))

    trajectories = merge_to_trajectories(
        sharegpt_records=sharegpt_records,
        probe_records=probe_records,
        tasks=tasks,
        model_id=args.model_id,
        engine_label=args.engine_label,
        engine_fingerprint=args.engine_fingerprint,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for traj in trajectories:
        out_path = out_dir / f"{traj.task_id}.json"
        out_path.write_text(traj.model_dump_json(indent=2), encoding="utf-8")
        print(f"[hermes-merge] {traj.task_id} -> {out_path}", flush=True)

    n_with_probes = sum(
        any(
            s.role == "assistant"
            and (s.prompt_token_ids is not None or s.response_logprobs is not None)
            for s in t.steps
        )
        for t in trajectories
    )
    print(
        f"[hermes-merge] {len(trajectories)} trajectories ({n_with_probes} with "
        "probes on >=1 turn)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
