"""End-to-end Hermes dense pipeline integration test.

Wires together every component shipped during cycle-3 v2 iter 2:

  1. ``logprob_capture_proxy.extract_probes`` consumes a synthetic
     vLLM-shape chat-completions response and emits a probe sidecar
     record.
  2. ``merge_probes_into_trajectories.merge_to_trajectories`` joins a
     synthetic hermes-agent ShareGPT record with the sidecar and
     produces a probe-bearing ``AgentTrajectory``.
  3. ``build_hermes_dense_input.build_rollouts_and_index`` flattens
     the trajectory into the ``run_fsdp_reference.py`` rollouts
     payload plus the pairing index.
  4. A fake trainer emits ``trainer_logprobs`` matching the index's
     length contract.
  5. ``pair_hermes_dense_reports.pair_reports`` joins index +
     trainer outputs into a ``DenseMismatchReport``.

The test asserts the resulting report carries:
  * ``num_policy_tokens`` equal to the captured response_token_ids
    length;
  * ``rollout_engine.name`` from the trajectory's engine fingerprint;
  * ``trainer_engine.name`` from the trainer side;
  * sensible ESS (1.0 for matched logprobs).

This is a true integration test — no boundaries are mocked except
the network IO (vLLM upstream + trainer forward pass). The pure-
function seams are exactly the ones the live pipeline uses, so a
passing test rules out internal contract drift across the components.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_script(name: str) -> ModuleType:
    spec_path = Path(__file__).resolve().parents[1] / "scripts" / "live" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"e2e_{name}", spec_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def proxy_mod() -> ModuleType:
    return _load_script("logprob_capture_proxy")


@pytest.fixture(scope="module")
def merger_mod() -> ModuleType:
    return _load_script("merge_probes_into_trajectories")


@pytest.fixture(scope="module")
def builder_mod() -> ModuleType:
    return _load_script("build_hermes_dense_input")


@pytest.fixture(scope="module")
def pair_mod() -> ModuleType:
    return _load_script("pair_hermes_dense_reports")


@pytest.fixture(scope="module")
def filler_mod() -> ModuleType:
    return _load_script("fill_prompt_token_ids")


class _FakeTokenizer:
    """Deterministic chat-template stub mirroring the one in
    test_hermes_token_extraction.py — keeps the e2e test offline."""

    ROLE_OPEN: dict[str, int] = {
        "system": 90001,
        "user": 90002,
        "assistant": 90003,
        "tool": 90004,
    }
    ROLE_CLOSE: int = 90099
    GEN_PROMPT_MARKER: int = 90100

    def apply_chat_template(
        self,
        conversation: list[dict],
        *,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
    ) -> list[int]:
        assert tokenize is True
        ids: list[int] = []
        for msg in conversation:
            role = msg["role"]
            ids.append(self.ROLE_OPEN[role])
            if role == "assistant":
                ids.append(self.GEN_PROMPT_MARKER)
            ids.extend(ord(c) for c in msg["content"])
            ids.append(self.ROLE_CLOSE)
        if add_generation_prompt:
            ids.append(self.ROLE_OPEN["assistant"])
            ids.append(self.GEN_PROMPT_MARKER)
        return ids


def _synth_vllm_response(tokens: list[tuple[str, int, float]]) -> bytes:
    content = [{"token": tok, "logprob": lp, "token_id": tid} for tok, tid, lp in tokens]
    return json.dumps(
        {
            "id": "chatcmpl-e2e",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "".join(t[0] for t in tokens),
                    },
                    "logprobs": {"content": content},
                    "finish_reason": "stop",
                }
            ],
        }
    ).encode("utf-8")


def _synth_sharegpt_record(prompt_index: int = 0) -> dict:
    return {
        "prompt_index": prompt_index,
        "conversations": [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "What is 2+2?"},
            {"from": "gpt", "value": "<think>\n</think>\nThe answer is 4."},
        ],
        "completed": True,
        "partial": False,
    }


def _task() -> dict:
    return {
        "task_id": "math-q1",
        "task_text": "What is 2+2?",
        "available_tools": [],
    }


def test_full_dense_pipeline_e2e(
    proxy_mod: ModuleType,
    merger_mod: ModuleType,
    builder_mod: ModuleType,
    pair_mod: ModuleType,
    filler_mod: ModuleType,
) -> None:
    # --- step 1: proxy.extract_probes(synthetic vLLM response) -----------
    tokens = [
        ("The", 791, -0.1),
        (" answer", 4320, -0.2),
        (" is", 374, -0.05),
        (" 4", 19, -0.01),
        (".", 13, -0.001),
    ]
    sidecar_record = proxy_mod.extract_probes(
        b"{}",
        _synth_vllm_response(tokens),
        prompt_index=0,
        turn_idx=0,
    )
    assert sidecar_record is not None
    assert sidecar_record["response_token_ids"] == [t[1] for t in tokens]

    # --- step 2: merger.merge_to_trajectories(sharegpt + sidecar) --------
    trajectories = merger_mod.merge_to_trajectories(
        sharegpt_records=[_synth_sharegpt_record(0)],
        probe_records=[sidecar_record],
        tasks=[_task()],
        model_id="Qwen/Qwen3-32B",
        engine_label="hermes-qwen3-32b-bf16",
        engine_fingerprint="sha256:hermes-qwen3-32b-bf16-tp4-l40s",
    )
    assert len(trajectories) == 1
    traj = trajectories[0]
    assistants = [s for s in traj.steps if s.role == "assistant"]
    assert len(assistants) == 1
    step = assistants[0]
    assert step.response_token_ids == [t[1] for t in tokens]
    assert step.response_logprobs == pytest.approx([t[2] for t in tokens])

    # --- step 3a: filler.fill_trajectory re-derives prompt_token_ids ---
    # This is the load-bearing step that replaces the previous synthetic
    # stamp: the proxy can't capture prompt_token_ids from
    # chat-completions, so we let the filler do the real chat-template
    # encoding from the trajectory's accumulated history.
    traj_with_prompt, n_filled = filler_mod.fill_trajectory(traj, _FakeTokenizer())
    assert n_filled == 1
    filled_step = traj_with_prompt.assistant_steps()[0]
    assert filled_step.prompt_token_ids is not None
    # Captured response IDs survived the fill step untouched.
    assert filled_step.response_token_ids == [t[1] for t in tokens]

    # --- step 3b: builder.build_rollouts_and_index ---------------------
    rollouts, index = builder_mod.build_rollouts_and_index(
        [traj_with_prompt], precision_class="bf16"
    )
    assert len(rollouts) == 1
    assert rollouts[0]["response_token_ids"] == [t[1] for t in tokens]
    assert index["entries"][0]["num_tokens"] == len(tokens)

    # --- step 4: fake trainer emits matched logprobs ---------------------
    # Matched (rollout_lp == trainer_lp) — ESS must be 1.0 end-to-end.
    fake_trainers = [
        {
            "trainer_logprobs": list(index["entries"][0]["response_logprobs"]),
            "engine": "fsdp",
            "sharding": "FULL_SHARD",
            "dtype": "bfloat16",
            "world_size": 4,
            "prompt_idx": index["entries"][0]["prompt_idx"],
        }
    ]

    # --- step 5: pair_reports --------------------------------------------
    reports = pair_mod.pair_reports(
        index=index,
        trainers=fake_trainers,
        trajectories_by_run_id={traj_with_prompt.run_id: traj_with_prompt},
        precision_class="bf16",
        trainer_engine_label="fsdp-bf16",
    )
    assert len(reports) == 1
    report = reports[0]
    assert report.num_policy_tokens == len(tokens)
    assert report.rollout_engine.name == "hermes-qwen3-32b-bf16"
    assert report.trainer_engine.name == "fsdp-bf16"
    assert report.precision_class == "bf16"
    # Matched logprobs across the whole pipeline => ESS == 1.0.
    assert report.ess == pytest.approx(1.0)
    assert report.delta_logprob_mean == pytest.approx(0.0)
    assert report.prompt_id == "math-q1-turn0"


def test_full_dense_pipeline_multitask_hash_pairing(
    proxy_mod: ModuleType,
    merger_mod: ModuleType,
    builder_mod: ModuleType,
    pair_mod: ModuleType,
    filler_mod: ModuleType,
) -> None:
    """Realistic multi-task run: two tasks share one sidecar (single
    proxy invocation, header-free). Probes are auto-keyed by hash of
    the first user message; the merger pairs each task's probes back
    to the right ShareGPT record purely by hash recomputation.
    Validates the setup-12 multi-task pairing path end-to-end."""
    task_a_text = "What is 2+2?"
    task_b_text = "What is 7*8?"
    # Build two synthetic responses, one per task.
    tokens_a = [
        ("The", 791, -0.1),
        (" answer", 4320, -0.2),
        (" 4", 19, -0.05),
    ]
    tokens_b = [
        ("The", 791, -0.1),
        (" answer", 4320, -0.2),
        (" 56", 5612, -0.05),
    ]

    # --- proxy: auto-derive prompt_index per task via hash ---------
    sidecar_a = proxy_mod.extract_probes(
        json.dumps({"messages": [{"role": "user", "content": task_a_text}]}).encode(),
        _synth_vllm_response(tokens_a),
        prompt_index=proxy_mod.derive_prompt_index(
            json.dumps({"messages": [{"role": "user", "content": task_a_text}]}).encode()
        ),
        turn_idx=0,
    )
    sidecar_b = proxy_mod.extract_probes(
        json.dumps({"messages": [{"role": "user", "content": task_b_text}]}).encode(),
        _synth_vllm_response(tokens_b),
        prompt_index=proxy_mod.derive_prompt_index(
            json.dumps({"messages": [{"role": "user", "content": task_b_text}]}).encode()
        ),
        turn_idx=0,
    )
    assert sidecar_a is not None and sidecar_b is not None
    # The two tasks hashed to different prompt_index values.
    assert sidecar_a["prompt_index"] != sidecar_b["prompt_index"]
    assert isinstance(sidecar_a["prompt_index"], str)

    # --- merger: pairs by hash recomputation per sharegpt record ---
    sharegpt_records = [
        {
            "prompt_index": 0,
            "conversations": [
                {"from": "system", "value": "sys"},
                {"from": "human", "value": task_a_text},
                {"from": "gpt", "value": "<think>\n</think>\nThe answer 4"},
            ],
            "completed": True,
            "partial": False,
        },
        {
            "prompt_index": 1,
            "conversations": [
                {"from": "system", "value": "sys"},
                {"from": "human", "value": task_b_text},
                {"from": "gpt", "value": "<think>\n</think>\nThe answer 56"},
            ],
            "completed": True,
            "partial": False,
        },
    ]
    tasks = [
        {"task_id": "qa-add", "task_text": task_a_text, "available_tools": []},
        {"task_id": "qa-mul", "task_text": task_b_text, "available_tools": []},
    ]
    trajectories = merger_mod.merge_to_trajectories(
        sharegpt_records=sharegpt_records,
        probe_records=[sidecar_a, sidecar_b],
        tasks=tasks,
        model_id="Qwen/Qwen3-32B",
        engine_label="hermes-qwen3-32b-bf16",
        engine_fingerprint="sha256:hermes-qwen3-32b-bf16-tp4-l40s",
    )
    # Each task's probes landed on the right trajectory.
    a_step = trajectories[0].assistant_steps()[0]
    b_step = trajectories[1].assistant_steps()[0]
    assert a_step.response_token_ids == [t[1] for t in tokens_a]
    assert b_step.response_token_ids == [t[1] for t in tokens_b]

    # --- filler + builder + pair: full pipeline ---------------------
    filled = []
    for traj in trajectories:
        f, _ = filler_mod.fill_trajectory(traj, _FakeTokenizer())
        filled.append(f)
    _, index = builder_mod.build_rollouts_and_index(filled, precision_class="bf16")
    assert len(index["entries"]) == 2  # one per task
    # Fake trainer matches each task's logprobs exactly → ESS=1 per turn.
    fake_trainers = [
        {
            "trainer_logprobs": list(e["response_logprobs"]),
            "engine": "fsdp",
            "sharding": "FULL_SHARD",
            "dtype": "bfloat16",
            "world_size": 4,
            "prompt_idx": e["prompt_idx"],
        }
        for e in index["entries"]
    ]
    reports = pair_mod.pair_reports(
        index=index,
        trainers=fake_trainers,
        trajectories_by_run_id={t.run_id: t for t in filled},
        precision_class="bf16",
        trainer_engine_label="fsdp-bf16",
    )
    # One report per task, both at ESS=1.0.
    assert len(reports) == 2
    assert {r.prompt_id for r in reports} == {"qa-add-turn0", "qa-mul-turn0"}
    for r in reports:
        assert r.ess == pytest.approx(1.0)


def test_full_dense_pipeline_propagates_divergence(
    proxy_mod: ModuleType,
    merger_mod: ModuleType,
    builder_mod: ModuleType,
    pair_mod: ModuleType,
    filler_mod: ModuleType,
) -> None:
    """A genuine rollout-vs-trainer logprob divergence on the wire
    must surface as a measurable ESS drop in the final report.
    Validates that no stage zeroes out the per-token deltas."""
    tokens = [(f"t{i}", 1000 + i, -0.1 - i * 0.01) for i in range(8)]
    sidecar = proxy_mod.extract_probes(
        b"{}", _synth_vllm_response(tokens), prompt_index=0, turn_idx=0
    )
    assert sidecar is not None
    trajectories = merger_mod.merge_to_trajectories(
        sharegpt_records=[_synth_sharegpt_record(0)],
        probe_records=[sidecar],
        tasks=[_task()],
        model_id="Qwen/Qwen3-32B",
        engine_label="hermes-qwen3-32b-fp8",
        engine_fingerprint="sha256:hermes-qwen3-32b-fp8",
    )
    traj, n_filled = filler_mod.fill_trajectory(trajectories[0], _FakeTokenizer())
    assert n_filled == 1
    _, index = builder_mod.build_rollouts_and_index([traj], precision_class="fp8")
    # Trainer disagrees by a per-token-varying delta. A constant
    # offset would leave ESS == 1.0 (the importance weights would be
    # uniform after normalisation); per-token variance is what causes
    # ESS to drop.
    rollout_lp = list(index["entries"][0]["response_logprobs"])
    trainer_lp = [lp + (0.5 if i % 2 == 0 else -0.5) for i, lp in enumerate(rollout_lp)]
    trainer = [
        {
            "trainer_logprobs": trainer_lp,
            "engine": "fsdp",
            "sharding": "FULL_SHARD",
            "dtype": "bfloat16",
            "world_size": 4,
            "prompt_idx": 0,
        }
    ]
    reports = pair_mod.pair_reports(
        index=index,
        trainers=trainer,
        trajectories_by_run_id={traj.run_id: traj},
        precision_class="fp8",
    )
    rep = reports[0]
    assert rep.num_policy_tokens == 8
    assert rep.delta_logprob_abs_mean == pytest.approx(0.5)
    # Per-token variance in the log-ratio drives ESS below 1.0; the
    # deltas must propagate end-to-end through every component.
    assert rep.ess < 0.99
    assert rep.max_abs_log_ratio == pytest.approx(0.5)
