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

    # --- step 3: builder.build_rollouts_and_index ------------------------
    # The proxy can't capture prompt_token_ids from chat-completions, so
    # we synthesise them here as a stand-in for what the upcoming
    # tokenizer-side encoding step would emit. Without prompt_token_ids
    # the builder skips the turn; that's a separate test in
    # test_build_hermes_dense_input.py.
    step_with_prompt = step.model_copy(
        update={"prompt_token_ids": [10, 11, 12, 13, 14, 15, 16, 17]}
    )
    traj_with_prompt = traj.model_copy(update={"steps": [step_with_prompt]})
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


def test_full_dense_pipeline_propagates_divergence(
    proxy_mod: ModuleType,
    merger_mod: ModuleType,
    builder_mod: ModuleType,
    pair_mod: ModuleType,
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
    step = (
        trajectories[0]
        .assistant_steps()[0]
        .model_copy(update={"prompt_token_ids": list(range(20, 25))})
    )
    traj = trajectories[0].model_copy(update={"steps": [step]})
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
