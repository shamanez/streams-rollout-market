"""End-to-end MoE-side pipeline test (mirror of the dense e2e test).

Wires every cycle-3 v2 component on the MoE side through their pure-
function seams:

  1. ``proxy.extract_probes`` consumes a synthetic vLLM-shape
     chat-completions response with both per-token logprobs AND
     per-token ``routed_experts``; the sidecar record carries
     ``response_routed_experts``.
  2. ``merge_probes_into_trajectories.merge_to_trajectories`` stitches
     the sidecar onto a synthetic hermes-agent ShareGPT record and
     produces an ``AgentTrajectory`` whose assistant step carries
     ``response_routed_experts``.
  3. ``fill_prompt_token_ids.fill_trajectory`` re-derives
     ``prompt_token_ids`` (the proxy can't capture them from
     chat-completions).
  4. ``build_hermes_dense_input.build_rollouts_and_index`` flattens
     into the rollouts payload + index.
  5. A fake trainer payload supplies ``expert_ids`` matching the
     index's contract.
  6. ``pair_hermes_moe_reports.pair_reports`` joins index + trainer
     into one ``RouterMismatchReport``.

Asserts both happy path (zero flip rate when rollout/trainer agree)
and divergence (non-zero router_flip_rate when they disagree on the
first layer).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_script(name: str) -> ModuleType:
    spec_path = Path(__file__).resolve().parents[1] / "scripts" / "live" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"moe_e2e_{name}", spec_path)
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
    return _load_script("pair_hermes_moe_reports")


@pytest.fixture(scope="module")
def filler_mod() -> ModuleType:
    return _load_script("fill_prompt_token_ids")


class _FakeTokenizer:
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


# MoE shape: 3 response tokens × 2 layers × top-1 from 4-expert pool.
_NUM_LAYERS = 2
_NUM_EXPERTS = 4
_TOP_K = 1


def _vllm_response_with_routed(
    tokens: list[tuple[str, int, float, list[list[int]]]],
) -> bytes:
    """Build a synthetic vLLM response matching the documented shape.

    Per https://docs.vllm.ai/en/latest/training/routed_experts_replay/,
    routed_experts is a TOP-LEVEL field on each choice with shape
    ``[gen_len, num_moe_layers, top_k]`` — the per-token list is the
    OUTER axis of one ``choices[0].routed_experts`` array.
    """
    content = [{"token": tok, "logprob": lp} for tok, _, lp, _ in tokens]
    token_ids = [tid for _, tid, _, _ in tokens]
    routed_experts = [re for _, _, _, re in tokens]
    return json.dumps(
        {
            "id": "chatcmpl-moe-e2e",
            "prompt_token_ids": [40, 41, 42, 43, 44],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "answer"},
                    "logprobs": {"content": content},
                    "token_ids": token_ids,
                    "routed_experts": routed_experts,
                    "finish_reason": "stop",
                }
            ],
        }
    ).encode("utf-8")


def _synth_sharegpt_record() -> dict:
    return {
        "prompt_index": 0,
        "conversations": [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "What is 2+2?"},
            {"from": "gpt", "value": "<think>\n</think>\nAnswer: 4"},
        ],
        "completed": True,
        "partial": False,
    }


def _task() -> dict:
    return {"task_id": "moe-q1", "task_text": "What is 2+2?", "available_tools": []}


def test_full_moe_pipeline_e2e_matched(
    proxy_mod: ModuleType,
    merger_mod: ModuleType,
    builder_mod: ModuleType,
    pair_mod: ModuleType,
    filler_mod: ModuleType,
) -> None:
    """Matched rollout/trainer expert_ids end-to-end → zero flip rate."""
    rollout_experts = [
        [[0], [1]],
        [[1], [2]],
        [[2], [3]],
    ]
    tokens = [
        ("A", 100, -0.1, rollout_experts[0]),
        ("B", 101, -0.2, rollout_experts[1]),
        ("C", 102, -0.3, rollout_experts[2]),
    ]

    # 1. proxy
    sidecar = proxy_mod.extract_probes(
        b"{}", _vllm_response_with_routed(tokens), prompt_index=0, turn_idx=0
    )
    assert sidecar is not None
    assert sidecar["response_routed_experts"] == rollout_experts

    # 2. merger
    trajectories = merger_mod.merge_to_trajectories(
        sharegpt_records=[_synth_sharegpt_record()],
        probe_records=[sidecar],
        tasks=[_task()],
        model_id="Qwen/Qwen3-30B-A3B",
        engine_label="hermes-qwen3-30b-a3b-bf16",
        engine_fingerprint="sha256:hermes-qwen3-30b-a3b-bf16-tp2-l40s",
    )
    step = trajectories[0].assistant_steps()[0]
    assert step.response_routed_experts == rollout_experts

    # 3. filler (re-derives prompt_token_ids)
    traj, n_filled = filler_mod.fill_trajectory(trajectories[0], _FakeTokenizer())
    # Proxy already captured prompt_token_ids via return_token_ids.
    assert n_filled == 0

    # 4. builder
    _, index = builder_mod.build_rollouts_and_index([traj], precision_class="bf16")
    assert len(index["entries"]) == 1

    # 5. fake trainer with matched expert_ids
    trainers = [
        {
            "prompt_idx": index["entries"][0]["prompt_idx"],
            "model": traj.model_id,
            "engine": "fsdp",
            "num_layers": _NUM_LAYERS,
            "num_experts": _NUM_EXPERTS,
            "top_k": _TOP_K,
            "expert_ids": rollout_experts,
            "world_size": 4,
        }
    ]

    # 6. pair
    reports = pair_mod.pair_reports(
        index=index,
        trainers=trainers,
        trajectories_by_run_id={traj.run_id: traj},
        precision_class="bf16",
    )
    assert len(reports) == 1
    rep = reports[0]
    assert rep.num_tokens == 3
    assert rep.num_layers == _NUM_LAYERS
    assert rep.router_flip_rate == 0.0
    assert rep.layer_flip_rates == [0.0, 0.0]


def test_full_moe_pipeline_e2e_divergence(
    proxy_mod: ModuleType,
    merger_mod: ModuleType,
    builder_mod: ModuleType,
    pair_mod: ModuleType,
    filler_mod: ModuleType,
) -> None:
    """Layer-0 disagreement across all tokens → flip rate 50% overall,
    layer-0 100%, layer-1 0%."""
    rollout_experts = [[[0], [1]], [[1], [2]], [[2], [3]]]
    trainer_experts = [[[3], [1]], [[3], [2]], [[3], [3]]]
    tokens = [
        ("A", 100, -0.1, rollout_experts[0]),
        ("B", 101, -0.2, rollout_experts[1]),
        ("C", 102, -0.3, rollout_experts[2]),
    ]

    sidecar = proxy_mod.extract_probes(
        b"{}", _vllm_response_with_routed(tokens), prompt_index=0, turn_idx=0
    )
    trajectories = merger_mod.merge_to_trajectories(
        sharegpt_records=[_synth_sharegpt_record()],
        probe_records=[sidecar],
        tasks=[_task()],
        model_id="Qwen/Qwen3-30B-A3B",
        engine_label="hermes-qwen3-30b-a3b-fp8",
        engine_fingerprint="sha256:hermes-qwen3-30b-a3b-fp8-tp2-l40s",
    )
    traj, _ = filler_mod.fill_trajectory(trajectories[0], _FakeTokenizer())
    _, index = builder_mod.build_rollouts_and_index([traj], precision_class="fp8")
    trainers = [
        {
            "prompt_idx": 0,
            "model": traj.model_id,
            "engine": "megatron",
            "num_layers": _NUM_LAYERS,
            "num_experts": _NUM_EXPERTS,
            "top_k": _TOP_K,
            "expert_ids": trainer_experts,
            "world_size": 4,
        }
    ]
    reports = pair_mod.pair_reports(
        index=index,
        trainers=trainers,
        trajectories_by_run_id={traj.run_id: traj},
        precision_class="fp8",
        trainer_engine_label="megatron-lm",
    )
    rep = reports[0]
    assert rep.trainer_engine.name == "megatron-lm"
    assert rep.router_flip_rate == pytest.approx(0.5)
    assert rep.layer_flip_rates[0] == pytest.approx(1.0)
    assert rep.layer_flip_rates[1] == pytest.approx(0.0)
