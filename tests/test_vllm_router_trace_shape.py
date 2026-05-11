"""Shape contract: a vLLM-side router trace must be compatible with the
megatron-side trace so ``compute_router_mismatch`` can pair them.

The live script ``scripts/live/run_vllm_moe_rollout.py`` produces
``/tmp/vllm_router.json`` from
``AsyncEngineArgs(enable_return_routed_experts=True)``. The trainer side
produces ``/tmp/megatron_router.json`` from the Megatron forward pass.
Both must agree on ``(num_tokens, num_layers, top_k)`` and have expert
IDs in ``[0, num_experts)``; otherwise ``RouterTrace`` rejects them.

This test imports the live script as a module (which is import-safe
because the vLLM imports are deferred inside the async helper) and
exercises only ``_routed_experts_to_lists`` and ``_engine_fingerprint``
plus the schema-shape assertions. It does NOT require vLLM to be
installed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from rollout_market.observatory.dense_mismatch_lab import EngineFingerprint
from rollout_market.observatory.router_mismatch_lab import (
    RouterTrace,
    compute_router_mismatch,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "live" / "run_vllm_moe_rollout.py"


def _load_script_module():
    """Load run_vllm_moe_rollout as a module without executing main()."""
    spec = importlib.util.spec_from_file_location(
        "run_vllm_moe_rollout", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_megatron_router_payload(
    *, num_tokens: int, num_layers: int, top_k: int, num_experts: int
) -> dict:
    """Stand-in for /tmp/megatron_router.json — same shape, deterministic IDs."""
    expert_ids = [
        [
            [(t + layer + k) % num_experts for k in range(top_k)]
            for layer in range(num_layers)
        ]
        for t in range(num_tokens)
    ]
    return {
        "model": "Qwen/Qwen3-30B-A3B",
        "engine": "megatron-lm",
        "engine_fingerprint": "sha256:megatron-0.13-bf16-Qwen_Qwen3-30B-A3B",
        "prompt_token_ids": [0] * 7,
        "response_token_ids": [0] * (num_tokens - 7 + 1),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "expert_ids": expert_ids,
        "seed": 1234,
        "rollout_label": "megatron-bf16",
    }


def _stub_vllm_routed_experts(
    *, num_tokens: int, num_layers: int, top_k: int, num_experts: int
) -> list[list[list[int]]]:
    """Stub a vLLM `routed_experts` ndarray-like value (nested lists work too)."""
    return [
        [
            [((t + 1) + layer + k) % num_experts for k in range(top_k)]
            for layer in range(num_layers)
        ]
        for t in range(num_tokens)
    ]


@pytest.fixture
def megatron_router_fixture(tmp_path: Path) -> dict:
    """A megatron_router.json-shaped payload written to a tmp file then read back.

    This mirrors how the live runbook hands the file between hosts (scp
    target → local JSON parse), so the test exercises the same
    on-disk shape contract.
    """
    num_tokens, num_layers, top_k, num_experts = 13, 48, 8, 128
    payload = _stub_megatron_router_payload(
        num_tokens=num_tokens,
        num_layers=num_layers,
        top_k=top_k,
        num_experts=num_experts,
    )
    out = tmp_path / "megatron_router.json"
    out.write_text(json.dumps(payload))
    return json.loads(out.read_text())


def test_vllm_routed_experts_coerced_to_nested_int_lists() -> None:
    """vLLM hands back a list-of-list-of-list-of-int. _routed_experts_to_lists
    should preserve the [token][layer][top_k] shape and force int."""
    mod = _load_script_module()
    raw = _stub_vllm_routed_experts(num_tokens=5, num_layers=4, top_k=3, num_experts=16)
    out = mod._routed_experts_to_lists(raw)
    assert isinstance(out, list)
    assert len(out) == 5
    assert all(len(t) == 4 for t in out)
    assert all(len(layer) == 3 for t in out for layer in t)
    assert all(isinstance(eid, int) for t in out for layer in t for eid in layer)


def test_vllm_engine_fingerprint_is_stable_and_includes_model() -> None:
    mod = _load_script_module()
    fp = mod._engine_fingerprint("Qwen/Qwen3-30B-A3B", "vllm-bf16", "0.22.0")
    assert fp.startswith("sha256:vllm-0.22.0-")
    assert "Qwen_Qwen3-30B-A3B" in fp
    # Stable across calls with the same inputs.
    assert fp == mod._engine_fingerprint("Qwen/Qwen3-30B-A3B", "vllm-bf16", "0.22.0")


def test_stub_vllm_trace_is_shape_compatible_with_megatron_fixture(
    megatron_router_fixture: dict,
) -> None:
    """End-to-end shape contract: a vLLM-produced trace built from the same
    (num_tokens, num_layers, top_k, num_experts) as the megatron fixture
    must round-trip through ``RouterTrace`` and pair cleanly in
    ``compute_router_mismatch``."""
    mt = megatron_router_fixture
    vllm_raw = _stub_vllm_routed_experts(
        num_tokens=len(mt["expert_ids"]),
        num_layers=mt["num_layers"],
        top_k=mt["top_k"],
        num_experts=mt["num_experts"],
    )
    mod = _load_script_module()
    vllm_expert_ids = mod._routed_experts_to_lists(vllm_raw)

    # Shape axes line up.
    assert len(vllm_expert_ids) == len(mt["expert_ids"])
    assert len(vllm_expert_ids[0]) == mt["num_layers"]
    assert len(vllm_expert_ids[0][0]) == mt["top_k"]

    # Each side validates as RouterTrace.
    rollout = RouterTrace(
        num_layers=mt["num_layers"],
        num_experts=mt["num_experts"],
        top_k=mt["top_k"],
        expert_ids=vllm_expert_ids,
    )
    trainer = RouterTrace(
        num_layers=mt["num_layers"],
        num_experts=mt["num_experts"],
        top_k=mt["top_k"],
        expert_ids=mt["expert_ids"],
    )

    # And compute_router_mismatch consumes the pair without raising.
    report = compute_router_mismatch(
        run_id="shape-test-001",
        model_id="Qwen/Qwen3-30B-A3B",
        rollout_engine=EngineFingerprint(
            name="vllm",
            version="0.22.0",
            fingerprint="sha256:vllm-0.22.0-vllm-bf16-Qwen_Qwen3-30B-A3B",
        ),
        trainer_engine=EngineFingerprint(
            name="megatron-lm",
            version="0.13",
            fingerprint="sha256:megatron-0.13-bf16-Qwen_Qwen3-30B-A3B",
        ),
        rollout_trace=rollout,
        trainer_trace=trainer,
    )
    assert 0.0 <= report.router_flip_rate <= 1.0
    assert 0.0 <= report.token_expert_disagreement_rate <= 1.0
    assert len(report.layer_flip_rates) == mt["num_layers"]
    assert report.num_tokens == len(mt["expert_ids"])


def test_routed_experts_rejects_wrong_top_level_shape() -> None:
    mod = _load_script_module()
    with pytest.raises(ValueError):
        mod._routed_experts_to_lists(42)  # not iterable in the expected way
