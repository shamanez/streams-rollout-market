"""Shape contract for scripts/live/run_fsdp_moe_reference.py.

The script's torch / transformers imports are deferred inside
``main()``; this test loads the script as a module (import-safe) and
exercises only the pure-Python helpers
(``routed_position_indices``, ``format_fsdp_router_payload``,
``_engine_fingerprint``).

Load-bearing invariants under test:
  - Position-index coverage matches vLLM's ``routed_experts`` length
    (``P + R - 1`` positions); needed so RouterTrace pairs cleanly.
  - Payload schema matches ``/tmp/vllm_router.json`` keys so the
    router-mismatch fixture builder can read either side.
  - Engine fingerprint is stable across calls and includes both model
    name and world size.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "live" / "run_fsdp_moe_reference.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "run_fsdp_moe_reference", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_routed_position_indices_matches_vllm_length() -> None:
    """For prompt_len=23, response_len=128 (typical run): length is 150."""
    mod = _load_script_module()
    positions = mod.routed_position_indices(23, 128)
    assert len(positions) == 23 + 128 - 1
    assert positions[0] == 0
    assert positions[-1] == 23 + 128 - 2
    # Edge cases.
    assert mod.routed_position_indices(0, 0) == []
    assert mod.routed_position_indices(1, 0) == []
    assert mod.routed_position_indices(0, 1) == []
    assert mod.routed_position_indices(1, 1) == [0]


def test_engine_fingerprint_is_stable_and_includes_model() -> None:
    mod = _load_script_module()
    fp = mod._engine_fingerprint("Qwen/Qwen3-30B-A3B", 4)
    assert fp.startswith("sha256:fsdp-")
    assert "Qwen_Qwen3-30B-A3B" in fp
    assert "ws4" in fp
    # Stable across calls.
    assert fp == mod._engine_fingerprint("Qwen/Qwen3-30B-A3B", 4)
    # Different world size → different fingerprint.
    assert fp != mod._engine_fingerprint("Qwen/Qwen3-30B-A3B", 8)


def test_format_fsdp_router_payload_matches_vllm_router_schema() -> None:
    """Payload keys must match /tmp/vllm_router.json (run_vllm_moe_rollout.py)."""
    mod = _load_script_module()
    payload = mod.format_fsdp_router_payload(
        model_id="Qwen/Qwen3-30B-A3B",
        prompt_token_ids=[1, 2, 3],
        response_token_ids=[4, 5],
        num_layers=48,
        num_experts=128,
        top_k=8,
        expert_ids=[[[0] * 8 for _ in range(48)] for _ in range(4)],
        seed=1234,
        world_size=4,
    )
    required_keys = {
        "model",
        "engine",
        "engine_fingerprint",
        "prompt_token_ids",
        "response_token_ids",
        "num_layers",
        "num_experts",
        "top_k",
        "expert_ids",
        "seed",
    }
    missing = required_keys - set(payload.keys())
    assert not missing, f"missing keys: {missing}"
    assert payload["engine"] == "fsdp"
    assert payload["num_layers"] == 48
    assert payload["num_experts"] == 128
    assert payload["top_k"] == 8
    assert payload["sharding"] == "FULL_SHARD"
    # Expert IDs shape: [num_tokens][num_layers][top_k]
    assert len(payload["expert_ids"]) == 4
    assert len(payload["expert_ids"][0]) == 48
    assert len(payload["expert_ids"][0][0]) == 8


def test_payload_shape_matches_vllm_router_module_keys() -> None:
    """The FSDP-MoE payload should be a drop-in for the same keys vLLM emits."""
    fsdp = _load_script_module()
    vllm_path = REPO_ROOT / "scripts" / "live" / "run_vllm_moe_rollout.py"
    spec = importlib.util.spec_from_file_location("run_vllm_moe_rollout", vllm_path)
    assert spec is not None and spec.loader is not None
    vllm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vllm_mod)

    fsdp_payload = fsdp.format_fsdp_router_payload(
        model_id="Qwen/Qwen3-30B-A3B",
        prompt_token_ids=[1],
        response_token_ids=[2],
        num_layers=2,
        num_experts=4,
        top_k=2,
        expert_ids=[[[0, 1], [2, 3]]],  # 1 token, 2 layers, top_k=2
        seed=1234,
        world_size=4,
    )
    # The vLLM script's _engine_fingerprint and _routed_experts_to_lists
    # are the canonical reference; we only need our payload to expose the
    # same axis names so build_router_input can pair the two.
    for k in ("num_layers", "num_experts", "top_k", "expert_ids",
              "prompt_token_ids", "response_token_ids", "engine_fingerprint"):
        assert k in fsdp_payload, f"FSDP payload missing {k}"
    # Sanity-check the vLLM helper is callable from the test path too.
    assert callable(vllm_mod._engine_fingerprint)
