"""Shape contract for scripts/live/build_router_input_pair.py.

Pure-Python tests on ``build_payload`` and the static ``VARIANTS``
table. The CLI ``main()`` is exercised indirectly by the live-run
fixtures under ``.claude/evidence/``.

Load-bearing invariants:
  - The four variant labels cover the (engine × precision × trainer)
    combinations that map to MoE matrix cells: vllm-{bf16,fp8} ×
    {fsdp,megatron}-bf16.
  - ``build_payload`` raises on axis mismatch (num_tokens, num_layers,
    num_experts, top_k) so RouterTrace never sees a corrupted pair.
  - Payload schema matches what ``rollout_market.cli.router_mismatch_lab``
    expects: rollout_engine + trainer_engine dicts with name/version/
    fingerprint, rollout_trace + trainer_trace with the four axes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "live" / "build_router_input_pair.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "build_router_input_pair", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec so dataclasses can resolve
    # forward references against the module's namespace.
    sys.modules["build_router_input_pair"] = module
    spec.loader.exec_module(module)
    return module


def _stub_trace(*, num_tokens: int, num_layers: int, top_k: int, num_experts: int, model: str = "Qwen/Qwen3-30B-A3B") -> dict:
    return {
        "model": model,
        "engine": "vllm",
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "expert_ids": [
            [
                [(t + layer + k) % num_experts for k in range(top_k)]
                for layer in range(num_layers)
            ]
            for t in range(num_tokens)
        ],
    }


def test_variants_cover_moe_matrix_cells() -> None:
    """The four labels span the (vllm precision × trainer engine) matrix."""
    mod = _load_script_module()
    labels = set(mod.VARIANTS.keys())
    assert labels == {
        "vllm-bf16-fsdp-bf16",
        "vllm-fp8-fsdp-bf16",
        "vllm-bf16-megatron-bf16",
        "vllm-fp8-megatron-bf16",
    }
    # Trainer fingerprints align with build_dense_input.py's TRAINER_OVERRIDES.
    fsdp_fp = mod.VARIANTS["vllm-bf16-fsdp-bf16"].trainer.fingerprint
    assert "fsdp" in fsdp_fp
    megatron_fp = mod.VARIANTS["vllm-bf16-megatron-bf16"].trainer.fingerprint
    assert "megatron-core" in megatron_fp


def test_build_payload_pairs_matched_traces() -> None:
    """Two identical-axes traces produce a valid payload with the right keys."""
    mod = _load_script_module()
    rollout = _stub_trace(num_tokens=83, num_layers=48, top_k=8, num_experts=128)
    trainer = _stub_trace(num_tokens=83, num_layers=48, top_k=8, num_experts=128)
    variant = mod.VARIANTS["vllm-bf16-fsdp-bf16"]
    payload = mod.build_payload(rollout=rollout, trainer=trainer, variant=variant)
    for k in ("run_id", "model_id", "rollout_engine", "trainer_engine",
              "rollout_trace", "trainer_trace", "notes"):
        assert k in payload, f"missing key {k}"
    assert payload["rollout_engine"]["name"] == "vllm"
    assert payload["trainer_engine"]["name"] == "fsdp"
    assert payload["rollout_trace"]["num_layers"] == 48
    assert len(payload["rollout_trace"]["expert_ids"]) == 83


def test_build_payload_raises_on_axis_mismatch() -> None:
    """Different num_layers must raise — RouterTrace would later reject."""
    mod = _load_script_module()
    rollout = _stub_trace(num_tokens=10, num_layers=48, top_k=8, num_experts=128)
    trainer = _stub_trace(num_tokens=10, num_layers=64, top_k=8, num_experts=128)
    variant = mod.VARIANTS["vllm-bf16-fsdp-bf16"]
    with pytest.raises(ValueError, match="num_layers"):
        mod.build_payload(rollout=rollout, trainer=trainer, variant=variant)


def test_build_payload_raises_on_num_tokens_mismatch() -> None:
    """Different num_tokens must raise — pairing-by-position requires equal length."""
    mod = _load_script_module()
    rollout = _stub_trace(num_tokens=83, num_layers=48, top_k=8, num_experts=128)
    trainer = _stub_trace(num_tokens=63, num_layers=48, top_k=8, num_experts=128)
    variant = mod.VARIANTS["vllm-bf16-fsdp-bf16"]
    with pytest.raises(ValueError, match="num_tokens"):
        mod.build_payload(rollout=rollout, trainer=trainer, variant=variant)


def test_build_payload_run_id_suffix() -> None:
    """run_suffix is appended for disambiguating multiple runs of the same variant."""
    mod = _load_script_module()
    rollout = _stub_trace(num_tokens=5, num_layers=2, top_k=2, num_experts=4)
    trainer = _stub_trace(num_tokens=5, num_layers=2, top_k=2, num_experts=4)
    variant = mod.VARIANTS["vllm-fp8-megatron-bf16"]
    p1 = mod.build_payload(rollout=rollout, trainer=trainer, variant=variant, run_suffix="001")
    p2 = mod.build_payload(rollout=rollout, trainer=trainer, variant=variant, run_suffix="002")
    assert p1["run_id"].endswith("-001")
    assert p2["run_id"].endswith("-002")
    assert p1["run_id"] != p2["run_id"]


def test_cli_main_writes_payload(tmp_path: Path) -> None:
    """End-to-end: CLI writes a JSON file with the right structure."""
    mod = _load_script_module()
    rollout = _stub_trace(num_tokens=5, num_layers=2, top_k=2, num_experts=4)
    trainer = _stub_trace(num_tokens=5, num_layers=2, top_k=2, num_experts=4)
    r_path = tmp_path / "rollout.json"
    t_path = tmp_path / "trainer.json"
    out_path = tmp_path / "out.json"
    r_path.write_text(json.dumps(rollout))
    t_path.write_text(json.dumps(trainer))

    rc = mod.main([
        "--rollout-input", str(r_path),
        "--trainer-input", str(t_path),
        "--variant", "vllm-bf16-fsdp-bf16",
        "--out", str(out_path),
    ])
    assert rc == 0
    payload = json.loads(out_path.read_text())
    assert payload["rollout_engine"]["name"] == "vllm"
    assert payload["trainer_engine"]["name"] == "fsdp"
    assert len(payload["rollout_trace"]["expert_ids"]) == 5
