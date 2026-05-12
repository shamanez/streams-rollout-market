"""Offline tests for scripts/live/run_fsdp_moe_reference.py helpers.

The script's heavy lifting (torch + FSDP + transformers forward pass)
only runs on the spot. These tests exercise the pure-Python pieces
that the multi-prompt support added:

  * ``load_rollouts``: prefers the multi-prompt ``/tmp/rollouts.json``
    list over the single ``/tmp/rollout.json``, and reports which
    output convention to follow.
  * ``routed_position_indices`` + ``format_fsdp_router_payload``
    contracts are preserved (the legacy helpers).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    spec_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "live" / "run_fsdp_moe_reference.py"
    )
    spec = importlib.util.spec_from_file_location("run_fsdp_moe_reference", spec_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return _load_module()


def test_load_rollouts_prefers_multi_list(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``/tmp/rollouts.json`` (list) exists it wins over the
    single-prompt ``/tmp/rollout.json``."""
    multi = tmp_path / "rollouts.json"
    single = tmp_path / "rollout.json"
    multi.write_text(
        json.dumps(
            [
                {"prompt_idx": 0, "prompt_token_ids": [1], "response_token_ids": [2]},
                {"prompt_idx": 1, "prompt_token_ids": [3], "response_token_ids": [4]},
            ]
        )
    )
    single.write_text(
        json.dumps({"prompt_idx": 99, "prompt_token_ids": [], "response_token_ids": []})
    )
    monkeypatch.setattr(script, "ROLLOUTS_PATH", multi)
    monkeypatch.setattr(script, "ROLLOUT_PATH", single)
    rollouts, is_multi = script.load_rollouts()
    assert is_multi is True
    assert [r["prompt_idx"] for r in rollouts] == [0, 1]


def test_load_rollouts_falls_back_to_single(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When only ``/tmp/rollout.json`` (single dict) exists, return a
    one-element list and ``is_multi=False`` so the legacy single-prompt
    output path stays active."""
    multi = tmp_path / "rollouts.json"
    single = tmp_path / "rollout.json"
    single.write_text(json.dumps({"prompt_token_ids": [1, 2], "response_token_ids": [3]}))
    monkeypatch.setattr(script, "ROLLOUTS_PATH", multi)
    monkeypatch.setattr(script, "ROLLOUT_PATH", single)
    rollouts, is_multi = script.load_rollouts()
    assert is_multi is False
    assert len(rollouts) == 1
    assert rollouts[0]["response_token_ids"] == [3]


def test_load_rollouts_rejects_non_list_multi(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/tmp/rollouts.json`` MUST be a list — a dict there is the
    operator confusing the two contracts and should surface clearly."""
    multi = tmp_path / "rollouts.json"
    multi.write_text(json.dumps({"prompt_token_ids": [], "response_token_ids": []}))
    monkeypatch.setattr(script, "ROLLOUTS_PATH", multi)
    monkeypatch.setattr(script, "ROLLOUT_PATH", tmp_path / "rollout.json")
    with pytest.raises(ValueError, match="must contain a list"):
        script.load_rollouts()


def test_routed_position_indices_matches_vllm_convention(
    script: ModuleType,
) -> None:
    """vLLM emits routed_experts for positions ``0..P+R-2``; the FSDP
    side must mirror that or RouterTrace pairing fails."""
    positions = script.routed_position_indices(prompt_len=3, response_len=2)
    assert positions == [0, 1, 2, 3]


def test_format_fsdp_router_payload_shape(script: ModuleType) -> None:
    """Payload schema is load-bearing — pair_hermes_moe_reports reads
    these exact keys."""
    payload = script.format_fsdp_router_payload(
        model_id="Qwen/Qwen3-30B-A3B",
        prompt_token_ids=[1, 2, 3],
        response_token_ids=[4, 5],
        num_layers=2,
        num_experts=4,
        top_k=1,
        expert_ids=[[[0], [1]], [[1], [2]], [[2], [3]], [[3], [0]]],
        seed=1234,
        world_size=4,
    )
    for key in (
        "model",
        "engine",
        "num_layers",
        "num_experts",
        "top_k",
        "expert_ids",
        "prompt_token_ids",
        "response_token_ids",
    ):
        assert key in payload
    assert payload["engine"] == "fsdp"
    assert payload["num_layers"] == 2
    assert payload["top_k"] == 1
