"""Tests for STEER ``cleanup.consolidate_build_dense_input``.

The three pre-cleanup scripts (`build_dense_input.py`,
`build_dense_input_fp8.py`, `build_dense_input_sglang.py`) collapse
into one env-driven script with a `--variant` flag. These tests pull
the script in as a module and exercise every variant by monkey-patching
`Path.read_text` so no /tmp file IO is needed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from rollout_market.observatory.dense_mismatch_lab import (
    DenseMismatchReport,
    EngineFingerprint,
    compute_dense_mismatch,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "live" / "build_dense_input.py"

SPEC = importlib.util.spec_from_file_location("build_dense_input", SCRIPT_PATH)
build_dense_input = importlib.util.module_from_spec(SPEC)
sys.modules["build_dense_input"] = build_dense_input
SPEC.loader.exec_module(build_dense_input)  # type: ignore[union-attr]


VARIANTS = list(build_dense_input.VARIANTS.keys())


def test_variant_set_matches_steer_directive():
    assert set(VARIANTS) == {
        "vllm-bf16",
        "vllm-fp8",
        "sglang-bf16",
        "sglang-fp8",
        "megatron-bf16",
    }


def test_legacy_scripts_removed():
    """STEER: the two _fp8 / _sglang variants must be deleted."""
    assert not (REPO_ROOT / "scripts" / "live" / "build_dense_input_fp8.py").exists()
    assert not (REPO_ROOT / "scripts" / "live" / "build_dense_input_sglang.py").exists()
    # The consolidated script still exists.
    assert SCRIPT_PATH.exists()


def _stub_rollout_payload() -> dict:
    return {
        "model": "Qwen/Qwen3-32B",
        "seed": 1234,
        "rollout_logprobs": [-0.5, -0.4, -0.45, -0.6],
        "prompt_text": "demo",
        "response_text": "demo response",
        "trainer_reference": "Qwen/Qwen3-32B",
    }


def _stub_trainer_payload() -> dict:
    return {
        "trainer_logprobs": [-0.52, -0.41, -0.46, -0.6],
    }


@pytest.mark.parametrize("variant_name", VARIANTS)
def test_every_variant_produces_valid_dense_input(variant_name: str):
    """Build the payload, feed it to `compute_dense_mismatch`, and ensure
    the result is a valid DenseMismatchReport with the variant-specific
    rollout_engine fingerprint / precision_class / quantization_class."""
    variant = build_dense_input.VARIANTS[variant_name]
    payload = build_dense_input.build_payload(
        variant, _stub_rollout_payload(), _stub_trainer_payload()
    )

    # (a) Valid DenseMismatchReport-shaped JSON.
    report = compute_dense_mismatch(
        run_id=payload["run_id"],
        model_id=payload["model_id"],
        checkpoint_digest=payload["checkpoint_digest"],
        tokenizer_hash=payload["tokenizer_hash"],
        rollout_engine=EngineFingerprint.model_validate(payload["rollout_engine"]),
        trainer_engine=EngineFingerprint.model_validate(payload["trainer_engine"]),
        precision_class=payload["precision_class"],
        prompt_id=payload["prompt_id"],
        rollout_logprobs=payload["rollout_logprobs"],
        trainer_logprobs=payload["trainer_logprobs"],
        mask=payload["mask"],
        quantization_class=payload["quantization_class"],
    )
    assert isinstance(report, DenseMismatchReport)

    # (b) Engine fingerprint / precision_class / quantization_class match
    # the variant metadata.
    assert payload["rollout_engine"]["name"] == variant.rollout_engine_name
    assert payload["rollout_engine"]["fingerprint"] == variant.rollout_engine_fingerprint
    assert payload["precision_class"] == variant.precision_class
    assert payload["quantization_class"] == variant.quantization_class
    assert payload["trainer_engine"]["name"] == variant.trainer_engine_name


def test_main_reads_via_monkeypatched_read_text(monkeypatch, tmp_path):
    """End-to-end CLI test: monkeypatch Path.read_text to return stubs
    and assert the script writes a valid DenseMismatchReport-shaped JSON
    to its output path."""
    rollout_stub = json.dumps(_stub_rollout_payload())
    trainer_stub = json.dumps(_stub_trainer_payload())

    real_read_text = Path.read_text

    def fake_read_text(self: Path, *a, **kw) -> str:  # type: ignore[no-redef]
        name = str(self)
        if "rollout" in name and "input" not in name:
            return rollout_stub
        if "trainer" in name and "input" not in name:
            return trainer_stub
        # Fall through to the real implementation for everything else
        # (e.g. reading back the script's own output for assertion).
        return real_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    out = tmp_path / "out.json"
    rc = build_dense_input.main(["--variant", "vllm-fp8", "--out", str(out)])
    assert rc == 0
    written = json.loads(out.read_text())
    # Round-trips through compute_dense_mismatch.
    report = compute_dense_mismatch(
        run_id=written["run_id"],
        model_id=written["model_id"],
        checkpoint_digest=written["checkpoint_digest"],
        tokenizer_hash=written["tokenizer_hash"],
        rollout_engine=EngineFingerprint.model_validate(written["rollout_engine"]),
        trainer_engine=EngineFingerprint.model_validate(written["trainer_engine"]),
        precision_class=written["precision_class"],
        prompt_id=written["prompt_id"],
        rollout_logprobs=written["rollout_logprobs"],
        trainer_logprobs=written["trainer_logprobs"],
        mask=written["mask"],
        quantization_class=written["quantization_class"],
    )
    assert report.precision_class == "fp8"
    assert report.quantization_class == "fp8"


def test_vllm_bf16_falls_back_to_legacy_input_paths(monkeypatch, tmp_path):
    """vllm-bf16 must read /tmp/rollout.json + /tmp/trainer.json when the
    new _vllm-bf16 paths are absent (back-compat clause)."""
    rollout_stub = json.dumps(_stub_rollout_payload())
    trainer_stub = json.dumps(_stub_trainer_payload())
    legacy_seen = {"rollout": False, "trainer": False}

    def fake_exists(self: Path) -> bool:  # type: ignore[no-redef]
        # New _vllm-bf16 paths don't exist; legacy paths do.
        s = str(self)
        if "vllm-bf16" in s:
            return False
        if s == "/tmp/rollout.json":
            return True
        if s == "/tmp/trainer.json":
            return True
        # Anything else (e.g. the output path's parent) → True.
        return True

    def fake_read_text(self: Path, *a, **kw) -> str:  # type: ignore[no-redef]
        s = str(self)
        if s == "/tmp/rollout.json":
            legacy_seen["rollout"] = True
            return rollout_stub
        if s == "/tmp/trainer.json":
            legacy_seen["trainer"] = True
            return trainer_stub
        raise FileNotFoundError(s)

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(Path, "read_text", fake_read_text)
    out = tmp_path / "out.json"
    rc = build_dense_input.main(["--variant", "vllm-bf16", "--out", str(out)])
    assert rc == 0
    assert legacy_seen["rollout"] is True
    assert legacy_seen["trainer"] is True


def test_trainer_label_override_swaps_trainer_engine_metadata():
    """The new --trainer-label flag lets a rollout variant be paired with
    a different trainer-side reference without spawning a new variant."""
    variant = build_dense_input.VARIANTS["vllm-fp8"]
    rollout = _stub_rollout_payload()
    trainer = _stub_trainer_payload()

    # Default: trainer engine is the variant's static hf-transformers.
    default_payload = build_dense_input.build_payload(variant, rollout, trainer)
    assert default_payload["trainer_engine"]["name"] == "hf-transformers"

    # Override: switch to fsdp.
    fsdp_override = build_dense_input.TRAINER_OVERRIDES["fsdp"]
    overridden = build_dense_input.build_payload(
        variant, rollout, trainer, trainer_override=fsdp_override
    )
    assert overridden["trainer_engine"]["name"] == "fsdp"
    assert overridden["trainer_engine"]["version"] == "torch-2.11"
    assert overridden["trainer_engine"]["fingerprint"] == (
        "sha256:fsdp-tp4-fullshard-bf16-cuda13"
    )
    # Run-id is suffixed so dashboards don't collide with the
    # hf-transformers version of the same rollout variant.
    assert overridden["run_id"].endswith("-vs-fsdp")
    assert overridden["run_id"] != default_payload["run_id"]
    # Rollout side is unchanged.
    assert overridden["rollout_engine"]["name"] == "vllm-fp8"
    assert overridden["precision_class"] == "fp8"

    # And the override note is recorded so downstream readers can grep
    # the report's notes for the trainer-side substitution.
    assert any("Trainer-side override: fsdp" in n for n in overridden["notes"])


def test_trainer_label_no_op_when_matches_variant_default():
    """Passing --trainer-label that matches the variant's static
    trainer name is a no-op (no run-id suffix, no override note)."""
    variant = build_dense_input.VARIANTS["vllm-fp8"]
    # vllm-fp8's default trainer is hf-transformers.
    hf_override = build_dense_input.TRAINER_OVERRIDES["hf-transformers"]
    payload = build_dense_input.build_payload(
        variant, _stub_rollout_payload(), _stub_trainer_payload(),
        trainer_override=hf_override,
    )
    assert payload["run_id"] == variant.run_id
    assert payload["trainer_engine"]["name"] == "hf-transformers"
    assert all("override" not in n.lower() for n in payload["notes"])
