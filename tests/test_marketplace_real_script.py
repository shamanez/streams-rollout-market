"""Tests for STEER ``cleanup.consolidate_marketplace_real``.

The pre-cleanup scripts (`marketplace_real.py`, `marketplace_real_fp8.py`)
collapse into one env-driven entry point with a `--variant {bf16, fp8}`
flag. These tests pull the consolidated script in as a module and
exercise each variant's honest-case SampleGroup construction; the fp8
variant must be rejected by `validate_group_against_lease` when the
manifest pins bf16.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

from rollout_market.validators import validate_group_against_lease

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "live" / "marketplace_real.py"

SPEC = importlib.util.spec_from_file_location("marketplace_real", SCRIPT_PATH)
marketplace_real = importlib.util.module_from_spec(SPEC)
sys.modules["marketplace_real"] = marketplace_real
SPEC.loader.exec_module(marketplace_real)  # type: ignore[union-attr]


def _stub_rollout() -> dict:
    return {
        "prompt_token_ids": [1, 2, 3],
        "response_token_ids": [4, 5, 6, 7],
        "rollout_logprobs": [-0.4, -0.5, -0.45, -0.6],
        "model": "Qwen/Qwen3-32B",
        "seed": 1234,
    }


def test_variant_set_matches_steer_directive():
    assert set(marketplace_real.VARIANTS) == {"bf16", "fp8"}


def test_legacy_fp8_script_removed():
    assert not (REPO_ROOT / "scripts" / "live" / "marketplace_real_fp8.py").exists()
    assert SCRIPT_PATH.exists()


def test_bf16_honest_group_is_accepted():
    variant = marketplace_real.VARIANTS["bf16"]
    rollout = _stub_rollout()
    group = marketplace_real.build_honest_group(
        variant, rollout=rollout, group_id="g-bf16-honest"
    )
    manifest = marketplace_real.build_manifest(variant)
    lease = marketplace_real.build_lease(variant)
    now = lease.issued_at + timedelta(seconds=30)
    decision = validate_group_against_lease(group, lease, manifest, now=now)
    assert decision is None, f"unexpected rejection: {decision}"


def test_fp8_honest_group_is_rejected_under_bf16_manifest():
    """STEER acceptance: the fp8 variant's honest case must be rejected
    by the validators when the manifest pins bf16."""
    variant = marketplace_real.VARIANTS["fp8"]
    rollout = _stub_rollout()
    group = marketplace_real.build_honest_group(
        variant, rollout=rollout, group_id="g-fp8-honest"
    )
    manifest = marketplace_real.build_manifest(variant)  # pins bf16 per Variant
    lease = marketplace_real.build_lease(variant)         # required bf16
    now = lease.issued_at + timedelta(seconds=30)
    decision = validate_group_against_lease(group, lease, manifest, now=now)
    assert decision is not None, "fp8 worker on bf16 manifest must be rejected"
    assert "precision" in decision.rejection_reason.value.lower(), (
        f"expected precision_mismatch, got {decision.rejection_reason}"
    )


def test_fp8_honest_group_is_accepted_when_manifest_pins_fp8():
    """Same fp8 SampleGroup is accepted when both lease and manifest
    pin fp8. This is the lease.model_copy path the script exercises in
    its second card."""
    variant = marketplace_real.VARIANTS["fp8"]
    rollout = _stub_rollout()
    group = marketplace_real.build_honest_group(
        variant, rollout=rollout, group_id="g-fp8-honest"
    )
    manifest = marketplace_real.build_manifest(variant).model_copy(
        update={"precision_class": "fp8"}
    )
    lease = marketplace_real.build_lease(variant).model_copy(
        update={"required_precision_class": "fp8"}
    )
    now = lease.issued_at + timedelta(seconds=30)
    decision = validate_group_against_lease(group, lease, manifest, now=now)
    assert decision is None, f"unexpected rejection: {decision}"


def test_bf16_toxic_tokenizer_group_is_rejected():
    variant = marketplace_real.VARIANTS["bf16"]
    rollout = _stub_rollout()
    group = marketplace_real.build_honest_group(
        variant, rollout=rollout, group_id="g-tox-tok",
        tokenizer_hash="tok-WRONG",
    )
    manifest = marketplace_real.build_manifest(variant)
    lease = marketplace_real.build_lease(variant)
    now = lease.issued_at + timedelta(seconds=30)
    decision = validate_group_against_lease(group, lease, manifest, now=now)
    assert decision is not None
    assert "tokenizer" in decision.rejection_reason.value.lower()


def test_bf16_no_logprobs_group_is_rejected():
    """Pass an all-zero logprob list; validators should flag missing
    behaviour-token logprobs per the same path the script exercises."""
    variant = marketplace_real.VARIANTS["bf16"]
    rollout = _stub_rollout()
    group = marketplace_real.build_honest_group(
        variant, rollout=rollout, group_id="g-tox-noprob",
        logprobs_override=[0.0] * len(rollout["response_token_ids"]),
    )
    manifest = marketplace_real.build_manifest(variant)
    lease = marketplace_real.build_lease(variant)
    now = lease.issued_at + timedelta(seconds=30)
    decision = validate_group_against_lease(group, lease, manifest, now=now)
    assert decision is not None
    assert "logprob" in decision.rejection_reason.value.lower()
