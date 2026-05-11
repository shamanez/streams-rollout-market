"""Validate the marketplace stack against the real Qwen3-32B rollout.

STEER ``cleanup.consolidate_marketplace_real``: the two pre-cleanup
scripts (``marketplace_real.py``, ``marketplace_real_fp8.py``) collapse
into this single env-driven entry point. Pick a variant with
``--variant``:

* ``bf16`` (default) — reads ``/tmp/rollout_bf16.json`` +
  ``/tmp/trainer_bf16.json`` (with ``/tmp/rollout.json`` +
  ``/tmp/trainer.json`` as the legacy fallback) and builds the
  honest + toxic_tokenizer + toxic_no_logprobs trio.
* ``fp8`` — reads ``/tmp/rollout_fp8.json`` + ``/tmp/trainer_fp8.json``
  and builds the realistic precision-mismatch case where a fp8 worker
  submits against a bf16-pinned manifest (validators MUST reject) and
  the matching fp8-pinned manifest (validators must accept).

Both variants then run the full validators + OPBC + LiveStore +
TrainerClient + FeedbackAggregator stack and print one line per state
transition. The module exposes `build_honest_group(variant, ...)` and
`build_lease(variant)` so the test suite can construct the same
SampleGroups without invoking `main()`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from rollout_market.contracts import (
    PolicyManifest,
    RolloutLease,
    SampleGroup,
    Trajectory,
)
from rollout_market.feedback import FeedbackAggregator, TrainerFeedback
from rollout_market.livestore import InMemoryLiveStore
from rollout_market.opbc import compute_budget_report, decide_group
from rollout_market.trainer_client import ReplayTier, TrainerClient
from rollout_market.validators import validate_group_against_lease


VariantName = Literal["bf16", "fp8"]


@dataclass(frozen=True)
class Variant:
    label: VariantName
    rollout_default: Path
    trainer_default: Path
    legacy_rollout: Path | None
    legacy_trainer: Path | None
    worker_id: str
    engine_name: str
    lease_id: str
    job_id: str
    assignment_id: str
    idempotency_key: str
    precision_class: str  # what the rollout side declares
    manifest_precision_class: str  # what the policy manifest pins (intent)


VARIANTS: dict[str, Variant] = {
    "bf16": Variant(
        label="bf16",
        rollout_default=Path("/tmp/rollout_bf16.json"),
        trainer_default=Path("/tmp/trainer_bf16.json"),
        legacy_rollout=Path("/tmp/rollout.json"),
        legacy_trainer=Path("/tmp/trainer.json"),
        worker_id="vllm-l40s-spot",
        engine_name="vllm",
        lease_id="lease-live-001",
        job_id="job-live-001",
        assignment_id="assign-live-001",
        idempotency_key="idem-live-001",
        precision_class="bf16",
        manifest_precision_class="bf16",
    ),
    "fp8": Variant(
        label="fp8",
        rollout_default=Path("/tmp/rollout_fp8.json"),
        trainer_default=Path("/tmp/trainer_fp8.json"),
        legacy_rollout=None,
        legacy_trainer=None,
        worker_id="vllm-fp8-l40s-spot",
        engine_name="vllm-fp8",
        lease_id="lease-live-fp8-001",
        job_id="job-live-fp8-001",
        assignment_id="assign-live-fp8-001",
        idempotency_key="idem-live-fp8-001",
        precision_class="fp8",
        manifest_precision_class="bf16",
    ),
}


def _resolve_input_paths(variant: Variant) -> tuple[Path, Path]:
    rollout = variant.rollout_default
    trainer = variant.trainer_default
    if variant.label == "bf16":
        if not rollout.exists() and variant.legacy_rollout and variant.legacy_rollout.exists():
            rollout = variant.legacy_rollout
        if not trainer.exists() and variant.legacy_trainer and variant.legacy_trainer.exists():
            trainer = variant.legacy_trainer
    return rollout, trainer


def build_manifest(variant: Variant) -> PolicyManifest:
    return PolicyManifest(
        policy_version="v-live-001",
        checkpoint_digest="sha256:hf-cache-Qwen3-32B-bf16-snapshot",
        tokenizer_hash="tok-qwen3",
        model_config_hash="cfg-qwen3-32b",
        precision_class=variant.manifest_precision_class,
    )


def build_lease(variant: Variant) -> RolloutLease:
    issued = datetime(2026, 5, 10, 7, 30, tzinfo=timezone.utc)
    return RolloutLease(
        lease_id=variant.lease_id,
        job_id=variant.job_id,
        assignment_id=variant.assignment_id,
        worker_id=variant.worker_id,
        policy_version="v-live-001",
        required_precision_class=variant.manifest_precision_class,
        max_context_tokens=4096,
        group_size=1,
        idempotency_key=variant.idempotency_key,
        issued_at=issued,
        expires_at=issued + timedelta(minutes=30),
    )


def build_honest_group(
    variant: Variant,
    *,
    rollout: dict,
    group_id: str = "g-live-honest",
    tokenizer_hash: str = "tok-qwen3",
    logprobs_override: list[float] | None = None,
) -> SampleGroup:
    prompt_ids = rollout["prompt_token_ids"]
    response_ids = rollout["response_token_ids"]
    n = len(response_ids)
    return SampleGroup(
        group_id=group_id,
        task_id="rl-explainer",
        prompt_id="p-rl-explainer-001",
        worker_id=variant.worker_id,
        assignment_id=variant.assignment_id,
        policy_version="v-live-001",
        checkpoint_digest="sha256:hf-cache-Qwen3-32B-bf16-snapshot",
        tokenizer_hash=tokenizer_hash,
        sampling_config_hash="cfg-temp07-top0.95-seed1234",
        engine_name=variant.engine_name,
        engine_version="0.20.1",
        device_type="L40S",
        precision_class=variant.precision_class,
        prompt_token_ids=prompt_ids,
        trajectories=[
            Trajectory(
                response_token_ids=response_ids,
                rollout_logprobs=(
                    logprobs_override
                    if logprobs_override is not None
                    else rollout["rollout_logprobs"]
                ),
                action_mask=[1] * n,
                tool_output_mask=[0] * n,
                reward=1.0,
            )
        ],
        policy_lag_steps=0,
    )


def _run_bf16(rollout: dict, trainer: dict, variant: Variant, store: InMemoryLiveStore) -> None:
    """bf16 path: honest + toxic_tokenizer + toxic_no_logprobs trio."""
    n = len(rollout["response_token_ids"])
    manifest = build_manifest(variant)
    lease = build_lease(variant)
    now = lease.issued_at + timedelta(seconds=30)
    trainer_logprobs = trainer["trainer_logprobs"]

    print("=== honest (real Qwen3-32B vLLM rollout) ===")
    g_h = build_honest_group(variant, rollout=rollout, group_id="g-live-honest")
    rejected = validate_group_against_lease(g_h, lease, manifest, now=now)
    assert rejected is None, f"unexpected rejection: {rejected}"
    print("  validators -> PASS (contract clean)")
    report = compute_budget_report(g_h, trainer_logprobs)
    decision = decide_group(report)
    print(f"  OPBC -> {decision.recommended_action} ({[r.value for r in decision.decision_reasons]})")
    print(f"  ESS={report.effective_sample_size:.4f}, max|log_ratio|={report.max_abs_log_ratio:.4f}")
    store.submit(g_h, decision)

    print()
    print("=== toxic_tokenizer ===")
    g_t = build_honest_group(
        variant, rollout=rollout, group_id="g-live-tox-tok",
        tokenizer_hash="tok-WRONG",
    )
    rejected = validate_group_against_lease(g_t, lease, manifest, now=now)
    print(f"  validators -> {rejected.rejection_reason.value if rejected else 'PASS'}")
    if rejected is not None:
        store.submit(g_t, rejected)

    print()
    print("=== toxic_no_logprobs ===")
    g_n = build_honest_group(
        variant, rollout=rollout, group_id="g-live-tox-noprob",
        logprobs_override=[0.0] * n,
    )
    rejected = validate_group_against_lease(g_n, lease, manifest, now=now)
    print(f"  validators -> {rejected.rejection_reason.value if rejected else 'PASS'}")
    if rejected is not None:
        store.submit(g_n, rejected)

    print()
    print("=== TrainerFeedback round-trip on honest group ===")
    agg = FeedbackAggregator()
    feedback = TrainerFeedback(
        group_id="g-live-honest",
        worker_id=variant.worker_id,
        engine_name=variant.engine_name,
        engine_version="0.20.1",
        policy_version="v-live-001",
        num_policy_tokens=n,
        ess_observed=report.effective_sample_size,
        clipped_fraction_observed=report.clipped_fraction,
        accepted_by_trainer=(decision.recommended_action == "train"),
        current_policy_logprob_mean=sum(trainer_logprobs) / len(trainer_logprobs),
        current_policy_logprob_std=0.5,
    )
    agg.ingest(feedback)
    s = agg.stats_for_engine(variant.engine_name)
    print(f"  {variant.engine_name} engine: count={s.count}, mean_ess={s.mean_ess:.4f}, "
          f"acceptance_rate={s.acceptance_rate}")


def _run_fp8(rollout: dict, trainer: dict, variant: Variant, store: InMemoryLiveStore) -> None:
    """fp8 path: precision-mismatch reject + fp8-pinned honest case."""
    manifest = build_manifest(variant)
    lease = build_lease(variant)
    now = lease.issued_at + timedelta(seconds=30)
    trainer_logprobs = trainer["trainer_logprobs"]

    print("=== fp8 worker submitting under bf16-pinned manifest ===")
    g_p = build_honest_group(variant, rollout=rollout, group_id="g-live-fp8-precision")
    rejected = validate_group_against_lease(g_p, lease, manifest, now=now)
    print(f"  validators -> {rejected.rejection_reason.value if rejected else 'PASS'}")
    if rejected is not None:
        store.submit(g_p, rejected)

    print()
    print("=== fp8 worker on an fp8-pinned manifest (correct match) ===")
    fp8_lease = lease.model_copy(update={"required_precision_class": "fp8"})
    fp8_manifest = manifest.model_copy(update={"precision_class": "fp8"})
    g_h = build_honest_group(variant, rollout=rollout, group_id="g-live-fp8-honest")
    rejected = validate_group_against_lease(g_h, fp8_lease, fp8_manifest, now=now)
    assert rejected is None, f"unexpected rejection: {rejected}"
    print("  validators -> PASS (precision pin matches)")
    report = compute_budget_report(g_h, trainer_logprobs)
    decision = decide_group(report)
    print(f"  OPBC -> {decision.recommended_action} ({[r.value for r in decision.decision_reasons]})")
    print(f"  ESS={report.effective_sample_size:.4f}, max|log_ratio|={report.max_abs_log_ratio:.4f}")
    store.submit(g_h, decision)

    print()
    print("=== TrainerFeedback round-trip on fp8 honest group ===")
    agg = FeedbackAggregator()
    n = len(rollout["response_token_ids"])
    feedback = TrainerFeedback(
        group_id="g-live-fp8-honest",
        worker_id=variant.worker_id,
        engine_name=variant.engine_name,
        engine_version="0.20.1",
        policy_version="v-live-001",
        num_policy_tokens=n,
        ess_observed=report.effective_sample_size,
        clipped_fraction_observed=report.clipped_fraction,
        accepted_by_trainer=(decision.recommended_action in ("train", "train_with_correction", "replay")),
        current_policy_logprob_mean=sum(trainer_logprobs) / len(trainer_logprobs),
        current_policy_logprob_std=0.5,
    )
    agg.ingest(feedback)
    s = agg.stats_for_engine(variant.engine_name)
    print(f"  {variant.engine_name} engine: count={s.count}, mean_ess={s.mean_ess:.4f}, "
          f"acceptance_rate={s.acceptance_rate}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS.keys()), default="bf16")
    parser.add_argument("--rollout-input", help="override rollout JSON path")
    parser.add_argument("--trainer-input", help="override trainer JSON path")
    args = parser.parse_args(argv)

    variant = VARIANTS[args.variant]
    rollout_path, trainer_path = _resolve_input_paths(variant)
    if args.rollout_input:
        rollout_path = Path(args.rollout_input)
    if args.trainer_input:
        trainer_path = Path(args.trainer_input)
    rollout = json.loads(rollout_path.read_text())
    trainer = json.loads(trainer_path.read_text())

    store = InMemoryLiveStore()
    if variant.label == "bf16":
        _run_bf16(rollout, trainer, variant, store)
    else:
        _run_fp8(rollout, trainer, variant, store)

    print()
    print("=== LiveStore state ===")
    counts = {s.value: c for s, c in store.count_by_state().items() if c}
    print(f"  {counts}")

    print()
    print(f"=== TrainerClient pull (FRESH tier, v-live-001, precision={variant.precision_class}) ===")
    client = TrainerClient(store)
    batch = client.fetch(
        policy_version="v-live-001",
        max_groups=10,
        precision_class=variant.precision_class if variant.label == "fp8" else None,
        replay_tier=ReplayTier.FRESH,
    )
    print(f"  served {batch.size} groups: {[r.group_id for r in batch.groups]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
