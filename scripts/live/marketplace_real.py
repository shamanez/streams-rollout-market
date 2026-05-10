"""Validate the full marketplace stack against the real Qwen3-32B rollout.

Builds three SampleGroups from the same rollout data:
  - honest: contract-clean, real vLLM logprobs, real HF teacher-forced logprobs
  - toxic_tokenizer: same data but tokenizer_hash deliberately wrong
  - toxic_no_logprobs: same data but rollout_logprobs zeroed

Pushes each through validate_group_against_lease + LiveStore.submit, then
runs the OPBC budget controller on the honest one. The full stack reads
real data without any fixture or synthetic substitution.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

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

ROLLOUT = json.loads(Path("/tmp/rollout.json").read_text())
TRAINER = json.loads(Path("/tmp/trainer.json").read_text())
prompt_ids = ROLLOUT["prompt_token_ids"]
response_ids = ROLLOUT["response_token_ids"]
rollout_logprobs = ROLLOUT["rollout_logprobs"]
trainer_logprobs = TRAINER["trainer_logprobs"]
N = len(response_ids)

manifest = PolicyManifest(
    policy_version="v-live-001",
    checkpoint_digest="sha256:hf-cache-Qwen3-32B-bf16-snapshot",
    tokenizer_hash="tok-qwen3",
    model_config_hash="cfg-qwen3-32b",
    precision_class="bf16",
)
issued = datetime(2026, 5, 10, 7, 30, tzinfo=timezone.utc)
lease = RolloutLease(
    lease_id="lease-live-001",
    job_id="job-live-001",
    assignment_id="assign-live-001",
    worker_id="vllm-l40s-spot",
    policy_version="v-live-001",
    required_precision_class="bf16",
    max_context_tokens=4096,
    group_size=1,
    idempotency_key="idem-live-001",
    issued_at=issued,
    expires_at=issued + timedelta(minutes=30),
)


def _group(group_id: str, *, tokenizer_hash: str = "tok-qwen3", logprobs=None) -> SampleGroup:
    return SampleGroup(
        group_id=group_id,
        task_id="rl-explainer",
        prompt_id="p-rl-explainer-001",
        worker_id="vllm-l40s-spot",
        assignment_id="assign-live-001",
        policy_version="v-live-001",
        checkpoint_digest="sha256:hf-cache-Qwen3-32B-bf16-snapshot",
        tokenizer_hash=tokenizer_hash,
        sampling_config_hash="cfg-temp07-top0.95-seed1234",
        engine_name="vllm",
        engine_version="0.20.1",
        device_type="L40S",
        precision_class="bf16",
        prompt_token_ids=prompt_ids,
        trajectories=[
            Trajectory(
                response_token_ids=response_ids,
                rollout_logprobs=logprobs if logprobs is not None else rollout_logprobs,
                action_mask=[1] * N,
                tool_output_mask=[0] * N,
                reward=1.0,
            )
        ],
        policy_lag_steps=0,
    )


def main() -> None:
    store = InMemoryLiveStore()

    print("=== honest (real Qwen3-32B vLLM rollout) ===")
    g_h = _group("g-live-honest")
    rejected = validate_group_against_lease(g_h, lease, manifest, now=issued + timedelta(seconds=30))
    assert rejected is None, f"unexpected rejection: {rejected}"
    print("  validators -> PASS (contract clean)")
    report = compute_budget_report(g_h, trainer_logprobs)
    decision = decide_group(report)
    print(f"  OPBC -> {decision.recommended_action} ({[r.value for r in decision.decision_reasons]})")
    print(f"  ESS={report.effective_sample_size:.4f}, |delta|_mean={report.max_abs_log_ratio:.4f}")
    store.submit(g_h, decision)

    print()
    print("=== toxic_tokenizer ===")
    g_t = _group("g-live-tox-tok", tokenizer_hash="tok-WRONG")
    rejected = validate_group_against_lease(g_t, lease, manifest, now=issued + timedelta(seconds=30))
    print(f"  validators -> {rejected.rejection_reason.value if rejected else 'PASS'}")
    if rejected is not None:
        store.submit(g_t, rejected)

    print()
    print("=== toxic_no_logprobs ===")
    g_n = _group("g-live-tox-noprob", logprobs=[0.0] * N)
    rejected = validate_group_against_lease(g_n, lease, manifest, now=issued + timedelta(seconds=30))
    print(f"  validators -> {rejected.rejection_reason.value if rejected else 'PASS'}")
    if rejected is not None:
        store.submit(g_n, rejected)

    print()
    print("=== LiveStore state ===")
    counts = {s.value: c for s, c in store.count_by_state().items() if c}
    print(f"  {counts}")

    print()
    print("=== TrainerClient pull (FRESH tier, v-live-001) ===")
    client = TrainerClient(store)
    batch = client.fetch(policy_version="v-live-001", max_groups=10, replay_tier=ReplayTier.FRESH)
    print(f"  served {batch.size} groups: {[r.group_id for r in batch.groups]}")

    print()
    print("=== TrainerFeedback round-trip on honest group ===")
    agg = FeedbackAggregator()
    feedback = TrainerFeedback(
        group_id="g-live-honest",
        worker_id="vllm-l40s-spot",
        engine_name="vllm",
        engine_version="0.20.1",
        policy_version="v-live-001",
        num_policy_tokens=N,
        ess_observed=report.effective_sample_size,
        clipped_fraction_observed=report.clipped_fraction,
        accepted_by_trainer=(decision.recommended_action == "train"),
        current_policy_logprob_mean=sum(trainer_logprobs) / len(trainer_logprobs),
        current_policy_logprob_std=0.5,
    )
    agg.ingest(feedback)
    s = agg.stats_for_engine("vllm")
    print(f"  vllm engine: count={s.count}, mean_ess={s.mean_ess:.4f}, acceptance_rate={s.acceptance_rate}")


if __name__ == "__main__":
    main()
