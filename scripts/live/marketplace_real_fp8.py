"""Validate the marketplace stack against the real FP8 Qwen3-32B rollout.

The FP8 rollout has measurably more drift than bf16. We expect OPBC to
still route this to `train` (ESS=0.995 is far above the 0.30 floor) but
the worst-token delta has doubled, and the dispatcher's record of this
run lets a future audit notice the precision-class effect.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
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

ROLLOUT = json.loads(Path("/tmp/rollout_fp8.json").read_text())
TRAINER = json.loads(Path("/tmp/trainer_fp8.json").read_text())
prompt_ids = ROLLOUT["prompt_token_ids"]
response_ids = ROLLOUT["response_token_ids"]
N = len(response_ids)

# Key contract decision: precision_class on the manifest stays "bf16" — the
# trainer pin — but the rollout group declares precision_class "fp8". This
# is the realistic case where a worker is quantized to save GPU memory and
# must still satisfy a bf16-pinned policy (it must NOT). validators should
# refuse it.
manifest = PolicyManifest(
    policy_version="v-live-001",
    checkpoint_digest="sha256:hf-cache-Qwen3-32B-bf16-snapshot",
    tokenizer_hash="tok-qwen3",
    model_config_hash="cfg-qwen3-32b",
    precision_class="bf16",
)
lease = RolloutLease(
    lease_id="lease-live-fp8-001",
    job_id="job-live-fp8-001",
    assignment_id="assign-live-fp8-001",
    worker_id="vllm-fp8-l40s-spot",
    policy_version="v-live-001",
    required_precision_class="bf16",
    max_context_tokens=4096,
    group_size=1,
    idempotency_key="idem-live-fp8-001",
    issued_at=datetime(2026, 5, 10, 7, 40, tzinfo=timezone.utc),
    expires_at=datetime(2026, 5, 10, 8, 10, tzinfo=timezone.utc),
)


def _group(group_id: str, *, precision: str) -> SampleGroup:
    return SampleGroup(
        group_id=group_id,
        task_id="rl-explainer",
        prompt_id="p-rl-explainer-001",
        worker_id="vllm-fp8-l40s-spot",
        assignment_id="assign-live-fp8-001",
        policy_version="v-live-001",
        checkpoint_digest="sha256:hf-cache-Qwen3-32B-bf16-snapshot",
        tokenizer_hash="tok-qwen3",
        sampling_config_hash="cfg-temp07-top0.95-seed1234",
        engine_name="vllm-fp8",
        engine_version="0.20.1",
        device_type="L40S",
        precision_class=precision,
        prompt_token_ids=prompt_ids,
        trajectories=[
            Trajectory(
                response_token_ids=response_ids,
                rollout_logprobs=ROLLOUT["rollout_logprobs"],
                action_mask=[1] * N,
                tool_output_mask=[0] * N,
                reward=1.0,
            )
        ],
        policy_lag_steps=0,
    )


def main() -> None:
    store = InMemoryLiveStore()
    now = datetime(2026, 5, 10, 7, 41, tzinfo=timezone.utc)

    print("=== fp8 worker submitting under bf16-pinned manifest ===")
    g_p = _group("g-live-fp8-precision", precision="fp8")
    rejected = validate_group_against_lease(g_p, lease, manifest, now=now)
    print(f"  validators -> {rejected.rejection_reason.value if rejected else 'PASS'}")
    if rejected is not None:
        store.submit(g_p, rejected)

    print()
    print("=== fp8 worker on an fp8-pinned manifest (correct match) ===")
    fp8_lease = lease.model_copy(update={"required_precision_class": "fp8"})
    fp8_manifest = manifest.model_copy(update={"precision_class": "fp8"})
    g_h = _group("g-live-fp8-honest", precision="fp8")
    rejected = validate_group_against_lease(g_h, fp8_lease, fp8_manifest, now=now)
    assert rejected is None, f"unexpected rejection: {rejected}"
    print("  validators -> PASS (precision pin matches)")
    report = compute_budget_report(g_h, TRAINER["trainer_logprobs"])
    decision = decide_group(report)
    print(f"  OPBC -> {decision.recommended_action} ({[r.value for r in decision.decision_reasons]})")
    print(f"  ESS={report.effective_sample_size:.4f}, max|log_ratio|={report.max_abs_log_ratio:.4f}, |delta|_mean={report.clipped_fraction:.4f}")
    store.submit(g_h, decision)

    print()
    print("=== LiveStore state ===")
    counts = {s.value: c for s, c in store.count_by_state().items() if c}
    print(f"  {counts}")

    print()
    print("=== TrainerClient pull (FRESH tier, v-live-001) ===")
    client = TrainerClient(store)
    batch = client.fetch(
        policy_version="v-live-001",
        max_groups=10,
        precision_class="fp8",
        replay_tier=ReplayTier.FRESH,
    )
    print(f"  served {batch.size} groups: {[r.group_id for r in batch.groups]}")

    print()
    print("=== Aggregated feedback ===")
    agg = FeedbackAggregator()
    agg.ingest(
        TrainerFeedback(
            group_id="g-live-fp8-honest",
            worker_id="vllm-fp8-l40s-spot",
            engine_name="vllm-fp8",
            engine_version="0.20.1",
            policy_version="v-live-001",
            num_policy_tokens=N,
            ess_observed=report.effective_sample_size,
            clipped_fraction_observed=report.clipped_fraction,
            accepted_by_trainer=(decision.recommended_action in ("train", "train_with_correction", "replay")),
            current_policy_logprob_mean=sum(TRAINER["trainer_logprobs"]) / len(TRAINER["trainer_logprobs"]),
            current_policy_logprob_std=0.5,
        )
    )
    s = agg.stats_for_engine("vllm-fp8")
    print(f"  vllm-fp8 engine: count={s.count}, mean_ess={s.mean_ess:.4f}, "
          f"acceptance_rate={s.acceptance_rate}")


if __name__ == "__main__":
    main()
