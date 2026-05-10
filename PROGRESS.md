# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-10

## Current phase
Phase 0-6 plan complete. **Live validation against real Qwen3-32B and
free-tier APIs done.** Remaining is operational follow-up.

## Completed
- Phase 0 scaffold and docs/configs/examples
- Phase 1.1: PolicyManifest + WorkerManifest. F401 cleanup.
- Phase 1.2: RolloutJob + RolloutLease.
- Phase 1.3: Typed RejectionReason + validators.py.
- Phase 2.1: Endpoint identity gap probe.
- Phase 2.2: Controlled dense mismatch lab.
- Phase 2.3: Small MoE router mismatch lab.
- Phase 3.1: LiveStore lifecycle.
- Phase 3.2: Trainer client API.
- Phase 3.3: Worker SDK + LocalWorkerBroker.
- Phase 4.1: Typed DecisionReason + replay path.
- Phase 4.2: Trainer feedback ingestion.
- Phase 5.1: Heartbeat / lease timeout / audit trail.
- Phase 5.2: k-of-n reload quorum.
- Phase 6: marketplace simulation, endpoint dashboard, dense dashboard,
  router dashboard.
- Live validation: free-tier endpoint probes (5 providers, 10 runs); real
  Qwen3-32B dense mismatch (vLLM 0.20.1 vs HF transformers 5.8.0, both
  bf16 on 4× L40S); full marketplace stack exercised end-to-end against
  the real rollout.

## Live findings (2026-05-10)

Endpoint probes:
  - urllib's default User-Agent is Cloudflare-blocked (error 1010) by
    Cerebras and Groq. Fixed in PR #20.
  - Groq llama-3.1-8b-instant returns 400 on logprobs=true with explicit
    "logprobs is not supported with this model".
  - NVIDIA NIM is heterogeneous: meta/llama-3.1-8b-instruct is plain
    OpenAI-shape; qwen/qwen3-next-80b-a3b-instruct exposes vLLM
    internals (prompt_token_ids, prompt_logprobs, kv_transfer_params).
  - Cerebras llama3.1-8b is the only free-tier endpoint that returns
    sampled_logprobs + top_logprobs + system_fingerprint together.
  - OpenRouter free tier (`:free` models) is upstream-rate-limited and
    not usable for rollout work.
  - Coverage matrix across 10 runs: 3/10 sampled_logprobs, 3/10
    top_logprobs, 1/10 token_ids, 1/10 seed_supported. The endpoint
    identity gap is empirically real.

Dense mismatch (Qwen3-32B, 128 tokens, vLLM bf16 vs HF bf16, same seed):
  - num_policy_tokens = 128
  - delta_logprob_mean = +0.0022
  - delta_logprob_abs_mean = 0.0121
  - sequence_log_ratio = +0.2797
  - ESS = 0.9991
  - second_moment = 1.0061
  - clipped_fraction = 0.0
  - veto_fraction = 0.0
  - max_abs_log_ratio = 0.1652  (worst single-token disagreement)
  - top_1pct_gradient_mass = 0.0092
  Verdict: vLLM bf16 vs HF bf16 on the same checkpoint is essentially
  on-policy — OPBC routes this to `train` with `within_budget`.

Marketplace stack on real data:
  - validate_group_against_lease: honest group PASS, toxic_tokenizer
    REJECT(tokenizer_mismatch), toxic_no_logprobs REJECT(missing_logprobs).
  - decide_group(real_report): `train` / `within_budget`.
  - InMemoryLiveStore: 1 accepted / 2 rejected, no duplicate group_id
    surprises.
  - TrainerClient(FRESH).fetch: serves only the accepted group.
  - FeedbackAggregator: per-engine (vllm) mean_ess=0.9991,
    acceptance_rate=1.0 on the real feedback record.

Reproducibility:
  scripts/live/{run_vllm_rollout.py,run_hf_reference.py,
  build_dense_input.py,marketplace_real.py}, plus scripts/live/README.md
  with the runbook.

## Next tasks (operational, not plan)
- Add a UUID suffix to runs/<UTC-ts>/ to stop concurrent CLI invocations
  in the same wall-clock second from colliding.
- Pre-cache an OpenCode base URL or remove the env var from the firewall
  list — we have the key but no documented endpoint.
- Try Qwen3-30B-A3B (MoE) on the spot instance with output_router_logits
  to populate the router_mismatch dashboard with real data.
- Cross-engine comparison: Qwen3-32B bf16 vs Qwen3-32B FP8 (both cached
  on the spot) to surface real precision-class drift.

## Known issues
- src/rollout_market/dispatcher.py and opbc.py fail `ruff format --check`
  (pre-existing whitespace).
- runs/<ts>/ collision when two CLI invocations finish in the same UTC
  second; workaround is `--out-root runs/live/<provider>` per call.

## Test status
- Last run: 2026-05-10, 232 passed, 0 failed (pytest -q)
- ruff check .: clean
- Real-data round-trip: scripts/live/marketplace_real.py exits 0
