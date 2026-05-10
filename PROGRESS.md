# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-10

## Current phase
Phase 0-6 plan complete. **Live validation across bf16 and FP8 done.**
Operational follow-ups remain (sglang as a second rollout engine, MoE
router live run, runs/<ts> UUID suffix).

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
- Live validation:
  - 5 free-tier endpoint probes (Cloudflare 1010 fix shipped in PR #20).
  - Real Qwen3-32B bf16 dense mismatch (vLLM 0.20.1 vs HF 5.8.0).
  - Real Qwen3-32B FP8 dense mismatch (vLLM-FP8 vs HF bf16) —
    cross-precision drift demonstrated end-to-end.
  - Marketplace stack validated against both runs (validators correctly
    refuse fp8 worker on bf16-pinned manifest with `precision_mismatch`;
    correctly accept fp8 worker on fp8-pinned manifest).

## Live findings (2026-05-10)

### Endpoint probes (10 runs, 5 providers)
  - urllib's default User-Agent is Cloudflare-blocked (error 1010) by
    Cerebras and Groq. Fixed in PR #20.
  - Groq llama-3.1-8b-instant returns 400 on logprobs=true with
    "logprobs is not supported with this model".
  - NVIDIA NIM is heterogeneous: `meta/llama-3.1-8b-instruct` is plain
    OpenAI-shape; `qwen/qwen3-next-80b-a3b-instruct` exposes vLLM
    internals (`prompt_token_ids`, `prompt_logprobs`, `kv_transfer_params`).
  - Cerebras llama3.1-8b is the only free-tier endpoint that returns
    sampled_logprobs + top_logprobs + system_fingerprint together.
  - OpenRouter `:free` tier is upstream-rate-limited.
  - Coverage: 3/10 sampled_logprobs, 3/10 top_logprobs, 1/10 token_ids,
    1/10 seed_supported.

### Dense mismatch (Qwen3-32B, 128 tokens, seed=1234, four cells)
| rollout engine | precision | ESS | mean \|Δlogp\| | max \|log_ratio\| | seq_log_ratio |
|---|---|---:|---:|---:|---:|
| vLLM 0.20.1 | bf16 | 0.9991 | 0.012 | 0.165 | +0.28 |
| vLLM 0.20.1 | fp8 | 0.9949 | 0.033 | 0.377 | -0.72 |
| sglang 0.5.11 | bf16 | 0.9898 | 0.061 | 0.550 | -5.40 |
| sglang 0.5.11 | fp8 | 0.9641 | 0.092 | 0.941 | -5.16 |

Trainer reference is HF transformers 5.8.0 bf16 with `sdpa` attention in all
four cells. clipped_fraction = 0, veto_fraction = 0 across the board, so all
four route OPBC -> `train` / `within_budget`.

Insights this surfaces:
- **Engine choice is at least as load-bearing as precision class.** sglang
  bf16 (ESS 0.99) drifts more from the HF reference than vLLM bf16 *or*
  vLLM FP8 do.
- **sglang has a systematic over-confidence** — the rollout side scores
  the response ~5 nats higher than HF over 128 tokens (negative
  sequence_log_ratio of ~-5). vLLM stays within ±1 nat.
- **FP8 adds drift on top of the engine baseline.** vLLM gains ~0.2 in
  worst max|log_ratio| moving bf16 -> fp8; sglang gains ~0.4. Quantization
  noise is engine-specific, not just a precision-class effect.

### Marketplace stack on real data
  - bf16 worker on bf16-pinned manifest -> validators PASS, OPBC `train`.
  - fp8 worker on bf16-pinned manifest -> validators REJECT
    (`precision_mismatch`).
  - fp8 worker on fp8-pinned manifest -> validators PASS, OPBC `train`.
  - Toxic tokenizer / missing-logprobs groups -> validators REJECT with
    the matching RejectionReason.
  - LiveStore lifecycle, TrainerClient(precision_class) filter, and
    FeedbackAggregator per-engine roll-up all work end-to-end on real
    data.

### Reproducibility
`scripts/live/{run_vllm_rollout.py,run_hf_reference.py,
build_dense_input.py,build_dense_input_fp8.py,
marketplace_real.py,marketplace_real_fp8.py}` plus
`scripts/live/README.md` runbook. Rollout/HF scripts are env-driven
(`MODEL`, `VLLM_DTYPE`, `ROLLOUT_LABEL`, `TRAINER_REFERENCE`) so
quantized variants reuse the same code path.

## Next tasks (operational)
- **sglang as a second rollout engine** for the same Qwen3 models on
  the spot instance — would populate the dense dashboard with a
  `sglang -> hf-transformers` engine pair and let us compare engines
  cleanly.
- **MoE router lab live run** with Qwen3-30B-A3B (or its FP8) using
  `output_router_logits=True`, plus the corresponding HF reference.
- **`runs/<UTC-ts>/` UUID suffix** so concurrent CLI invocations in
  the same wall-clock second don't collide.

## Known issues
- src/rollout_market/dispatcher.py and opbc.py still fail
  `ruff format --check` (pre-existing whitespace).

## Test status
- Last run: 2026-05-10, 232 passed, 0 failed (pytest -q)
- ruff check .: clean
- Real-data round-trips: scripts/live/marketplace_real{,_fp8}.py exit 0
