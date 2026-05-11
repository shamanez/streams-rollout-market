# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-11

## Current phase
**Phase 0-6 plan COMPLETE.** All originally-planned operational
follow-ups (sglang second rollout engine, MoE router live run, runs
UUID suffix) have shipped. The repo is in evidence-extension mode:
the open work is research-grade follow-ups in `docs/future_research.md`,
not new plan items.

## Completed (Phase 0-6 plan)
- Phase 0: scaffold, docs, configs, examples.
- Phase 1.1: PolicyManifest + WorkerManifest.
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
- Phase 6: marketplace simulation + 4 dashboards (endpoint, dense,
  router, agent) with inline glossaries and modern charts.

## Completed (operational follow-ups, originally listed as "next")
- STEER `dashboard.glossary_verl_rewrite`: rewrote
  `src/rollout_market/observatory/_glossary.py` to a frozen
  `GlossaryEntry` dataclass carrying four required fields (what,
  aggregation, cap, drop_behavior) per verl's train-inference
  correction module language. Cap fields name the actual constants —
  `BudgetPolicy.clamp = 20.0 nats`, `BudgetPolicy.veto_abs_log_ratio =
  30.0 nats`, `BudgetPolicy.max_clipped_fraction = 0.10` — and the
  drop behavior uses the verl vocabulary (clipped in place vs entire
  group quarantined; "no drop — observability only" for capability
  metrics). Renderers updated to surface a `<dl>` of (per / cap / on
  cap) under each entry. `tests/test_glossary.py` (7 cases) asserts
  the four-field shape, the 15 STEER-required terms, constant
  references, and verl-style drop vocabulary. 287 passed.
- STEER `dashboard.engine_filter`: added a render-time split on dense,
  router, and agent dashboards. The headline (`data-section=
  "headline-engines"`) shows only vLLM rollouts paired with FSDP or
  Megatron trainer-refs; the collapsible `<details data-section=
  "all-engines">` appendix carries sglang and HF-as-engine. Ingestion
  is unchanged — `as_dict()` and the JSON sidecar still contain every
  row. New `src/rollout_market/observatory/_engine_filter.py` carries
  the classifier; new `tests/test_engine_filter.py` (5 cases) asserts
  the split. Dense headline populated with 8 real `vllm→fsdp` runs;
  router/agent headlines correctly empty until FSDP/Megatron data is
  collected for them. 280 passed.
- STEER `dashboard.endpoint_retired`: removed the endpoint card from
  the public `/docs/index.html` (publish_dashboards.py) and the runtime
  live index (serve_dashboards.py); deleted `docs/endpoint_dashboard.{html,json}`
  from the published surface; `src/rollout_market/observatory/endpoint_dashboard.py`
  and its tests retained for internal use. Added `tests/test_publish_dashboards.py`
  asserting CARDS / _HEADLINES / rendered headline all exclude endpoint.
  275 passed.
- sglang as a second rollout engine on the spot instance
  (PR #23, #25): four-cell dense matrix vLLM/sglang × bf16/fp8.
- MoE router live run with Qwen3-30B-A3B FP8 vs bf16 (PR #26).
- `runs/<UTC-ts>-<uuid4>` suffix to stop second-of collisions
  (PR #24, see `src/rollout_market/cli/_runs.py`).
- Agent trajectory matrix on the four-engine cell (PR #28).
- FSDP as a third trainer-side reference (PR #33).
- One-shot dashboard host on the spot instance (PR #34).
- Dashboards consolidated under `/docs` with research-question
  framing (PR #35).
- `publish_dashboards.py` snapshot for the public dashboards repo
  (PR #30).

## Live findings (carried forward from 2026-05-10)

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

### Dense mismatch (Qwen3-32B, 8 prompts × 128 tokens, seed=1234)

| rollout engine | precision | mean ESS | mean \|Δlogp\| | worst max\|log_ratio\| | mean seq_log_ratio |
|---|---|---:|---:|---:|---:|
| vLLM 0.20.1 | bf16 | **0.9987** | 0.014 | 0.362 | +0.13 |
| vLLM 0.20.1 | fp8 | 0.9920 | 0.033 | 0.756 | -0.34 |
| sglang 0.5.11 | bf16 | 0.9845 | 0.066 | 0.711 | -4.67 |
| sglang 0.5.11 | fp8 | **0.9733** | 0.076 | **1.140** | -4.81 |

Trainer reference: HF transformers 5.8.0 bf16 with `sdpa` attention.
`clipped_fraction = 0` and `veto_fraction = 0` across all 32 runs, so
OPBC routes every cell to `train` / `within_budget`. Engine choice
dominates precision class; sglang carries a ~-4.7 nat systematic
sequence-log-ratio bias vLLM does not.

### Agent trajectory matrix (Qwen3-32B, 6 multi-step tool-using tasks)

| rollout engine | fully matched | first-div step | tool-jaccard | answer match rate |
|---|---:|---:|---:|---:|
| vLLM FP8 | 2/6 | 1.0 | 0.87 | **33%** |
| sglang bf16 | 1/6 | 1.8 | 0.83 | **17%** |
| sglang FP8 | 1/6 | 0.6 | **0.54** | **17%** |

Engine and precision change what an agent actually does end-to-end,
not just by how much its logprobs drift. vLLM FP8 flips the final
answer on 4/6 tasks; sglang variants flip 5/6.

### FSDP as a third trainer-side reference (forward-only)
FSDP produces logprobs bit-identical to HF transformers
`device_map="auto"` for forward-only inference. Expected — FSDP
shards parameters and all-gathers them in the forward pass, but the
matmuls and softmax run on the same kernels. The divergence would
appear in the backward pass (gradient all-reduce ordering), which is
out of scope for this repo. **Megatron-LM** is the next trainer
engine that *would* move forward-pass numbers (deferred — needs
HF→Megatron checkpoint conversion).

### MoE router (Qwen3-30B-A3B, 64 tokens × 48 layers × top_k=8)

| metric | value | reading |
|---|---:|---|
| router_flip_rate | 5.14% | 1 in 20 (token, layer) pairs flips top-1 expert |
| token_expert_disagreement_rate | **98.44%** | almost every token has ≥1 layer with set disagreement |
| worst layer flip rate | 12.50% | one layer flipped top-1 on 1 in 8 tokens |
| mean per-layer flip rate | 5.14% | no concentration in early/late layers |

FP8 barely shifts top-1 routing but reorders the lower-rank top-k
experts almost everywhere — the regime the router lab was designed
to surface.

### Marketplace stack on real data
- bf16 worker on bf16-pinned manifest → validators PASS, OPBC `train`.
- fp8 worker on bf16-pinned manifest → validators REJECT
  (`precision_mismatch`).
- fp8 worker on fp8-pinned manifest → validators PASS, OPBC `train`.
- Toxic tokenizer / missing-logprobs groups → validators REJECT with
  the matching RejectionReason.
- LiveStore lifecycle, TrainerClient(precision_class) filter, and
  FeedbackAggregator per-engine roll-up all work end-to-end on real
  data.

### Reproducibility
`scripts/live/` holds the env-driven runbook (`MODEL`, `VLLM_DTYPE`,
`ROLLOUT_LABEL`, `TRAINER_REFERENCE`); see `scripts/live/README.md`.

## Next work (research follow-ups, not plan tasks)

The Phase 0-6 plan is done. Open empirical questions are tracked in
`docs/future_research.md`. The high-leverage items, ranked roughly by
effort-to-evidence ratio:

1. **n ≥ 30 statistical pass** over the existing dense + agent matrix
   — single biggest credibility win, ~3h of spot work.
2. **Sequence-length scaling** (128 / 512 / 2048 token responses) —
   shows where ESS clipping kicks in.
3. **Megatron-LM as the fourth trainer-side reference** — the only
   engine that would actually move forward-pass numbers vs HF/FSDP.
   ~1 day; blocked on HF→Megatron checkpoint conversion.
4. **MoE router at multi-prompt / longer horizon** — extend the 64-
   token n=1 result.
5. **Cross-hardware** (H100/H200 FP8 native, MI300X) — flips or
   strengthens the FP8 finding.
6. **Real (vs stub) tools in the agent matrix** — isolates engine
   drift from tool-noise.
7. **GRPO/PPO single-step downstream** — show engine drift becomes a
   different parameter delta. Out of project scope (firewall), but
   *consumable from* this repo's contracts.

## Known issues
- `src/rollout_market/dispatcher.py` still fails `ruff format --check`
  (pre-existing whitespace; `opbc.py` was cleaned up, `dispatcher.py`
  remains).

## Test status
- pytest -q: **287 passed, 0 failed** (2026-05-11).
- ruff check .: clean.
- Real-data round-trips: `scripts/live/marketplace_real{,_fp8}.py`
  exit 0.
