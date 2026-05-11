# PROGRESS.md — Agent Handoff State

## Last updated
2026-05-11

## Current phase
**STEER is armed — drive the dashboard to 8 green tiles.** STEER.md
re-armed `.claude/feature-results.json` with 11 acceptance criteria.
13 entries are `passes: true` (incl. `dashboard.matrix_per_model`,
shipped this iteration); 10 remain `false`. The autonomous loop
(`/autonomous-loop`) picks the first false entry, runs
`/implement-feature` against its `directive` field, captures evidence
under `.claude/evidence/`, and continues until every entry is true
(then deletes STEER.md per its stop clause).

The load-bearing claim being filled in: for MoE models, the rollout
engine's per-(token, layer) expert selection diverges from the
trainer's. If a worker's vLLM activates experts `{A, B, C}` for token
X but the trainer's gradient flows to `{A, B, D}`, the update lands
on the wrong parameters. Measuring `router_flip_rate` between vLLM
and Megatron/FSDP is the central MoE marketplace metric.

## Resume in a new session
**Single command:** `/autonomous-loop`

The loop reads `STEER.md` + `.claude/feature-results.json` + this
file, picks the first false entry, and runs end-to-end without
further input. Prereqs the loop checks itself: `AGENT_STOP` absent,
`STEER.md` present, weekly usage under 95%.

Operator overrides:
- `touch AGENT_STOP` — halt after the current iteration.
- `bash .claude/scripts/steer.sh "<note>"` — redirect mid-run.
- `rm AGENT_STOP` — resume.

## Shipped this session (2026-05-11)

- **`model.megatron_convert_qwen3_32b` — DONE.** Bind-mount fix on
  branch `research/megatron-qwen3-32b-bindmount` got the slimerl
  container past tokenizer init and produced the dist-ckpt in 10
  minutes. Evidence at
  `.claude/evidence/model_megatron_convert_qwen3_32b/`
  (`latest_checkpointed_iteration.txt = release` + `EVIDENCE.md`
  + `spot_state.txt`). The 62 GB checkpoint lives at
  `~/megatron_conversion/qwen3_32b_torch_dist/release/` on the spot
  (4 `__r_s.distcp` shards + `.metadata`).
- **`vllm.router_trace_emit` — DONE** on sibling branch
  `research/vllm-router-trace-emit`. Adds
  `scripts/live/run_vllm_moe_rollout.py` (uses
  `AsyncEngineArgs(enable_return_routed_experts=True)`) and
  `tests/test_vllm_router_trace_shape.py` (4 tests, no vLLM import
  required — script loaded as a module so the deferred imports stay
  uninstantiated). Evidence at
  `.claude/evidence/vllm_router_trace_emit/EVIDENCE.md`.

## Remaining false entries (in loop pick order)

1. `dense.qwen3_32b.vllm_bf16_megatron` — **spot**, unblocked now
   that `qwen3_32b_torch_dist/release/` exists.
2. `dense.qwen3_32b.vllm_fp8_fsdp` — **spot**, independent.
3. `dense.qwen3_32b.vllm_fp8_megatron` — **spot**, unblocked.
4. `moe.qwen3_30b_a3b.vllm_bf16_megatron_router` — **spot**, depends
   on existing Megatron MoE checkpoint + the new
   `run_vllm_moe_rollout.py` (now in repo).
5. `moe.qwen3_30b_a3b.vllm_bf16_fsdp_router` — **spot**, needs a
   `run_fsdp_moe_reference.py` (not in repo yet) +
   `run_vllm_moe_rollout.py`.
6. `moe.qwen3_30b_a3b.vllm_fp8_megatron_router` — **spot**, MoE
   Megatron-ckpt + `run_vllm_moe_rollout.py`.
7. `moe.qwen3_30b_a3b.vllm_fp8_fsdp_router` — **spot**, needs
   `run_fsdp_moe_reference.py`.
8. `dashboard.high_quality_render` — **local**, final acceptance
   gate; depends on entries above producing real tile data.

## Test status
- pytest -q on `research/megatron-qwen3-32b-bindmount`: **327
  passed, 0 failed** (2026-05-11).
- pytest -q on `research/vllm-router-trace-emit`: **331 passed**
  (+4 from `test_vllm_router_trace_shape.py`).
- ruff check .: clean.

## Reproducibility
Live runs use the env-driven runbook in `scripts/live/`
(`MODEL`, `VLLM_DTYPE`, `ROLLOUT_LABEL`, `TRAINER_REFERENCE`,
`MEGATRON_CKPT_DIR`, `DEVICE_*`); see `scripts/live/README.md`.
Historical completed-work narrative and empirical findings tables
live in git history (`git log`) and the rendered dashboards under
`/docs/` — they are not duplicated here.
