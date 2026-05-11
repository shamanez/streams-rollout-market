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

Loop branch `research/megatron-qwen3-32b-bindmount` carries the
session's work (sibling `research/vllm-router-trace-emit` was merged
in via `3a9b074`):

- **`model.megatron_convert_qwen3_32b` — DONE.** Bind-mount fix:
  launcher mounts the full hub repo (not the broken staging dir) so
  snapshot's `../../blobs/...` relative symlinks resolve inside the
  slimerl container. 10-min conversion → 62 GB dist-ckpt at
  `~/megatron_conversion/qwen3_32b_torch_dist/release/` on spot.
  Evidence: `.claude/evidence/model_megatron_convert_qwen3_32b/`.
- **`vllm.router_trace_emit` — DONE.** `scripts/live/run_vllm_moe_rollout.py`
  drives `AsyncEngineArgs(enable_return_routed_experts=True)` and
  writes `/tmp/vllm_router.json`. 4 shape-contract tests in
  `tests/test_vllm_router_trace_shape.py` (no vLLM import — script
  loaded as a module).
- **`dense.qwen3_32b.vllm_fp8_fsdp` — DONE.** Real spot run:
  vLLM-FP8 rollout (54s) + FSDP-bf16 teacher-force (284s).
  Headline: **ESS=0.995, clipped_fraction=0.0** — FP8 inference
  does NOT violate OPBC thresholds for dense Qwen3-32B against an
  FSDP-bf16 trainer. Code lever: new `--trainer-label
  {hf-transformers,fsdp,megatron-lm}` on
  `scripts/live/build_dense_input.py` so one rollout variant can
  pair with multiple trainer references without a variant
  explosion.

## Remaining false entries (in loop pick order)

1. `dense.qwen3_32b.vllm_bf16_megatron` — **spot**, depends on a
   new `scripts/live/run_megatron_reference.py` (does NOT exist —
   the slimerl MoE runbook noted this script "to follow in a
   separate PR"). The Megatron forward pass needs to run inside the
   slimerl container, load the torch-dist checkpoint at
   `/root/Qwen3-32B_torch_dist/release/`, teacher-force the same
   (prompt + response) the vLLM rollout produced, extract per-token
   logprobs, and write `/tmp/trainer_megatron-bf16.json`. The next
   loop iteration should *first* write this script (setup PR), then
   run it (run PR), then ingest (writeup PR).
2. `dense.qwen3_32b.vllm_fp8_megatron` — depends on (1).
3. `moe.qwen3_30b_a3b.vllm_bf16_megatron_router` — needs the existing
   MoE Megatron checkpoint + the new `run_vllm_moe_rollout.py`
   (already in repo). Also needs a `build_megatron_router_input.py`
   that pairs vLLM router trace vs Megatron-side router trace from
   the existing MoE script.
4. `moe.qwen3_30b_a3b.vllm_bf16_fsdp_router` — needs a new
   `scripts/live/run_fsdp_moe_reference.py` (extension of
   `run_fsdp_reference.py` with `output_router_logits=True`) plus
   `run_vllm_moe_rollout.py`.
5. `moe.qwen3_30b_a3b.vllm_fp8_megatron_router` — same as (3) but
   FP8 rollout.
6. `moe.qwen3_30b_a3b.vllm_fp8_fsdp_router` — same as (4) but
   FP8 rollout.
7. `dashboard.high_quality_render` — final gate; depends on entries
   above producing real tile data. Re-render + codex review.

## Test status
- pytest -q: **333 passed, 0 failed** (2026-05-11). +6 since last
  session: 4 from `test_vllm_router_trace_shape.py`, 2 from new
  `test_build_dense_input_script::test_trainer_label_*`.
- ruff check .: clean.
- ruff check .: clean.

## Reproducibility
Live runs use the env-driven runbook in `scripts/live/`
(`MODEL`, `VLLM_DTYPE`, `ROLLOUT_LABEL`, `TRAINER_REFERENCE`,
`MEGATRON_CKPT_DIR`, `DEVICE_*`); see `scripts/live/README.md`.
Historical completed-work narrative and empirical findings tables
live in git history (`git log`) and the rendered dashboards under
`/docs/` — they are not duplicated here.
