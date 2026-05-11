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

## In-flight (Megatron Qwen3-32B conversion — bind-mount retry running)

**Bug:** the first run's staging dir
`~/megatron_conversion/hf_model_qwen3_32b/` held host-absolute
symlinks into `~/hf-cache/hub/models--Qwen--Qwen3-32B/blobs/...` that
the container could not resolve, so `tokenizer.json` was dangling and
all 4 ranks died at tokenizer init.

**Fix (this iteration, branch `research/megatron-qwen3-32b-bindmount`):**

- New host-side launcher `scripts/live/megatron_qwen3_32b_launch.sh`
  bind-mounts `~/hf-cache/hub/models--Qwen--Qwen3-32B:/root/Qwen3-32B-repo:ro`
  (the full hub repo, read-only) instead of the broken staging dir.
  The snapshot's `../../blobs/...` *relative* symlinks resolve inside
  that bind-mount.
- Existing `scripts/live/megatron_qwen3_32b_runner.sh` now reads
  `HF_CHECKPOINT` from env (default: auto-detect the lone snapshot
  under `/root/Qwen3-32B-repo/snapshots/`) and asserts
  `config.json` is a real file before invoking `torchrun`.

**Run launched 2026-05-11 ~12:22 UTC.** `docker run` is up; first log
lines confirm the snapshot resolved to
`/root/Qwen3-32B-repo/snapshots/9216db.../`, MODEL_ARGS sourced from
the image's `qwen3-32B.sh`
(`--num-layers 64 --hidden-size 5120 --ffn-hidden-size 25600
--num-attention-heads 64 --num-query-groups 8 --kv-channels 128
--vocab-size 151936 --qk-layernorm ...`), and
`convert_hf_to_torch_dist.py` started without the tokenizer crash.
A background poll on the operator side watches for
`latest_checkpointed_iteration.txt` (success) or
`Traceback|FAILED|RuntimeError|OutOfMemory` (fail) in the convert log.

**To close out `model.megatron_convert_qwen3_32b` after success:**
```bash
mkdir -p .claude/evidence/model_megatron_convert_qwen3_32b
scp my-vllm-spot-instance:\
'~/megatron_conversion/qwen3_32b_torch_dist/latest_checkpointed_iteration.txt' \
  .claude/evidence/model_megatron_convert_qwen3_32b/
# then flip .claude/feature-results.json[model.megatron_convert_qwen3_32b]
# to passes: true with that path as evidence.
```

## Remaining false entries (in loop pick order)

1. `model.megatron_convert_qwen3_32b` — **spot**, retry (see above).
2. `vllm.router_trace_emit` — **spot**. Use vLLM's
   `enable_return_routed_experts=True` on `AsyncEngineArgs`. Read
   `CompletionOutput.routed_experts` (shape `[seq_position, moe_layer,
   top_k]`, length `num_prompt_tokens + num_generated_tokens - 1`,
   expert IDs in `[0, num_experts)`). Ref:
   https://docs.vllm.ai/en/latest/examples/rl/routed_experts_e2e/ .
   Caveat: returns selected expert IDs only, not gate logits —
   sufficient for `router_flip_rate`.
3. `dense.qwen3_32b.vllm_bf16_megatron` — **spot**, depends on (1).
4. `dense.qwen3_32b.vllm_fp8_fsdp` — **spot**, independent.
5. `dense.qwen3_32b.vllm_fp8_megatron` — **spot**, depends on (1).
6. `moe.qwen3_30b_a3b.vllm_bf16_megatron_router` — **spot**, depends
   on (2) + existing Megatron MoE checkpoint.
7. `moe.qwen3_30b_a3b.vllm_bf16_fsdp_router` — **spot**, depends on (2).
8. `moe.qwen3_30b_a3b.vllm_fp8_megatron_router` — **spot**, depends
   on (2) + (1).
9. `moe.qwen3_30b_a3b.vllm_fp8_fsdp_router` — **spot**, depends on (2).
10. `dashboard.high_quality_render` — **local**, final acceptance
    gate; depends on entries above producing real tile data.

## Test status
- pytest -q: **327 passed, 0 failed** (2026-05-11).
- ruff check .: clean.

## Reproducibility
Live runs use the env-driven runbook in `scripts/live/`
(`MODEL`, `VLLM_DTYPE`, `ROLLOUT_LABEL`, `TRAINER_REFERENCE`,
`MEGATRON_CKPT_DIR`, `DEVICE_*`); see `scripts/live/README.md`.
Historical completed-work narrative and empirical findings tables
live in git history (`git log`) and the rendered dashboards under
`/docs/` — they are not duplicated here.
