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

## In-flight (Megatron Qwen3-32B conversion — FAILED, needs retry)

**Setup committed** (`fe550be feat(megatron): runner for Qwen3-32B …`):
`scripts/live/megatron_qwen3_32b_runner.sh` sources the
`qwen3-32B.sh` MODEL_ARGS already shipping inside the
`slimerl/slime:latest` image. Copied to spot at
`~/megatron_conversion/runner_qwen3_32b.sh`.

**Run launched 2026-05-11 12:06 UTC** on `my-vllm-spot-instance`
under a backgrounded `docker run`. Output log:
`~/megatron_conversion/logs/qwen3_32b_convert.out`. HF model staged
at `~/megatron_conversion/hf_model_qwen3_32b/` (symlinks into
`~/hf-cache/hub/models--Qwen--Qwen3-32B/snapshots/9216db.../`).
Target torch-dist checkpoint under
`~/megatron_conversion/qwen3_32b_torch_dist/`.

**Status: failed at tokenizer init.** All 4 ranks raised
`ValueError: Couldn't instantiate the backend tokenizer ... You need
to have sentencepiece or tiktoken installed to convert a slow
tokenizer to a fast one.` from
`/root/slime/slime/backends/megatron_utils/initialize.py:77`. The
target dir is empty (no atomic save happened); safe to retry.

Most likely cause: the symlinks staged into
`~/megatron_conversion/hf_model_qwen3_32b/` point at host paths
(`/home/ubuntu/hf-cache/...`) that aren't visible inside the
container, so the readable `tokenizer.json` is dangling. The next
iteration should try, in order:

1. **Bind-mount the hf-cache snapshot directly** (cleanest):
   ```bash
   SNAP=$(ssh my-vllm-spot-instance \
     'ls -d ~/hf-cache/hub/models--Qwen--Qwen3-32B/snapshots/*/' | head -1)
   # then in the docker run, replace
   #   -v $HOME/megatron_conversion/hf_model_qwen3_32b:/root/Qwen3-32B
   # with
   #   -v ${SNAP%/}:/root/Qwen3-32B:ro
   ```
2. **Copy (not symlink) the tokenizer files** into the staging dir
   (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`,
   `merges.txt`) so they're real files inside the bind-mount.
3. **Install sentencepiece inside the container** before running:
   `docker run ... slimerl/slime:latest bash -lc "pip install -q
   sentencepiece && /root/runner_qwen3_32b.sh"`.

**To close out `model.megatron_convert_qwen3_32b` after a successful
retry:**
```bash
ssh my-vllm-spot-instance \
  'ls ~/megatron_conversion/qwen3_32b_torch_dist/latest_checkpointed_iteration.txt && \
   tail -3 ~/megatron_conversion/logs/qwen3_32b_convert.out'

mkdir -p .claude/evidence/model_megatron_convert_qwen3_32b
scp my-vllm-spot-instance:\
'~/megatron_conversion/qwen3_32b_torch_dist/latest_checkpointed_iteration.txt' \
  .claude/evidence/model_megatron_convert_qwen3_32b/

# Then flip .claude/feature-results.json[model.megatron_convert_qwen3_32b]
# to passes: true with evidence pointing at the scp'd marker.
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
