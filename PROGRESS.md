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

## Remaining false entries — concrete plan for end-to-end execution

The next `/autonomous-loop` invocation MUST run all of these without
halting. Each numbered block below is one loop iteration. Setup
commits (writing new scripts) and run commits (spot runs) are both
valid units of work; do not stop because an iteration "looks big".

### Iter A — write `scripts/live/run_megatron_reference.py` (setup)

- Runs inside the slimerl/slime:latest container (same image used by
  `megatron_qwen3_32b_launch.sh`).
- Reads `/tmp/rollout.json` (prompt_token_ids + response_token_ids
  from the vLLM rollout side); teacher-forces (prompt + response)
  through Megatron-LM at TP=4 with the torch-dist ckpt at
  `/root/Qwen3-32B_torch_dist/release/` (mount the host path
  `~/megatron_conversion/qwen3_32b_torch_dist`).
- Reuse MODEL_ARGS from the image's `/root/slime/scripts/models/qwen3-32B.sh`
  (dense, no MoE flags).
- Extract per-response-token logprobs from logits via gather on
  the response token IDs.
- Write `/tmp/trainer_megatron-bf16.json` in the same shape as
  `run_hf_reference.py`'s output (`{model, trainer_logprobs, engine,
  engine_fingerprint}`).
- Add a host-side launcher `scripts/live/megatron_reference_launch.sh`
  (mirrors `megatron_qwen3_32b_launch.sh`). Add a smoke test in
  `tests/test_megatron_reference_shape.py` that imports the script
  and asserts the helper functions exist and produce well-shaped
  stub output (no Megatron import; deferred inside the run path).
- This iteration is SETUP — no feature-results flip. Commit prefix
  `feat(megatron):`.

### Iter B — run `dense.qwen3_32b.vllm_bf16_megatron` (live)

- Spot: `MODEL=Qwen/Qwen3-32B ROLLOUT_LABEL=vllm-bf16 python ~/run_vllm_rollout.py`.
- Inside container: invoke `run_megatron_reference.py` via the new
  launcher.
- scp `/tmp/rollout_vllm-bf16.json` + `/tmp/trainer_megatron-bf16.json`
  back; `python scripts/live/build_dense_input.py --variant megatron-bf16`;
  ingest into `runs/live/dense/`. Flip
  `dense.qwen3_32b.vllm_bf16_megatron` to `true`. Commit prefix
  `live(megatron):`.

### Iter C — run `dense.qwen3_32b.vllm_fp8_megatron` (live)

- Same as Iter B but `MODEL=Qwen/Qwen3-32B-FP8 ROLLOUT_LABEL=vllm-fp8`.
- Reuse `--variant vllm-fp8 --trainer-label megatron-lm` (the
  trainer-label override we shipped this session). Flip
  `dense.qwen3_32b.vllm_fp8_megatron`. Commit prefix `live(fp8):`.

### Iter D — write `scripts/live/run_fsdp_moe_reference.py` (setup)

- Extension of `run_fsdp_reference.py`: same FSDP wrap pattern,
  but pass `output_router_logits=True` to the model and collect
  top-k expert IDs per (token, layer) for the MoE model. Write
  `/tmp/fsdp_router.json` in the same shape as
  `/tmp/vllm_router.json` (see `run_vllm_moe_rollout.py`).
- Smoke test: import as a module + assert shape helpers.
- SETUP iteration, no flip. Commit prefix `feat(fsdp):`.

### Iter E — run `moe.qwen3_30b_a3b.vllm_bf16_fsdp_router` (live)

- Spot: vLLM-bf16 MoE rollout via `run_vllm_moe_rollout.py`; FSDP
  reference via the new `run_fsdp_moe_reference.py`. Build router
  input (use `build_router_input.py` adapted for vLLM-trainer
  pairing, or write a sibling that maps the two trace files).
- Ingest into `runs/live/router/`. Flip. Commit prefix `live(moe-fsdp):`.

### Iter F — run `moe.qwen3_30b_a3b.vllm_fp8_fsdp_router` (live)

- Same as Iter E with `MODEL=Qwen/Qwen3-30B-A3B-FP8`. Flip. Commit
  prefix `live(moe-fp8-fsdp):`.

### Iter G — run `moe.qwen3_30b_a3b.vllm_bf16_megatron_router` (live)

- vLLM-bf16 MoE rollout via `run_vllm_moe_rollout.py`; Megatron
  router trace via the existing `run_moe_router.py` path (which
  already writes the trainer-side trace). Pair into a router-
  mismatch fixture and ingest. Flip. Commit prefix `live(moe-megatron):`.

### Iter H — run `moe.qwen3_30b_a3b.vllm_fp8_megatron_router` (live)

- Same as Iter G with the FP8 MoE rollout. Flip. Commit prefix
  `live(moe-fp8-megatron):`.

### Iter I — re-render + codex review `dashboard.high_quality_render`

- `python -m rollout_market.cli.dense_dashboard ...` then
  `python -m rollout_market.cli.router_dashboard ...` then
  `python scripts/live/publish_dashboards.py`.
- Inspect `docs/index.html`: 8 green tiles, no `mx-empty` / `mx-tbd`
  placeholders in the visible matrix area.
- `/codex-review --effort low` against the rendered HTML with the
  STEER-mandated prompt. Save verdict to
  `.claude/evidence/dashboard_high_quality_render/codex_verdict.txt`.
  Flip the last entry. Commit prefix `docs(dashboards):`.

When Iter I lands true, the autonomous loop's "all entries true"
stop clause fires; delete `STEER.md` on the way out per its own stop
clause.

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
