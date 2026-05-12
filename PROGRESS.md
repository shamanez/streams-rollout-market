# PROGRESS.md — Agent Handoff State

## Last updated
2026-05-12

## Current phase
**Hermes Agent matrix paused on Iter 3 (2/6 iters done, 1 blocker).**
STEER.md is re-armed for a six-iteration cycle that adds 4 new
traffic-light tiles below the existing 8 (real `hermes-agent`
multi-turn trajectories over Qwen3-32B + Qwen3-30B-A3B at
vLLM-bf16/fp8).

Iter 1 shipped: `scripts/live/hermes_agent_runner.py` + pure
`adapt_hermes_record` / `adapt_hermes_jsonl` adapters with 18 new
tests.

Iter 2 shipped: `scripts/live/agent_tasks_hermes.json` (12
multi-turn tasks per the approved plan: 1 no-op-trivia, 6 multi-tool
tasks, 1 four-tool long-horizon). Expanded `_SIM_FILES` in
`scripts/live/agent_runner.py` with `/tmp/fib.py` (off-by-one bug)
and `/tmp/clip.py` (correct clipping function for the
read-then-verify task). 11 new tests in
`test_agent_tasks_hermes_schema.py`. Total 376 passing (+29 since
session start).

**Iter 3 blocker** (recorded in
`scripts/live/hermes_integration_notes.md`): cloned NousResearch/
hermes-agent into /tmp and discovered the real batch trajectory
output is ShareGPT `from/value` JSONL with inline `<think>` /
`<tool_call>` / `<tool_response>` XML — NOT the OpenAI chat-
completions JSONL the Iter 1 adapter was built against. The Iter 1
adapter and its 18 tests describe the correct *target* schema
(AgentTrajectory) but the wrong *source* shape. Three credible
paths forward (A: pivot adapter to ShareGPT shape; B: relax STEER
to "hermes tool-call parser via our existing runner"; C: defer
Iter 3-6) — each represents a different scope contract that needs
operator input. Loop halts cleanly; operator can pick a path and
re-arm STEER.

The previous-cycle dashboard restructure is still live;
`/docs/index.html` continues to answer the load-bearing question:

  *Engine + precision + device DO change inference-time behaviour
  enough to matter for crowdsourced MoE rollouts.* fp8 inference
  adds 2-3% router_flip_rate over the bf16 baseline, regardless of
  whether the trainer reference is FSDP-bf16 or Megatron-bf16.

Branch hygiene (2026-05-12): all merged research/setup branches
deleted locally + on origin. Repo now carries `main` +
`research/megatron-and-engine-purge` (predecessor branch, kept for
historical reference — its two commits were superseded by the work
that landed on main).

## Resume in a new session
**Single command:** `/autonomous-loop`

With STEER.md gone, the loop will fall back to PROGRESS.md "Next
work" (see below) and `docs/future_research.md` for the next unit
of work.

Operator overrides:
- `touch AGENT_STOP` — halt after the current iteration.
- `bash .claude/scripts/steer.sh "<note>"` — re-arm STEER.md.
- `rm AGENT_STOP` — resume.

## How to view the dashboard
```bash
python scripts/live/publish_dashboards.py    # re-render docs/
python -m http.server -d docs 8000 &
open http://localhost:8000/
```
Full live-experiment runbook lives in `CLAUDE.md#how-to-run`.

## Shipped this session (2026-05-11)

Branch `research/megatron-qwen3-32b-bindmount` carries the full
9-iteration arc that drove the dashboard to 8 tiles.

### Iter A — `feat(megatron): run_megatron_reference.py`
- Dense Qwen3-32B Megatron-bf16 teacher-force inside slimerl/slime.
- Uses stock Megatron's `model_provider` + `gpt_builder` (skips
  slime's training wrappers). `wrap_with_ddp=False` to avoid the
  30 GiB DDP grad-buffer OOM on L40S.
- `gather_from_tensor_model_parallel_region(logits)` before
  log_softmax — output is TP-sharded across the vocab dim by default.
- `CUDA_DEVICE_MAX_CONNECTIONS=1` env var for TP > 1.
- Setup commit, no feature-results flip.

### Iter B — `live(megatron): dense.qwen3_32b.vllm_bf16_megatron`
- vLLM-bf16 rollout + Megatron-bf16 teacher-force.
- Headline: ESS=0.9994, clipped=0, veto=0, delta_logprob_abs=0.0101.

### Iter C — `live(fp8): dense.qwen3_32b.vllm_fp8_megatron`
- vLLM-fp8 rollout + Megatron-bf16 teacher-force.
- Headline: ESS=0.9948, clipped=0, veto=0, delta_logprob_abs=0.0364.
- Confirms fp8 inference does NOT break OPBC thresholds against a
  Megatron-bf16 trainer either.

### Iter D — `feat(fsdp): run_fsdp_moe_reference.py`
- FSDP MoE teacher-force with `output_router_logits=True`. Emits
  length-(P+R-1) traces matching vLLM's `routed_experts` shape so
  RouterTrace can pair them.

### Iter E — `live(moe-fsdp): moe.qwen3_30b_a3b.vllm_bf16_fsdp_router`
- New `build_router_input_pair.py` with 4 variants covering the
  (vllm-bf16, vllm-fp8) × (fsdp-bf16, megatron-bf16) MoE matrix.
- Headline: **router_flip_rate=4.3%** for vLLM-bf16 vs FSDP-bf16.

### Iter F — `live(moe-fp8-fsdp): moe.qwen3_30b_a3b.vllm_fp8_fsdp_router`
- vLLM-fp8 (TP=2 because the FP8 expert ffn_intermediate=768 yields
  192 per shard at TP=4, which fails vLLM's FP8 block_n=128).
- Headline: **router_flip_rate=7.1%** for vLLM-fp8 vs FSDP-bf16.

### Iter G — `live(moe-megatron): moe.qwen3_30b_a3b.vllm_bf16_megatron_router`
- New `run_megatron_moe_reference.py` + `megatron_moe_reference_launch.sh`
  (mirrored from the existing on-spot script + bind-mounts).
- Headline: **router_flip_rate=4.1%** for vLLM-bf16 vs Megatron-bf16.

### Iter H — `live(moe-fp8-megatron): moe.qwen3_30b_a3b.vllm_fp8_megatron_router`
- Closes the 4-cell MoE matrix.
- Headline: **router_flip_rate=6.3%** for vLLM-fp8 vs Megatron-bf16.

### Iter I — `docs(dashboards): dashboard.high_quality_render`
- Fixed `router_dashboard.py` to emit `precision_class` on each row
  (derived from rollout_engine name) so the matrix bins MoE reports
  into bf16 / fp8 columns instead of a phantom "—" column.
- All 8 matrix tiles populated, no `mx-empty` / `mx-tbd`
  placeholders in the visible area.
- Codex review verdict: **PASS**.

## 8-tile matrix state

| matrix                 | trainer  | bf16    | fp8     |
|------------------------|----------|---------|---------|
| Dense (Qwen3-32B) ESS  | FSDP     | 0.9987  | 0.9949  |
| Dense (Qwen3-32B) ESS  | Megatron | 0.9994  | 0.9948  |
| MoE (Qwen3-30B-A3B) fr | FSDP     | 4.3%    | 7.1%    |
| MoE (Qwen3-30B-A3B) fr | Megatron | 4.1%    | 6.3%    |

Pattern: dense ESS is engine-and-precision-robust (>0.99 everywhere).
MoE router_flip_rate is dominated by vLLM-vs-trainer kernel
differences (~4% at matched precision), with fp8 inference adding
2-3% more divergence regardless of trainer engine.

## Next work (post-STEER)

The dashboard is at 8 tiles, but each MoE cell is n=1 — one prompt,
one response. Open candidates for follow-up iterations:

1. **MoE matrix n=8.** Repeat each MoE cell with 8 different prompts
   so the (router_flip_rate, layer-distribution) numbers carry error
   bars. Reuse `scripts/live/prompts.json`. Mirror the dense matrix
   precedent (n=8 for the bf16-vs-FSDP corner).
2. **Layerwise heatmap.** The current router dashboard reports
   per-pair scalar flip rates; surface the per-layer flip array as
   a row-of-bars in each tile's drill-in. Worst layer is the most
   actionable signal for OPBC routing decisions.
3. **OPBC integration of router_flip_rate.** Add an OPBC reason
   `high_router_flip_rate` that quarantines MoE groups whose
   measured flip rate exceeds a configurable threshold. Currently
   the OPBC only inspects logprob-level mismatch.

## Test status
- pytest -q: **376 passed, 0 failed** (2026-05-12). +29 since
  last session: 18 from `test_hermes_agent_runner.py`, 11 from
  `test_agent_tasks_hermes_schema.py`.
- ruff check .: clean.
- `.claude/feature-results.json`: 23 prior entries pass; 6 new
  `hermes_agent.*` entries — 2 true (`runner_and_adapter`,
  `realistic_task_suite`), 4 false (3 live, 1 render).

## Reproducibility
Live runs use the env-driven runbook in `scripts/live/`:
- Dense: `MODEL`, `VLLM_DTYPE`, `ROLLOUT_LABEL`, `TRAINER_REFERENCE`.
- MoE rollout: `run_vllm_moe_rollout.py` (with
  `enable_return_routed_experts=True`).
- FSDP MoE reference: `run_fsdp_moe_reference.py` under torchrun.
- Megatron dense reference: `run_megatron_reference.py` +
  `megatron_reference_launch.sh` (slimerl/slime container,
  bind-mounted `~/megatron_conversion/qwen3_32b_torch_dist/`).
- Megatron MoE reference: `run_megatron_moe_reference.py` +
  `megatron_moe_reference_launch.sh` (same image, bind-mounted
  `~/megatron_conversion/megatron_ckpt/` + `~/megatron_conversion/hf_model/`).
Historical narrative + empirical findings live in `git log` and the
rendered dashboards under `/docs/`.
