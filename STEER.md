# STEER.md — Operator Redirect

Seeded at 2026-05-13T02:27:15Z from Plan Mode (this conversation).

Stop clause: when every entry in `.claude/feature-results.json` has
`passes:true`, delete this STEER.md as the final commit.

---

# Initiative: MoE rollouts × precision matrix (bf16 + fp8) end-to-end

## Why

The MoE router dashboard currently has 2 cells filled (FSDP-bf16,
Megatron-bf16), each with n=1 over a single bf16 rollout of the
`no-op-trivia` task. We want a **2-row × 2-column matrix** with
multiple trajectories per cell:

|              | FSDP-bf16 (trainer) | Megatron-bf16 (trainer) |
|--------------|--------------------:|------------------------:|
| **vLLM bf16 rollout** | populated (need more n) | populated (need more n) |
| **vLLM fp8 rollout**  | empty                  | empty                  |

Same model family (Qwen3-30B-A3B), MoE only — no dense. The point is
to see whether FP8 quantization at inference time bends the
`router_flip_rate` and `ESS` numbers vs the bf16 trainer.

## Global directives

* MoE only — leave Qwen3-32B-dense alone.
* Use existing pipeline scripts under `scripts/live/`. Do not add
  trainer code outside `tests/` or `examples/`.
* All artefacts under `runs/live/`. Aggregate via
  `python -m rollout_market.cli.router_dashboard` then republish
  with `python scripts/live/publish_dashboards.py`.
* Free-tier rule and spot-only rule still apply (no AWS infra
  changes; the one spot is the only host).
* Tag rollouts with engine label `hermes-qwen3-30b-a3b-{bf16,fp8}`
  and trainer engine label `{fsdp,megatron}-bf16-moe` so the
  dashboard matrix lines up.

## Acceptance gates

* `pytest -q && ruff check .` green at the end of every iteration.
* Final dashboard shows ≥4 cells (2 rollout-precisions × 2 trainers)
  with n ≥ 3 trajectories per cell.

## Iterations

### 1. spot_pull_qwen3_moe_fp8: download Qwen/Qwen3-30B-A3B-FP8 to the spot

Fetch the FP8 quantized MoE weights from HuggingFace into the spot's
`~/hf-cache/`. Use the existing huggingface-cli (or `from
huggingface_hub import snapshot_download`) in `~/rmenv-dev/`. Verify
the cache layout matches the bf16 model's:
`~/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/<sha>/`
with `config.json`, `tokenizer.json`, `*.safetensors`. Then create a
convenience symlink at
`~/checkpoint/qwen3-30b-a3b-fp8/hf` → the snapshot dir (matching
the bootstrap layout). Acceptance: `ls ~/checkpoint/qwen3-30b-a3b-
fp8/hf/config.json` exists and reports `"quantization_config":
{"quant_method": "fp8"...}`.

### 2. spot_smoke_test_moe_fp8_serve: vLLM-dev FP8 serve + routed_experts probe

Adapt `scripts/live/vllm_serve_moe_dev.sh` (or write a sibling
`vllm_serve_moe_fp8_dev.sh`) to launch Qwen3-30B-A3B-FP8 with the
patched dev wheel, `--enable-return-routed-experts`,
`--enable-expert-parallel`, `--no-async-scheduling`. After
`/v1/models` returns 200, call `/v1/chat/completions` directly with
a single user turn and verify the response carries
`choices[0].routed_experts` with shape `[gen_tokens, 48, 8]`.
Acceptance: routed_experts shape correct, GPU memory usage roughly
half of the bf16 case (FP8 is half the precision width).

### 3. spot_drive_bf16_task_batch: capture 4-6 bf16 MoE trajectories

Drive the bf16 MoE (the existing `vllm_serve_moe_dev.sh` path) via
`minimal_agent_driver.py` against a fixed task list — pick 4 tasks
from `scripts/live/agent_tasks_hermes.json` (e.g. `math-tokens-
per-gpu-day`, `code-fix-factorial`, `mixed-research-and-math`,
`plan-3step-experiment`) plus the 2 `agent_tasks_cwc_repo.json`
tasks = 6 tasks total. Run each one through the proxy on port 8001
so per-token logprobs + routed_experts land in
`/tmp/hermes_probes.jsonl`. Merge into `runs/live/agent/hermes/
qwen3-30b-a3b/bf16/` and build `/tmp/rollouts_bf16.json` for the
trainer side. Acceptance: 6 trajectory JSONs under
`runs/live/agent/hermes/qwen3-30b-a3b/bf16/` each with non-null
`response_routed_experts`.

### 4. spot_drive_fp8_task_batch: capture matching fp8 MoE trajectories

Repeat iter 3 against the FP8 serve from iter 2 over the *same* 6
tasks. Output trajectories under `runs/live/agent/hermes/qwen3-30b-
a3b/fp8/` (note the `/fp8/` subdir keeps the precision in the path).
Build `/tmp/rollouts_fp8.json`. Acceptance: 6 fp8 trajectory JSONs,
all with `response_routed_experts` populated.

### 5. spot_fsdp_teacher_force_both_precisions

Push both `/tmp/rollouts_bf16.json` and `/tmp/rollouts_fp8.json`
through `scripts/live/run_fsdp_moe_reference.py`. The trainer always
runs in **bf16** (FSDP weights are loaded from the bf16 HF dir; we
only vary the rollout side). Outputs:
`/tmp/trainers_fsdp_moe_bf16_rollouts.json` and
`/tmp/trainers_fsdp_moe_fp8_rollouts.json` plus their `_router.json`
siblings. Acceptance: each output has shape `[gen_tokens, 48, 8]`
matching the corresponding rollout window.

### 6. spot_megatron_teacher_force_both_precisions

Same as iter 5 but Megatron. Write `/tmp/rollout.json` for the
bf16 batch first, run `megatron_moe_reference_launch.sh`, capture
`/tmp/trainer_megatron-bf16-moe-bf16-rollouts.json` and
`/tmp/megatron_router-bf16-rollouts.json`. Repeat with the fp8
batch. Use the same patched launcher (already fixed to dereference
HF symlinks). Acceptance: both batches produce valid router traces.

### 7. laptop_pair_all_reports

Pull all four trainer outputs to the laptop. Run
`pair_hermes_moe_reports.py` four times with the matching
`--rollout-engine-label hermes-qwen3-30b-a3b-{bf16,fp8}` and
`--trainer-engine-label {fsdp,megatron}-bf16-moe`. This produces 4
× 6 = 24 `router_mismatch_report.json` files under
`runs/live/router/<rollout-vs-trainer>/<run>/`. Acceptance: 24
reports written, all with `router_flip_rate` and `token_expert_
disagreement_rate` populated.

### 8. laptop_aggregate_republish_2x2_matrix

Re-aggregate via `python -m rollout_market.cli.router_dashboard
--reports-glob 'runs/live/router/**/router_mismatch_report.json'
--out-dir runs/live/router_dashboard`. The aggregator already keys
on `(rollout_engine, trainer_engine)` pairs, so the 2×2 matrix
populates automatically. Republish via `publish_dashboards.py`.
Update `_select_hermes_router_cells` in publish_dashboards.py if a
new column header (`bf16 + fp8 · L40S`) is required. Acceptance:
`docs/index.html` shows a 2×2 router matrix with 4 traffic-light
tiles, each tile reporting `n ≥ 3 run(s)` and a flip-rate value;
`docs/router_dashboard.html` shows per-precision per-trainer
breakdown; tests + ruff green.
