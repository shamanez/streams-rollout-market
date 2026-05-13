# PROGRESS.md — Agent Handoff State

## Last updated
2026-05-13 — **cycle Remote-vLLM in progress**: 2 of 7 iterations
landed (operator kit + client-side capture driver). Operator paused
mid-cycle to resume in a new session. Cycle FP8 (the previous one)
remains complete and merged.

## Status — to resume

Next iteration: **iter 3 `tunnel_smoke_against_spot`** (see
STEER.md). To resume the cycle in a fresh session:

```bash
/autonomous-loop
```

The loop reads STEER.md + `.claude/feature-results.json` and picks
the first `passes:false` entry — which is iter 3.

### What iter 3 needs

* Spot powered on with vLLM-MoE bf16 serving on port 8000. Start
  it with:
  ```bash
  ssh my-vllm-spot-instance 'cd ~ && tmux new-session -d -s vllm \
      bash ~/streams-rollout-market/scripts/live/vllm_serve_moe_dev.sh'
  ```
* Laptop-side SSH local port-forward so `localhost:8001` reaches
  the spot's `:8000` (proves the over-HTTP capture path):
  ```bash
  ssh -fNL 8001:localhost:8000 my-vllm-spot-instance
  ```
* The new client-side driver (`scripts/live/remote_capture_driver.py`)
  pointed at `http://localhost:8001` with one task
  (`no-op-trivia`), then verify the sidecar JSONL line has
  `response_routed_experts` shape `[gen_tokens, 48, 8]` and
  `response_logprobs` length matching `response_token_ids`.

After iter 3, iters 4–7 are straightforward (drive 6-task batch,
teacher-force locally on spot, pair, dashboard).

## Cycle Remote-vLLM iterations (in order)

1. `bundle_operator_serve_kit` ✓ — operator quickstart + tarball
   build script under `scripts/live/operator_kit/`.
2. `client_side_capture_driver` ✓ — `scripts/live/remote_capture_driver.py`
   + 2 unit tests verifying the sidecar shape matches the proxy's
   `extract_probes` output exactly.
3. `tunnel_smoke_against_spot` ← **resume here**
4. `drive_6_task_batch_remote_style`
5. `teacher_force_remote_rollouts`
6. `pair_remote_results`
7. `dashboard_add_remote_row`

## Cycle FP8 final result (the matrix)

|              | vLLM bf16 rollout | vLLM fp8 rollout |
|--------------|------------------:|-----------------:|

## Cycle FP8 final result (the matrix)

|              | vLLM bf16 rollout | vLLM fp8 rollout |
|--------------|------------------:|-----------------:|
| FSDP-bf16    | 6.01% (n=19)      | 8.29% (n=16)     |
| Megatron-bf16| 5.72% (n=19)      | 8.34% (n=16)     |

(Color band: all amber 5–15%. Per-cell n is the number of
probe-bearing assistant turns across 6 Hermes-Agent tasks.)

Two findings:
- **FP8 quantization adds ~2.3-2.6 pp to router_flip_rate** vs bf16
  rollouts. The trainer is always bf16, so this is the marketplace
  cost of accepting FP8-quantized rollouts.
- **The trainer engine choice barely matters** (FSDP and Megatron
  agree within 0.3 pp on every cell — strong consistency signal
  for the metric itself).

## Cycle FP8 — iterations (in order completed)

1. `spot_pull_qwen3_moe_fp8` — Qwen3-30B-A3B-FP8 cached (~31 GB).
2. `spot_smoke_test_moe_fp8_serve` — FP8 serve via patched HTTP
   shim emits `routed_experts` shape `[gen_tokens, 48, 8]`.
3. `spot_drive_fp8_task_batch` — 6 FP8 trajectories captured
   (16 probe-bearing assistant turns).
4. `spot_drive_bf16_task_batch` — 6 bf16 trajectories captured
   (19 probe-bearing assistant turns incl. preserved no-op-trivia).
5. `spot_fsdp_teacher_force_both_precisions` — FSDP MoE × bf16
   rollouts (19 payloads, 290s) and × fp8 rollouts (16 payloads,
   229s); MODEL env override forces bf16 trainer regardless of
   rollout precision.
6. `spot_megatron_teacher_force_both_precisions` — Megatron MoE ×
   both precisions in 5min12s wall after the launcher's five
   patches (see iter06 evidence).
7. `laptop_pair_all_reports` — 70 router_mismatch_reports across
   four cells via `pair_hermes_moe_reports.py` × 4.
8. `laptop_aggregate_republish_2x2_matrix` — extended
   `_render_hermes_router_matrix` to take `rollout_engines` as a
   tuple (one column per rollout precision); 4 traffic-light tiles
   on `docs/index.html`.

## Cycle FP8 — launcher / pipeline fixes worth keeping

- `scripts/live/megatron_moe_reference_launch.sh` gained:
  * auto-detection of single-dict vs list rollouts file
  * `HF_FLAT_REUSE` env var to skip the ~60 GB `cp -rL` on
    back-to-back launches
  * `--network=host` (defensive against the broken docker0
    bridge teardown loop)
  * dropped `--ipc=host` (was causing containers to stay in
    Created state indefinitely)
  * `OUT_SUFFIX` env var so parallel launches don't trample each
    other's outputs.
- `scripts/live/vllm_serve_moe_fp8_dev.sh` — thin sibling of the
  bf16 launcher, overrides MODEL + LOG.
- `scripts/live/publish_dashboards._render_hermes_router_matrix`
  now accepts a tuple of `rollout_engines` for multi-column
  matrices.

## How to start a new initiative

(Unchanged from above — Plan Mode → `/seed-from-plan` →
`/autonomous-loop`.)

## Reproduction (durable)

`CLAUDE.md` "End-to-end reproduction". Cycle-FP8 specifically:
swap the bf16 serve for `vllm_serve_moe_fp8_dev.sh` in step 1 and
run the laptop pipeline with `--precision-class fp8`.

## Post-cycle operator-directed work (2026-05-13)
- **Megatron MoE router tile filled.** HF→torch-dist conversion ran
  on the spot (~62 GB distcp shards under `~/checkpoint/qwen3-30b-
  a3b/megatron/release/`). First teacher-force attempt failed on
  tokenizer instantiation inside the slime container because the HF
  snapshot dir is full of relative symlinks pointing *outside* the
  bind-mount; fixed `scripts/live/megatron_moe_reference_launch.sh`
  to dereference with `cp -rL` to a temp dir before mounting. Run
  completed in 55.3s. Paired vs the existing rollout: `router_flip_
  rate = 3.03%` (green, worst layer 9.1%). Both MoE × FSDP (3.50%)
  and MoE × Megatron (3.03%) tiles are now live. Merged as
  `582aacf`.
- **cwc-long-running-agents Q&A trajectories.** Drove the patched
  MoE dev wheel via `minimal_agent_driver` against two new tasks in
  `scripts/live/agent_tasks_cwc_repo.json`. The agent used parallel
  read_file in one task and sequential read_file in the other; both
  trajectories captured full per-token logprobs + per-(token, MoE
  layer, top-8) `response_routed_experts`. Fixed an `agent_dashboard
  ._render_step` bug (`ToolCallRecord.arguments` → `arguments_json`)
  that had been silently producing an empty viewer on prior
  publishes. Viewer now shows 4 trajectories. Merged as `11fd7dc`.

## Completed this cycle
- `spot_build_vllm_dev_routed_experts` (iter 01) — wheel + four-file
  HTTP-shim patch on the spot's `~/rmenv-dev/`.
- `spot_serve_moe_router_replay` (iter 02) —
  `scripts/live/vllm_serve_moe_dev.sh`. Smoke: `routed_experts` shaped
  `[20, 48, 8]` on Qwen3-30B-A3B.
- `spot_start_capture_proxy` (iter 03) — proxy on 8001 with engine-
  buffer trim so `response_routed_experts` aligns 1:1 with
  `response_token_ids`.
- `spot_drive_no_op_trivia` (iter 04) — added missing
  `--enable-auto-tool-choice --tool-call-parser hermes` to the
  launcher; minimal_agent_driver answered no-op-trivia in one chat
  call; probe shape `[22, 48, 8]` aligned.
- `laptop_merge_probes` (iter 05) — `runs/live/agent/hermes/
  qwen3-30b-a3b/bf16/no-op-trivia.json` carries non-null
  `response_routed_experts`.
- `spot_trainer_router_traces` (iter 06) — FSDP MoE teacher-force in
  80s; `expert_ids` shape `[510, 48, 8]` matches vLLM at position 0.
  Megatron leg deferred (no distcp shards on this spot).
- `laptop_pair_rollout_vs_trainer` (iter 07) — pair script slices
  trainer trace to gen window. **First real `router_flip_rate` on the
  wire: 3.50% (green band).**
- `laptop_aggregate_republish_dashboard` (iter 08) — yellow TBD
  placeholder section replaced by a live `hermes-agent-moe-router-
  matrix` reading `router_dashboard.json`. FSDP tile shows 3.50%
  green; Megatron tile shows TBD. Tests + CLAUDE.md updated.

## Standing notes (next initiatives may want)
- `prompt_routed_experts` is not in the engine output for this wheel;
  the surgical HTTP-shim patch is response-side only. Capturing it
  would need a deeper patch into the `RequestOutput` pipeline.
- The MoE-router matrix has one (count=1) data point. Repeating
  iter 04 → iter 07 with the other tasks in `agent_tasks_hermes.json`
  would tighten per-cell statistics.
- ~~Megatron MoE distcp shards at
  `~/checkpoint/qwen3-30b-a3b/megatron/release/` remain empty on the
  current spot; provisioning them would unlock the Megatron tile.~~
  **Resolved 2026-05-13** — see Post-cycle operator-directed work.

## How to start a new initiative

1. **Plan**: enter Plan Mode (`Shift+Tab`), describe the initiative
   in plain English. Claude drafts a plan and exits plan mode. Hit
   Accept.
2. **Seed**: `/seed-from-plan` (no args) — reads the plan from
   conversation context, writes `STEER.md` (the plan verbatim) and
   `.claude/feature-results.json` (one `passes:false` entry per
   iteration), and commits.
3. **Drive**: `/autonomous-loop` walks the queue end-to-end.

## Reproduction (durable)

Full single-host pipeline (laptop ↔ spot) documented in `CLAUDE.md`
under "End-to-end reproduction".
