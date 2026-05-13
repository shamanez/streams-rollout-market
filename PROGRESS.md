# PROGRESS.md — Agent Handoff State

## Last updated
2026-05-13 — initiative complete; STEER.md retired. Two operator-
directed follow-ups landed after the queue closed (see "Post-cycle
operator-directed work" below).

## Status
Idle. All eight queue entries `passes:true`. STEER.md deleted as the
final commit per its own stop clause.

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
