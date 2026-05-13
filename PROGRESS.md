# PROGRESS.md — Agent Handoff State

## Last updated
2026-05-13 — iter 01 done; iter 02 next.

## Status
Active. Initiative: wire MoE router-flip-rate end-to-end. STEER.md seeded.

## Completed this cycle
- `spot_build_vllm_dev_routed_experts` (iter 01) — see
  `.claude/evidence/iter01_spot_vllm_dev/`. Pivot from source build to
  wheel + shim patch. Rationale and verification in the iter01.md.

## Queue (next up)
- `spot_serve_moe_router_replay` — launcher + curl-the-served-endpoint smoke.
- `spot_start_capture_proxy`, `spot_drive_no_op_trivia`,
  `laptop_merge_probes`, `spot_trainer_router_traces`,
  `laptop_pair_rollout_vs_trainer`, `laptop_aggregate_republish_dashboard`.

## Known gaps for the rest of the queue
- `prompt_routed_experts` (prompt-side routing) is not in the patched
  wheel; only generation-side `routed_experts` is exposed. The
  marketplace `router_flip_rate` metric is response-token-only, so
  this does not block the dashboard. Verification step 4 of STEER.md
  has been narrowed accordingly.
- Megatron MoE distcp shards at `$CKPT_ROOT/qwen3-30b-a3b/megatron/release/`
  are empty on the current spot. Iter 06's Megatron leg will either
  trigger a fresh conversion (~80 min, GPU-bound) or land FSDP-only
  and leave the Megatron tile TBD.

## How to start a new initiative

1. **Plan**: enter Plan Mode (`Shift+Tab`), describe the initiative
   in plain English. Claude drafts a plan and exits plan mode. Hit
   Accept.
2. **Seed**: `/seed-from-plan` (no args) — reads the plan from
   conversation context, writes `STEER.md` (the plan verbatim) and
   `.claude/feature-results.json` (one `passes:false` entry per
   iteration), and commits.
3. **Drive**: `/autonomous-loop` walks the queue end-to-end.

No template required. The plan can be in whatever shape Plan Mode
produced — the skill auto-detects iteration boundaries.

## Reproduction (durable)

Full single-host pipeline (laptop ↔ spot) documented in `CLAUDE.md`
under "End-to-end reproduction".
