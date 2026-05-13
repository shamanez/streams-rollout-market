# PROGRESS.md — Agent Handoff State

## Last updated
2026-05-13 — paused mid-cycle. **Cycle Remote-vLLM: 2 of 7 done.**
Resume with `/autonomous-loop` (picks up at iter 3).

## How to resume (hands-off)

Type `/autonomous-loop` in a fresh session. The loop reads
`STEER.md` + `.claude/feature-results.json` and starts the first
`passes:false` entry — currently iter 3 `tunnel_smoke_against_spot`.

### Iter 3 prep (operator does this once)

1. Spot up with bf16 vLLM-MoE on port 8000:
   ```bash
   ssh my-vllm-spot-instance 'cd ~ && tmux new-session -d -s vllm \
       bash ~/streams-rollout-market/scripts/live/vllm_serve_moe_dev.sh'
   ```
2. Laptop-side port-forward so `localhost:8001` → spot's `:8000`:
   ```bash
   ssh -fNL 8001:localhost:8000 my-vllm-spot-instance
   ```

After that, `/autonomous-loop` drives iters 3 → 7 unattended.

## Cycle Remote-vLLM — iteration checklist

1. `bundle_operator_serve_kit` ✓ — operator tarball at
   `/tmp/operator_kit.tgz` (built via
   `scripts/live/operator_kit/build_kit.sh`).
2. `client_side_capture_driver` ✓ —
   `scripts/live/remote_capture_driver.py` + 2 unit tests verify the
   sidecar JSONL shape matches the proxy's `extract_probes` output.
3. `tunnel_smoke_against_spot` ← **next**
4. `drive_6_task_batch_remote_style`
5. `teacher_force_remote_rollouts`
6. `pair_remote_results`
7. `dashboard_add_remote_row`

Each iteration's directive lives verbatim in `STEER.md` and
`.claude/feature-results.json`. The loop's stop clause: when all
seven entries are `passes:true`, delete `STEER.md` as the final
commit.

## How to start a NEW initiative (after this cycle finishes)

1. **Plan**: enter Plan Mode (`Shift+Tab`), describe the initiative
   in plain English. Claude drafts + exits plan mode. Accept.
2. **Seed**: `/seed-from-plan` — reads the plan from the
   conversation, writes `STEER.md` (verbatim) and
   `.claude/feature-results.json` (one `passes:false` per
   iteration), commits.
3. **Drive**: `/autonomous-loop` walks the queue end-to-end.

Both skills live under `.claude/skills/`.

## Past cycle results (live on the dashboard)

* **Cycle FP8** — 2×2 matrix on `docs/index.html`: bf16 rollouts
  ~6.0% flip rate, FP8 rollouts ~8.3% across both FSDP and Megatron
  trainers. FP8 quantization costs ~2.3-2.6 pp of router agreement;
  trainer engine choice contributes < 0.3 pp.
* **Cycle MoE-Router** — first `router_flip_rate` numbers landed
  (FSDP 3.50%, Megatron 3.03% on Hermes-Agent tasks).
* **Post-cycle work (2026-05-13)** — Megatron MoE tile filled
  (slime container symlink + `cp -rL` fix in
  `megatron_moe_reference_launch.sh`); cwc-long-running-agents Q&A
  trajectories added to the agent viewer (4 trajectories total).

Full per-cycle write-ups under `.claude/evidence/iterNN_<slug>/`.

## Standing notes

* `prompt_routed_experts` is response-side only on the current
  wheel; capturing prompt-side experts would need a deeper patch
  into vLLM's `RequestOutput` pipeline.
* The MoE-router matrix could be tightened by repeating the capture
  → teacher-force → pair loop with more tasks from
  `agent_tasks_hermes.json`.
* End-to-end reproduction on a fresh spot is in `docs/REPRODUCE.md`;
  `scripts/live/bootstrap_spot.sh` is idempotent and covers everything
  cycle FP8 needs.

## Operator controls

* **Halt**: `bash .claude/scripts/kill-switch.sh` (or
  `touch AGENT_STOP`).
* **Redirect**: `bash .claude/scripts/steer.sh "<directive>"`
  rewrites `STEER.md`; next iteration picks it up.
* **Resume after halt**: `rm AGENT_STOP`.
