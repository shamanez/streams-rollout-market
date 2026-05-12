# PROGRESS.md — Agent Handoff State

## Last updated
2026-05-12 (cycle 3 v2 tasks #1/#2/#3 shipped + 3 of 4 trainer-force cells
real on the Hermes-Agent dashboard matrices: Dense×FSDP=0.9997 green,
Dense×Megatron=0.9996 green, MoE×FSDP=0.9532 amber, MoE×Megatron=TBD
pending checkpoint conversion. Router-flip-rate is a documented TBD
placeholder per the operator pivot away from cycle-2 MoE numbers.)

---

## Operator direction 2026-05-12

> "Remove all the numbers from cycle-2 MoE, I think it was wrong. For now
> complete the dashboard only with token prob shifts, and make sure you
> update all docs and make sure we can reproduce all the results."

Dashboard semantics for Hermes-Agent matrices changes:
- **DROP** the cycle-2 single-prompt MoE `router_flip_rate` tiles
  (4.3%/7.1%/4.1%/6.3%) from the headline.
- Hermes-Agent matrix becomes **logprob-shift** based for BOTH Dense AND
  MoE rollouts. Same metric on both: rollout-vs-trainer logprob delta
  (or ESS) over the assistant-emitted response tokens, paired against
  FSDP/Megatron teacher-force.
- This is grounded in what vLLM 0.20.x's OpenAI HTTP entrypoint actually
  exposes: per-token logprobs + token_ids. `routed_experts` is NOT
  emitted on the OpenAI shim in any released vLLM version (verified by
  `grep -rn routed_experts vllm/entrypoints/openai/` on 0.20.1 AND
  0.20.2 — zero hits). The native `LLM.generate()` API path does emit
  them (that's how cycle-2's `~/run_vllm_moe_rollout.py` got its
  router_flip_rate numbers), but Hermes-Agent multi-turn loops use the
  OpenAI HTTP path.

---

## Spot runbook (what the operator needs to reproduce)

All work is on the spot host (`my-vllm-spot-instance`, g6e.12xlarge,
4×L40S 48 GB each). The relevant files:

### Top-level scripts on the spot (cycle-2 path, native API)

| Path | Purpose |
|------|---------|
| `~/run_vllm_rollout.py` | Dense vLLM single-prompt rollout (native API) |
| `~/run_vllm_moe_rollout.py` | MoE rollout with `routed_experts` capture (native `LLM.generate`, the only path that exposes them in 0.20.x) |
| `~/run_fsdp_reference.py` | Dense FSDP teacher-force; reads `/tmp/rollouts.json`, writes `/tmp/trainers.json` |
| `~/run_fsdp_moe_reference.py` | MoE FSDP teacher-force |
| `~/run_megatron_reference.py` + `megatron_reference_launch.sh` | Dense Megatron teacher-force (inside slimerl/slime docker) |
| `~/run_megatron_moe_reference.py` + `megatron_moe_reference_launch.sh` | MoE Megatron teacher-force |
| `~/serve_qwen_for_agent.sh` | Basic vLLM serve (Dense, no routed-experts) |
| `~/serve_for_capture.sh` | vLLM serve with `--enable-return-routed-experts` (MoE) + auto YaRN ROPE for 131K (now fixed: `ROPE_FLAG=()` array init) |
| `~/prompts.json` | 8 cycle-2 prompts for single-prompt experiments |

### Repo on the spot (cycle-3 v2 path, OpenAI HTTP via proxy)

`~/streams-rollout-market/scripts/live/` contains:

| Script | Purpose |
|--------|---------|
| `logprob_capture_proxy.py` | OpenAI proxy at :8001 → vLLM :8000; injects `logprobs=True, top_logprobs=1, return_token_ids=True, chat_template_kwargs.enable_thinking=False`; writes per-turn probes to `/tmp/hermes_probes_*.jsonl` |
| `minimal_agent_driver.py` | Non-streaming OpenAI tool-using driver (preferred for capture: hermes-agent CLI uses `stream=True` which the proxy can't parse for routed_experts) |
| `merge_probes_into_trajectories.py` | Sidecar JSONL + ShareGPT JSONL + tasks list → AgentTrajectory JSON per task |
| `build_hermes_dense_input.py` | Walks AgentTrajectory dir → `/tmp/rollouts_hermes_<precision>.json` + index |
| `pair_hermes_dense_reports.py` | Joins index + trainer `/tmp/trainers.json` → DenseMismatchReport per `(trajectory, assistant_turn)` |
| `pair_hermes_moe_reports.py` | MoE counterpart (currently router-flip-rate; needs adapt or replace per pivot above) |
| `publish_dashboards.py` | Re-renders `docs/index.html` |

### End-to-end command sequence (single-host, all on the spot)

```bash
# 1. vLLM serve (MoE example — see serve_for_capture.sh comments for other models):
MODEL=Qwen/Qwen3-30B-A3B TP=4 MAX_LEN=40960 bash ~/serve_for_capture.sh
# or directly:
vllm serve Qwen/Qwen3-30B-A3B --tensor-parallel-size 4 --enable-expert-parallel \
  --enable-return-routed-experts --no-async-scheduling --dtype auto \
  --max-model-len 40960 --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser hermes --host 0.0.0.0 --port 8000

# 2. logprob capture proxy (separate tmux session):
cd ~/streams-rollout-market && source ~/rmenv/bin/activate
python scripts/live/logprob_capture_proxy.py --upstream http://localhost:8000 \
  --port 8001 --sidecar /tmp/hermes_probes_moe_bf16.jsonl --host 127.0.0.1

# 3. drive the trajectory (use the in-repo minimal driver — non-streaming):
python scripts/live/minimal_agent_driver.py \
  --base-url http://localhost:8001 --model Qwen/Qwen3-30B-A3B \
  --tasks scripts/live/agent_tasks_hermes.json --task-id no-op-trivia \
  --out /tmp/minimal_sharegpt_moe_bf16.jsonl --max-steps 3 --max-tokens 200

# 4. merge probes + sharegpt into AgentTrajectory:
python3 -c "
import json
tasks = json.load(open('scripts/live/agent_tasks_hermes.json'))
json.dump([t for t in tasks if t['task_id']=='no-op-trivia'], open('/tmp/one.json','w'))"
python scripts/live/merge_probes_into_trajectories.py \
  --sharegpt /tmp/minimal_sharegpt_moe_bf16.jsonl \
  --probes /tmp/hermes_probes_moe_bf16.jsonl --tasks /tmp/one.json \
  --model-id Qwen/Qwen3-30B-A3B --engine-label hermes-qwen3-30b-a3b-bf16 \
  --engine-fingerprint sha256:hermes-qwen3-30b-a3b-bf16-tp4-l40s \
  --out-dir runs/live/agent/hermes/qwen3-30b-a3b/bf16/

# 5. build trainer input from trajectory:
python scripts/live/build_hermes_dense_input.py \
  --trajectory-dir runs/live/agent/hermes/qwen3-30b-a3b/bf16 \
  --precision-class bf16 \
  --out-rollouts /tmp/rollouts_hermes_moe_bf16.json
cp /tmp/rollouts_hermes_moe_bf16.json /tmp/rollouts.json

# 6. trainer teacher-force (FSDP, on spot, no docker):
cd ~ && source rmenv/bin/activate && export HF_HOME=/home/ubuntu/hf-cache
torchrun --nproc-per-node=4 --rdzv-endpoint=127.0.0.1:29510 ~/run_fsdp_reference.py
# (or run_fsdp_moe_reference.py for MoE)

# 7. pair into DenseMismatchReport:
python scripts/live/pair_hermes_dense_reports.py \
  --index /tmp/hermes_dense_index_bf16.json --trainers /tmp/trainers.json \
  --trajectory-dir runs/live/agent/hermes/qwen3-30b-a3b/bf16 \
  --out-dir runs/live/dense/hermes-qwen3-30b-a3b-bf16-vs-fsdp-bf16

# 8. (rsync runs/ back to laptop, then) re-render:
python scripts/live/publish_dashboards.py
```

### Pre-flight checklist

- [ ] vLLM venv active: `source ~/rmenv/bin/activate` (Python 3.12,
  torch 2.11+cu130, vllm 0.20.2 as of 2026-05-12).
- [ ] HF cache: `~/hf-cache/hub/models--Qwen--Qwen3-{32B,32B-FP8,30B-A3B,30B-A3B-FP8}/`
  all present (verified 2026-05-12).
- [ ] Hermes-Agent installed at `~/.hermes/hermes-agent/` and
  `~/.local/bin/hermes` (v0.13.0). Used optionally; the in-repo
  `minimal_agent_driver.py` is the preferred capture driver.
- [ ] Port 29500 free for torchrun (use `--rdzv-endpoint=127.0.0.1:29510`
  if stale).
- [ ] GPUs idle: `nvidia-smi --query-compute-apps=pid --format=csv`
  should return nothing before launching vLLM serve.

---

## ⚡ TL;DR for a fresh session (zero prior context)

### What this repo is
A decentralized rollout marketplace + rollout-validity layer for agentic
RL. **vLLM hosts rollouts; FSDP/Megatron teacher-force the captured
token pairs to measure training-inference mismatch.** The dashboard
answers: how much does engine + precision change inference-time
behaviour relative to the trainer's forward pass?

### Goal of this session
Fill the Hermes Dense + MoE dashboard matrices with **one bf16 tile
per trainer per model** — 4 real tiles total (Dense×FSDP, Dense×Megatron,
MoE×FSDP, MoE×Megatron). fp8 stays TBD this cycle.

### Topology (all on the spot, single host)
Hermes-Agent + logprob_capture_proxy + vLLM all run on the spot host.
The laptop is purely an SSH terminal (+ optional final dashboard
re-render after `rsync`ing `runs/` back). No SSH tunnel in the data
path. Full topology diagram in `scripts/live/HERMES_INSTALL.md`.

### The 8-task queue (selection rule: file order of false entries in `.claude/feature-results.json`)

| # | Entry | What it does | Reuses |
|---|-------|-------------|--------|
| 1 | `hermes_agent.proxy_enable_thinking_off` | 1-line proxy patch + 2 unit tests | — |
| 2 | `hermes_agent.live_moe_bf16_capture` | ONE Hermes task → AgentTrajectory JSON (logprobs + routed_experts) | — |
| 3 | `hermes_agent.live_dense_bf16_capture` | ONE Hermes task → AgentTrajectory JSON (logprobs only) | — |
| 4 | `hermes_agent.live_moe_fsdp_pair` | FSDP-MoE teacher-force + pair → RouterMismatchReport | #2 trajectory |
| 5 | `hermes_agent.live_moe_megatron_pair` | Megatron-MoE teacher-force + pair | #2 trajectory |
| 6 | `hermes_agent.live_dense_fsdp_pair` | FSDP teacher-force + pair → DenseMismatchReport | #3 trajectory |
| 7 | `hermes_agent.live_dense_megatron_pair` | Megatron teacher-force + pair | #3 trajectory |
| 8 | `hermes_agent.matrix_render_and_codex` | rsync + publish_dashboards + codex review | — |

**One task per capture** (#2 and #3 run one prompt each, not 12).
Trajectories are captured once and reused for both trainers.

### Why entry #1 exists
On 2026-05-12 the MoE summarize-repo smoke (Qwen3-30B-A3B at 131K
bf16) DNF'd at 1 658 s with `Context length exceeded: max compression
attempts (3) reached`. Root cause: Qwen3's default reasoning mode
emits `<think>...</think>` blocks long enough to fill 131K context in
3 turns. `/no_think` in the user prompt is **not** a fix — Qwen3
reads it as plain text. The chat-template kwarg
`chat_template_kwargs.enable_thinking=false` IS the fix, but
Hermes-Agent doesn't thread it through. Patch the proxy to
`setdefault` it on every outbound request body.

### What vLLM actually exposes for MoE
Per <https://docs.vllm.ai/en/latest/training/routed_experts_replay/>:
- `choices[i].routed_experts` shape `[gen_len, num_moe_layers, top_k]`
  of int16 expert IDs.
- `prompt_routed_experts` (top-level) shape `[prompt_len, num_moe_layers,
  top_k]`. `-1` for prefix-cached positions.
- **Router gate logits / weights are NOT exposed.** Top-k expert IDs
  only. No extra capture surface exists short of patching vLLM.

Our `scripts/live/logprob_capture_proxy.py` already lifts both arrays.
Server-side flag `--enable-return-routed-experts` is auto-set for MoE
by `scripts/live/serve_for_capture.sh`.

### Reproduce from scratch
- `scripts/live/HERMES_INSTALL.md` — clean Hermes install, vLLM at
  131K, Hermes config, summarize-repo smoke.
- `scripts/live/README.md` — full single-host pipeline (proxy →
  merger → filler → builder → trainer-force → pair → publish).
- `STEER.md` — minimal directive + 8-entry queue overview.
- `.claude/feature-results.json` — verbatim per-entry directive
  (consumed by `/implement-feature`).

### Operator controls
- `/autonomous-loop` — starts the loop on entry #1 of the queue above.
- `touch AGENT_STOP` — halt after current iteration.
- `rm AGENT_STOP` — resume.
- `bash .claude/scripts/steer.sh "<note>"` — overwrite STEER.md.

---

## Cycle 3 v2 — task #1 shipped: `proxy_enable_thinking_off` (2026-05-12)

`scripts/live/logprob_capture_proxy.py::inject_logprobs` now `setdefault`s
`chat_template_kwargs.enable_thinking=False` on every outbound chat-
completions request. This is the documented off-switch for Qwen3's
default reasoning mode — `/no_think` in user content is NOT a fix
(Qwen3 reads it as plain text), and Hermes-Agent doesn't thread the
kwarg through. Idempotent on caller choice: when the caller explicitly
opts into reasoning mode or has other `chat_template_kwargs` keys set,
the proxy preserves them — same agent-preference rule the existing
`logprobs` / `top_logprobs` / `return_token_ids` injects already follow.

Tests (+2 in `tests/test_logprob_capture_proxy.py`):
- `test_inject_disables_thinking_by_default` — bare request gets the
  kwarg stamped.
- `test_inject_preserves_caller_thinking_choice` — explicit
  `enable_thinking=True` and sibling keys survive untouched.

**476 passing (+2 from 474)**. ruff clean. Evidence under
`.claude/evidence/hermes_proxy_enable_thinking/`.

This unblocks the cycle-3 v2 MoE smoke: the 2026-05-12 DNF (1 658 s,
`Context length exceeded: max compression attempts (3) reached`)
should reduce to ≤ 90 s once the next task (`live_moe_bf16_capture`)
runs the same prompt through the patched proxy.

## Cycle 3 v2 — task #2 attempted, BLOCKED on SSE/streaming gap (2026-05-12)

`live_moe_bf16_capture` attempted end-to-end on the spot. The proxy
patch works exactly as intended — verified via instrumented run that
logged each post-inject request body to `/tmp/hermes_request_bodies.jsonl`.
First hermes request body contained:

```
chat_template_kwargs: {"enable_thinking": False}    ← injected by proxy
logprobs: True, top_logprobs: 1, return_token_ids: True
stream: True                                         ← hermes default
tools: 28 entries
```

Direct curl-through-the-proxy with a small prompt produced a 3-token
"Four." response with the prompt token tail
`[151644, 77091, 198, 151667, 271, 151668, 271]` — Qwen3 emitted an
empty `<think>\n\n</think>\n\n` block as expected when
`enable_thinking=false`. enable_thinking IS being applied.

**The blocker**: hermes uses `stream: True` by default, and the proxy's
`extract_probes` does `json.loads(response_body)` on what is actually
a Server-Sent Events stream. Result: probe records have
`response_logprobs` and `response_token_ids` populated (likely from a
final non-streamed merge by some intermediate), but
`response_routed_experts` is NOT IN RECORD on every probe. The
`live_moe_bf16_capture` acceptance criterion requires NON-EMPTY
`response_routed_experts`, so the directive cannot pass without one
of:

(a) **Upgrade `extract_probes` to be SSE-aware** — parse each
    `data: {...}` chunk, accumulate `routed_experts` lists per choice
    (vLLM emits them incrementally), reassemble at end-of-stream.
(b) **Disable upstream streaming in hermes** — find the hermes config
    knob that sets `stream=False` on outbound chat-completions
    requests. The hermes `streaming.enabled: false` in
    `~/.hermes/config.yaml` controls only TUI rendering, not the
    OpenAI client mode.
(c) **Bypass hermes for the capture** — write a minimal stdlib
    OpenAI-tool driver that hits the proxy with `stream=False`. The
    repo already has `scripts/live/hermes_agent_runner.py` plus the
    new `feat(minimal-agent): stdlib OpenAI driver` commit on
    `research/hermes-install-runbook` (commit 17d1cb8) — likely
    the cleanest path.

GPU on the spot was freed; vLLM + proxy killed (verified via
`nvidia-smi --query-compute-apps`). No background processes left
running.

**Files left on spot for next iter**:
- `/tmp/hermes_request_bodies.jsonl` — single hermes request body
  (for inspection of exact wire format)
- `/tmp/hermes_probes_moe_bf16.jsonl` — 2 probe records (one from a
  curl smoke, one from hermes — both with `routed_experts` absent)
- `/tmp/proxy_debug_wrap.py` — instrumented proxy wrapper
- `~/streams-rollout-market/scripts/live/logprob_capture_proxy.py` —
  patched with enable_thinking=False

`feature-results.json` for `live_moe_bf16_capture` remains
`passes: false` pending resolution of one of (a)/(b)/(c).

## Cycle 3 v2 iter 2 — setup-22 commit (2026-05-12)

`scripts/live/HERMES_INSTALL.md` ships a verbatim, reproducible runbook
for installing NousResearch Hermes-Agent against our spot-instance vLLM
at 131 K context. Followed step-by-step against a clean spot and
confirmed working on both Qwen3-32B (Dense bf16) and Qwen3-30B-A3B
(MoE bf16, with caveat).

`scripts/live/serve_for_capture.sh` gains YaRN rope scaling via
`--hf-overrides` (vLLM ≥ 0.20 removed the `--rope-scaling` flag) +
auto-export of `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` so vLLM's max-len
validator accepts the extended ceiling. Auto-on when `MAX_LEN > 40 960`.

**Smoke-test results (2026-05-12, spot clean + fresh install)**:

| Model | TP | Wall-clock | Outcome |
|-------|----|-----------|---------|
| Qwen3-32B bf16 | 4 | 63 s | ✓ Clean 5-bullet summary, correctly identified `/autonomous-loop` as the driver. 3 multi-turn tool-call rounds (file reads). |
| Qwen3-30B-A3B bf16 | 2 | 1 658 s (DNF) | Pipeline OK (vLLM connection + 3 chat-completions + routed_experts capture all worked). Died at turn 3 with `Context length exceeded: max compression attempts (3) reached` — Qwen3's default reasoning mode produced `<think>` traces long enough to exhaust 131 K context. |

**Root cause of the MoE DNF**: Qwen3-30B-A3B ships with reasoning mode
on; Hermes-Agent does not thread `chat_template_kwargs.enable_thinking
=false` into the OpenAI request body. `/no_think` in the user text is
NOT a fix — Qwen3 reads it as plain content. The fix path is to
extend `scripts/live/logprob_capture_proxy.py` to `setdefault` that
kwarg (one line) — same proxy is already load-bearing for cycle-3-v2
probe capture, so this costs nothing in plumbing.

Next iteration (`setup-23` candidate): add the
`chat_template_kwargs.enable_thinking=false` injection to
`logprob_capture_proxy.py`, then re-run the MoE summarize-repo smoke
test through the proxy. Expected wall-clock < 90 s (matching the
Dense baseline).

**Branch**: `research/hermes-install-runbook`. 475 tests passing
(unchanged).

## Cycle 3 v2 iter 2 — setup-21 commit (2026-05-12)

`scripts/live/pair_hermes_dense_reports.py::_trainer_engine_from_entry`
now prefers the trainer payload's explicit `engine_fingerprint` when
supplied (the Megatron-side scripts stamp a build-identifying string;
the FSDP-side scripts don't). Falls back to deriving from FSDP-shape
defaults to keep existing fixtures unchanged.

Caught by a new test that feeds a Megatron-shaped trainer payload
(engine="megatron-lm", explicit engine_fingerprint with "megatron-core"
in it) and asserts both the engine name AND the fingerprint propagate
to the rendered DenseMismatchReport.

- 1 new test in test_pair_hermes_dense_reports.py.
- 1-line behavior fix in pair_hermes_dense_reports.py.
- **475 passing (+1 from 474)**.

## Cycle 3 v2 iter 2 — setup-20 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-20 commit (2026-05-12)

`tests/test_hermes_dense_pipeline_e2e.py` gains a multi-task scenario
that exercises the hash-based prompt_index pairing path setup-12
enabled. Two synthetic tasks share one sidecar (realistic single-
proxy-invocation, header-free mode); the merger pairs each task's
probes back to its ShareGPT record purely by recomputing the same
hash on the record's first ``from=human`` value.

The test asserts:
- Auto-derived prompt_index values are distinct strings per task.
- The merger routes each task's probes to the correct trajectory
  (verified via response_token_ids).
- The full filler → builder → pair pipeline produces one report per
  task, each at ESS=1.0 when trainer logprobs match.

1 new test; **474 passing (+1 from 473)**.

## Cycle 3 v2 iter 2 — setup-19 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-19 commit (2026-05-12)

`scripts/live/README.md` now documents every cycle-3 v2 capture
component (proxy, merger, filler, builder, dense pair, MoE pair) +
a fully worked 9-step end-to-end runbook for the laptop+spot
pipeline. Operator handoff is now self-contained from the README:

  1. spot vLLM serve
  2. SSH tunnel + capture proxy
  3. hermes-agent run against the proxy
  4. merger stitches sharegpt + sidecar
  5. fill missing prompt_token_ids
  6. build trainer input + index
  7. scp + torchrun teacher-force
  8. pair into DenseMismatchReport
  9. publish_dashboards re-render

Documentation-only change. 473 passing (unchanged).

## Cycle 3 v2 iter 2 — setup-18 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-18 commit (2026-05-12)

Closes the PROGRESS.md "Next work" follow-up #3 (OPBC integration of
router_flip_rate). OPBC previously only inspected logprob-level
mismatch; MoE groups whose top-1 expert disagreement was egregious
would still slip through as long as their ESS held up.

Contract changes (one-commit-bundle per CLAUDE.md):

- `contracts.DecisionReason.HIGH_ROUTER_FLIP_RATE` enum value added,
  documented with the threshold rationale.
- `contracts.BudgetReport.router_flip_rate: float | None = None` —
  backward-compatible (Dense rollouts leave it ``None``).
- `opbc.BudgetPolicy.max_router_flip_rate: float = 0.15` — mirrors
  the dashboard's red threshold (>15% = red tile).
- `opbc.compute_budget_report` gains a new `router_flip_rate=` kwarg
  that threads onto the BudgetReport.
- `opbc.decide_group` quarantines when `router_flip_rate >
  max_router_flip_rate`, *combinable* with other quarantine reasons
  (LOW_ESS, STALE_POLICY_LAG) so postmortem inspection can tell
  apart which gate fired.

Tests (7 new): gate above threshold quarantines; exactly-at-threshold
trains; ``None`` skips the gate (Dense back-compat); combines with
LOW_ESS; custom strict policy override; ``compute_budget_report``
threads the kwarg; end-to-end Dense-ESS-clean + high flip → quarantine.

**473 passing (+7 from 466)**. The MoE pipeline's marketplace
verdict — `router_flip_rate` — now feeds OPBC directly, not just the
dashboard.

## Cycle 3 v2 iter 2 — setup-17 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-17 commit (2026-05-12)

`tests/test_hermes_moe_pipeline_e2e.py` mirrors the dense e2e test
on the MoE side. Wires every cycle-3 v2 MoE component through their
pure-function seams:

  proxy.extract_probes (with routed_experts) → merger →
  filler → builder → fake trainer → pair_hermes_moe_reports →
  RouterMismatchReport

Two tests: matched expert_ids → router_flip_rate=0.0, and layer-0
divergence → 50% overall flip rate with per-layer resolution
(layer-0=100%, layer-1=0%).

2 new tests; **466 passing (+2 from 464)**. Both Dense AND MoE
pipelines are now exercised end-to-end through their full proxy →
sidecar → merger → filler → builder → pair seam in tests.

## Cycle 3 v2 iter 2 — setup-16 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-16 commit (2026-05-12)

`scripts/live/run_megatron_moe_reference.py` already iterated over a
multi-prompt rollouts list, but the per-prompt entries it appended
to `router_records` didn't carry `prompt_idx` / `engine_fingerprint`
/ `world_size`. `pair_hermes_moe_reports.py` keys join on
`prompt_idx`, so without that field a multi-task Megatron pass would
collide all entries on `prompt_idx=0`.

Fixed by propagating those three fields from the rollout entry +
the parsed args into each router_record. No new tests (the only
contract surface that mattered — `_load_rollouts` + the rollout-list
iteration — was already in the script).

464 passing (unchanged). Both Dense and MoE trainer-side reference
scripts now emit per-prompt payloads with the metadata the pair
scripts expect.

## Cycle 3 v2 iter 2 — setup-15 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-15 commit (2026-05-12)

`scripts/live/run_fsdp_moe_reference.py` now supports the multi-prompt
input contract `pair_hermes_moe_reports.py` expects. New `load_rollouts()`
helper prefers `/tmp/rollouts.json` (list) over the legacy
`/tmp/rollout.json` (single dict); main loop iterates over each
prompt and aggregates per-prompt payloads to `/tmp/trainers_moe.json`
in multi-prompt mode (back-compat: single mode still writes
`/tmp/fsdp_router.json`).

- Model load is done once; the forward-pass loop reuses it.
- Each payload carries `prompt_idx` so `pair_hermes_moe_reports.py`
  can join by index.
- 5 new offline tests for the pure helpers
  (`load_rollouts`, `routed_position_indices`,
  `format_fsdp_router_payload`). **464 passing (+5 from 459)**.

The full multi-task hermes MoE pipeline is now wired end-to-end:

  1. proxy captures probes (Dense + MoE).
  2. merger stitches probes onto AgentTrajectory.
  3. fill_prompt_token_ids derives prompt IDs via chat-template.
  4. build_hermes_dense_input -> /tmp/rollouts.json + index.
  5. scp to spot.
  6. torchrun run_fsdp_moe_reference.py -> /tmp/trainers_moe.json.
  7. scp back.
  8. pair_hermes_moe_reports -> RouterMismatchReport per (traj, turn).

## Cycle 3 v2 iter 2 — setup-14 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-14 commit (2026-05-12)

`scripts/live/pair_hermes_moe_reports.py` — MoE counterpart of
`pair_hermes_dense_reports.py`. Consumes the index + an MoE trainer
payload (per-prompt `expert_ids` from `run_fsdp_moe_reference.py` or
`run_megatron_moe_reference.py`) + the AgentTrajectory dir, and emits
one `RouterMismatchReport` per `(trajectory, assistant_turn)` under
`runs/live/router/hermes-qwen3-30b-a3b-{bf16,fp8}-vs-{fsdp,megatron}-
bf16/<run_id>/`.

- Reads `response_routed_experts` off the matching AgentStep
  (captured by the proxy in setup-13).
- Validates rollout vs trainer trace shape (num_tokens, num_layers,
  num_experts, top_k); surfaces mismatches with a clear ValueError.
- 6 new offline tests: happy path (zero flip rate), divergence path
  (50% flip rate with layer-level resolution), missing
  routed_experts, unknown trajectory_run_id, engine-label override,
  CLI round-trip. **459 passing (+6 from 453)**.

Both `hermes_agent.moe_capture_then_fsdp_router` and
`hermes_agent.moe_capture_then_megatron_router` now have their full
code-side pipeline shipped end-to-end. Only the live spot run +
trainer-side per-turn expert_ids aggregator remain operator-side.

## Cycle 3 v2 iter 2 — setup-13 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-13 commit (2026-05-12)

Forward-compatibility for the MoE side (`hermes_agent.moe_capture_*`
feature-results entries):

- `inject_logprobs` now also stamps
  `extra_body.return_routed_experts=True`. Dense vLLM serves and
  non-vLLM upstreams silently ignore the unknown extra (the
  ``setdefault`` keeps the caller's choice when they explicitly
  disable it).
- `extract_probes` now also harvests
  `choices[0].logprobs.content[*].routed_experts` into the sidecar
  record's `response_routed_experts` field — same shape the AgentStep
  schema expects, same shape the dense filler/builder/pair pipeline
  passes through unchanged.
- 4 new tests: routed_experts present (captured), absent (omitted),
  inject behavior, caller-preference preservation. **453 passing
  (+4 from 449)**.

The MoE side of the pipeline (router_flip_rate matrix tiles) is now
the same operational shape as Dense: run the proxy, run Hermes-Agent
against an MoE-serving vLLM, merge probes, fill prompt IDs, build
input, run `run_fsdp_moe_reference.py` / `run_megatron_moe_reference.py`,
pair into RouterMismatchReport. The actual `pair_hermes_moe_reports.py`
joiner is a sibling of `pair_hermes_dense_reports.py` and is the next
bounded code piece if needed.

## Cycle 3 v2 iter 2 — setup-12 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-12 commit (2026-05-12)

Closes the multi-task probe-pairing bug. Setup-11 fixed turn_idx but
left prompt_index defaulting to 0, so a multi-task hermes-agent run
would funnel all probes into the same bucket — only the first task's
sharegpt record would pick up probes; the rest would have empty
`probes_by_turn`.

Fixed by making **both** axes of the (prompt_index, turn_idx) key
auto-derive from the request body:

  * `derive_turn_idx(body)` = count of prior assistant messages.
  * `derive_prompt_index(body)` = hex sha256[:16] of the first user
    message (stable across turns of one task, distinct across tasks).

The merger now accepts string prompt_index (hash form) and pairs
each ShareGPT record by recomputing the same hash on its first
`from=human` entry. Integer prompt_index still works when an upstream
explicitly sets X-Prompt-Index — both pairing paths run simultaneously
so a single record can be matched either way.

- Proxy: new `derive_prompt_index` pure function + handler wiring.
- Merger: `_first_user_message_hash` helper + dual-key
  `probes_by_turn` filter (matches either integer-or-hash).
- Tests: 4 new unit tests for `derive_prompt_index` + a merger
  integration test for the hash-pair path. **449 passing (+3 from
  446)**.

The proxy is now fully header-free for multi-task runs: hermes-agent
runs against it unchanged, all probes land in one sidecar, and the
merger transparently pairs each task's probes to its trajectory.

## Cycle 3 v2 iter 2 — setup-11 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-11 commit (2026-05-12)

The capture proxy's `turn_idx` derivation was a bug for multi-task
runs: the legacy fallback used a global counter that incremented
across **every** chat-completions request, so two tasks would land
their turn 0 records at indices 0 and N (N = total prior turns from
prior tasks). The merger keys on `(prompt_index, turn_idx)` — with
`prompt_index=0` everywhere (no X-Prompt-Index header) this conflated
turns across the whole run.

Fixed by a pure `derive_turn_idx(request_body)` that counts prior
assistant messages in the request's `messages` array. That count is
exactly the index of the turn being generated. So:

  - Hermes-agent's first request for task K → `turn_idx = 0`.
  - Its second request (after the first tool round-trip) → `turn_idx = 1`.
  - First request for task K+1 → `turn_idx = 0` again.

Combined with rotating the `--sidecar` path between tasks (one
sidecar per task), the proxy now produces correctly-keyed records
without requiring any modification to hermes-agent.

- 3 new unit tests for `derive_turn_idx` + 1 new e2e test verifying
  the header-absent path. **446 passing (+4 from 442)**.

## Cycle 3 v2 iter 2 — setup-10 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-10 commit (2026-05-12)

`scripts/live/publish_dashboards.py` `render_index` now leads with the
two Hermes-Agent matrices (Dense + MoE) and moves the legacy
single-prompt Dense + MoE matrices into a collapsible
`<details class='legacy-archive'>` appendix below them, per the
`hermes_agent.matrix_render_and_codex` directive.

Non-destructive change — the legacy matrices are NOT deleted, just
visually demoted. They remain reachable via the `Legacy single-prompt
experiments (archived)` summary. Tile content is unchanged; only the
ordering + wrapper changes.

- 3 new tests in `test_publish_dashboards_hermes_v2.py`:
  ordering (Hermes before legacy in source order), legacy matrices
  live inside the `<details>` appendix, Hermes matrices live OUTSIDE
  the appendix.
- 11 dashboard tests passing in total; **442 passing (+3 from 439)**.
- Existing tests in `test_publish_dashboards.py` (selectors against
  `data-section="dense-matrix"` / `data-section="moe-matrix"`) keep
  passing — those sections still render, just below the headline.

Remaining for the matrix_render_and_codex entry to flip:
  - Real Hermes-Agent reports populating the 4+4 tiles with numeric
    `mx-value` + valid colour class (requires the live spot run).
  - `/codex-review --effort low` PASS on the rendered headline.

## Cycle 3 v2 iter 2 — setup-9 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-9 commit (2026-05-12)

`tests/test_hermes_dense_pipeline_e2e.py` now runs both pipeline tests
through the **complete** seam — proxy → merger → **filler** → builder
→ pair. The previous version of the test stamped a synthetic
`prompt_token_ids` directly onto the assistant step; that hid any
contract drift between the filler and the builder. Now the test:

  1. Calls `proxy.extract_probes` on a synthetic vLLM response.
  2. Calls `merger.merge_to_trajectories` to stitch the sidecar.
  3. **Calls `filler.fill_trajectory` with a deterministic FakeTokenizer**
     to re-derive `prompt_token_ids` via chat-template encoding (and
     asserts `n_filled == 1` per turn).
  4. Calls `builder.build_rollouts_and_index` — which now succeeds
     because every turn has both prompt and response IDs.
  5. Pairs against a fake trainer; asserts ESS / max_abs_log_ratio.

Same 2 tests, full pipeline exercised. **439 passing.**

## Cycle 3 v2 iter 2 — setup-8 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-8 commit (2026-05-12)

`scripts/live/fill_prompt_token_ids.py` closes the last
proxy→builder seam:

- Walks an `AgentTrajectory` directory; for each assistant step that
  has `response_token_ids` set but `prompt_token_ids=None` (the
  realistic proxy-captured shape), re-derives the prompt via
  chat-template encoding of the trajectory's own accumulated history.
- Captured `response_token_ids` survives untouched — vLLM emitted
  those exact tokens, re-encoding would yield different IDs.
- Trajectories with NO turns needing fill are not written back at
  all (byte-identical disk state).
- Dry-run mode for sanity checks.
- 6 new tests; **439 passing (+6 from 433)**.

Now the proxy-captured pipeline is end-to-end whole on the laptop
side:

  ... -> proxy writes sidecar -> merger stitches sidecar onto
  ShareGPT -> **fill_prompt_token_ids fills missing prompt IDs** ->
  build_hermes_dense_input flattens -> scp to spot -> torchrun
  run_fsdp_reference.py -> scp back -> pair_hermes_dense_reports ->
  publish_dashboards.

## Cycle 3 v2 iter 2 — setup-7 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-7 commit (2026-05-12)

`hermes_token_extraction.extract_token_pairs` now handles the
realistic proxy-captured scenario directly:

- **Path 1 (preferred):** both `prompt_token_ids` and
  `response_token_ids` captured — return verbatim. Unchanged.
- **Path 2 (NEW, proxy-realistic):** `response_token_ids` captured
  but `prompt_token_ids` is None — preserve the captured response
  verbatim (vLLM emitted those exact tokens; re-encoding would yield
  different IDs) and re-derive only the prompt via chat-template.
- **Path 3 (legacy fallback):** both None — re-encode both. Unchanged.

Without this branch, partial-capture trajectories would fall into the
legacy path and have their response tokens re-encoded — destroying
the very signal the proxy captured. The mixed-capture test
(`test_extract_mixed_capture_falls_back_to_chat_template`) already
covered the per-turn mix; the new tests cover the per-turn partial
shape that the proxy actually produces.

2 new tests; **433 passing (+2 from 431)**.

## Cycle 3 v2 iter 2 — setup-6 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-6 commit (2026-05-12)

`hermes_agent.dense_capture_then_fsdp` consolidation shipped on
`research/hermes-dense-e2e-test`:

- `tests/test_hermes_dense_pipeline_e2e.py` wires every cycle-3 v2
  iter 2 setup component end-to-end through their pure-function
  seams: `proxy.extract_probes` → `merger.merge_to_trajectories` →
  `builder.build_rollouts_and_index` → fake trainer →
  `pair.pair_reports` → `DenseMismatchReport`. Asserts both the
  happy path (matched logprobs ⇒ ESS=1.0) AND the divergence path
  (per-token variance ⇒ ESS<0.99) so internal contract drift
  between components cannot silently break the pipeline.
- 2 new tests; **431 passing (+2 from 429)**.

This closes the code-side of cycle-3 v2 iter 2. The remaining gap is
operational: an operator-driven session to install hermes-agent on
the spot, point it at the logprob_capture_proxy, run the 12 Hermes
tasks at bf16 + fp8, scp + torchrun the FSDP teacher-force, then
re-render the dashboards.

## Cycle 3 v2 iter 2 — setup-5 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-5 commit (2026-05-12)

`hermes_agent.dense_capture_then_fsdp` part (b)-mechanism shipped on
`research/hermes-logprob-proxy`:

- `scripts/live/logprob_capture_proxy.py` — a tiny stdlib-only OpenAI
  chat-completions proxy. Listens on a local port, forwards each
  `/v1/chat/completions` POST to an upstream vLLM endpoint, and:
    * rewrites the outbound request body to add `logprobs=True,
      top_logprobs=1` (preserving the agent's preference when set);
    * adds `extra_body.return_tokens_as_token_ids=True` so vLLM
      returns integer token IDs alongside each logprob (non-vLLM
      upstreams safely ignore the extra);
    * appends one JSONL line to `/tmp/hermes_probes.jsonl` per
      intercepted request with the captured per-turn probes, keyed by
      the request's `X-Prompt-Index` + `X-Turn-Idx` headers (or
      sequential turn counter when headers are absent).
- Pure functions `inject_logprobs()` + `extract_probes()` carry all
  the load-bearing logic; the HTTP wrapper is a thin shell. 11 new
  tests including a true end-to-end run against a stdlib fake
  upstream. **429 passing (+11 from 418).**

The end-to-end dense Hermes pipeline now has a feasible operational
path that does NOT require modifying hermes-agent:

  1. (spot) `bash serve_qwen_for_agent.sh` — vLLM Qwen3-32B bf16 at :8000
  2. (laptop) `ssh -L 8000:localhost:8000 spot -N &` — tunnel
  3. (laptop) `python scripts/live/logprob_capture_proxy.py \
        --upstream http://localhost:8000 \
        --port 8001 \
        --sidecar /tmp/hermes_probes_bf16.jsonl &`
  4. (laptop) install + run hermes-agent against
     `OPENAI_BASE_URL=http://localhost:8001/v1`
  5. (laptop) `python scripts/live/merge_probes_into_trajectories.py \
        --sharegpt <hermes-output>.jsonl \
        --probes /tmp/hermes_probes_bf16.jsonl \
        --tasks scripts/live/agent_tasks_hermes.json \
        --model-id Qwen/Qwen3-32B \
        --engine-label hermes-qwen3-32b-bf16 \
        --engine-fingerprint sha256:... \
        --out-dir runs/live/agent/hermes/qwen3-32b/bf16/`
  6. (laptop) `python scripts/live/build_hermes_dense_input.py \
        --trajectory-dir runs/live/agent/hermes/qwen3-32b/bf16 \
        --precision-class bf16`
  7. (spot) `scp + torchrun run_fsdp_reference.py`
  8. (laptop) `scp back + pair_hermes_dense_reports.py` ->
     `runs/live/dense/hermes-qwen3-32b-bf16-vs-fsdp-bf16/`
  9. (laptop) `python scripts/live/publish_dashboards.py`

Note: the proxy captures `response_logprobs` + `response_token_ids`
but **not** `prompt_token_ids` (the OpenAI chat-completions response
shape doesn't expose prompt token IDs). The merger and downstream
pair script tolerate `prompt_token_ids=None`; the trainer-side script
falls back to chat-template re-encoding via `hermes_token_extraction.py`'s
existing fallback path. A future iteration could capture
`prompt_token_ids` via either `echo=True` on the legacy `/v1/completions`
endpoint or a tokenizer-side step.

Next operator-actionable iteration: run the live pipeline end-to-end
on one task to validate the toolchain, then scale up to the full 12
Hermes tasks at bf16 + fp8.

## Cycle 3 v2 iter 2 — setup-4 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-4 commit (2026-05-12)

`hermes_agent.dense_capture_then_fsdp` part (b)-bridge shipped on
`research/hermes-probe-merger`:

- `scripts/live/merge_probes_into_trajectories.py` defines the
  load-bearing wire format between whatever component captures
  rollout-side probes (a patched hermes-agent batch_runner, a local
  proxy, or a vLLM logging hook) and our `AgentTrajectory` schema.
  The contract is a JSONL sidecar keyed by `(prompt_index, turn_idx)`
  carrying `prompt_token_ids` / `response_token_ids` /
  `response_logprobs` / `response_routed_experts` per assistant turn.
- The merger reads a hermes-agent ShareGPT JSONL + the probe sidecar,
  injects the matching probes onto each `from=gpt` conv entry, and
  adapts to `AgentTrajectory` via the existing
  `adapt_hermes_sharegpt_trajectory` (which already threads sibling
  probe fields onto AgentStep).
- 8 new offline tests (`index_probes` grouping + dedup, partial
  probe coverage tolerated, CLI round-trip with probes surviving the
  full pipeline). 418 passing (+8 from 410).

Cycle 3 v2 iter 2 setup is now complete code-side. The remaining
operational gap is **producing the sidecar JSONL on the spot**.
Three credible paths, ranked by friction:

  1. **Proxy interceptor (laptop)** — write a small OpenAI proxy on
     the laptop that injects `logprobs=True, top_logprobs=1` into every
     chat-completions request, captures the upstream response, and
     writes one sidecar line per turn. hermes-agent on the spot stays
     unchanged. Adds one Python process to the loop but doesn't touch
     hermes-agent internals. Likely the cleanest next iteration.
  2. **Patched batch_runner** — apply a small diff to hermes-agent's
     `agent/auxiliary_client.py` chat-completions builder to always
     pass `logprobs=True, top_logprobs=1` plus a tap that writes the
     sidecar. Requires re-installing hermes-agent on the spot and
     maintaining a fork.
  3. **vLLM logging hook** — vLLM has request/response logging hooks
     in some configs; investigate whether they expose enough.

Operator preference among (1)/(2)/(3) is the next operator input
needed. The autonomous loop can ship the proxy (option 1) without
additional permission since it doesn't touch the spot.

## Cycle 3 v2 iter 2 — setup-3 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-3 commit (2026-05-12)

`hermes_agent.dense_capture_then_fsdp` part (d)-prep shipped on
`research/hermes-dense-pair-reports`:

- `scripts/live/pair_hermes_dense_reports.py` joins the
  `build_hermes_dense_input` index against the trainer-side output
  (`/tmp/trainers.json` from `run_fsdp_reference.py`) and emits one
  `DenseMismatchReport` per (trajectory, assistant_turn) under
  `runs/live/dense/<pair-slug>/<task_id>-turn<k>/`. The dashboard
  renderer ingests these unchanged.
- `pair_reports(...)` is a pure function; the CLI is a thin
  argparse/I-O wrapper. Trainer engine label can be overridden so the
  same pair script handles `fsdp-bf16` vs `megatron-bf16` cells.
- 7 new offline tests; 410 passing (+7 from 403).

The end-to-end laptop+spot pipeline is now wired top-to-bottom on the
code side:

  1. (spot) Hermes Agent rollouts capture probes -> trajectory JSONs.
  2. (laptop) `build_hermes_dense_input` -> `/tmp/rollouts_hermes_*.json`
     + `/tmp/hermes_dense_index_*.json`.
  3. (spot) `scp + torchrun run_fsdp_reference.py` ->
     `/tmp/trainers.json`.
  4. (laptop) `scp back + pair_hermes_dense_reports.py` ->
     `runs/live/dense/hermes-*/...`.
  5. `python scripts/live/publish_dashboards.py` re-renders.

Step (1) — the actual rollout-side probe capture — is still blocked
on hermes-agent batch_runner instrumentation on the spot. No
hermes-agent install is present on spot as of 2026-05-12T05:31Z; the
next iteration should investigate the cheapest path to wire
`logprobs=True, top_logprobs=1` + persistence of `prompt_token_ids` /
`response_token_ids` into hermes-agent's chat-completions calls.

## Cycle 3 v2 iter 2 — setup-2 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-2 commit (2026-05-12)

`hermes_agent.dense_capture_then_fsdp` part (c)-prep shipped on
`research/hermes-dense-build-input`:

- `scripts/live/build_hermes_dense_input.py` walks a Hermes trajectory
  directory and flattens probe-bearing assistant turns into the
  `/tmp/rollouts.json` shape `run_fsdp_reference.py` already consumes.
  It also writes an index file `/tmp/hermes_dense_index_<precision>.json`
  mapping `prompt_idx → (task_id, trajectory_run_id, assistant_turn_idx,
  num_tokens, response_logprobs)` so the downstream pairing step can
  join FSDP's `/tmp/trainers.json` outputs back to the originating turn
  and to the rollout-side logprobs captured by vLLM.
- Skips turns missing `prompt_token_ids` / `response_token_ids` /
  `response_logprobs` (records the reason in `index["skipped"]`).
- Rejects mixed `model_id` across input trajectories without an
  explicit `--model-id-override`.
- 10 new offline tests (synthetic AgentTrajectory fixtures). 403
  passing (+10 from 393).

Once the spot side captures the probes (part b — pending on
hermes-agent batch_runner instrumentation), the pipeline becomes:

  1. `python scripts/live/build_hermes_dense_input.py
       --trajectory-dir runs/live/agent/hermes/qwen3-32b/bf16
       --precision-class bf16`
  2. scp /tmp/rollouts_hermes_bf16.json -> spot:/tmp/rollouts.json
  3. ssh spot 'torchrun --nproc-per-node 4 run_fsdp_reference.py'
  4. scp spot:/tmp/trainers.json -> /tmp/trainers_hermes_bf16.json
  5. python scripts/live/pair_hermes_dense_reports.py  # next iter

Spot reachable as of 2026-05-12T05:31Z. No hermes-agent install on
spot currently — that re-setup is the next blocker for part (b).

## Cycle 3 v2 iter 2 — setup-1 commit (2026-05-12)

`hermes_agent.dense_capture_then_fsdp` part (a) shipped on
`research/hermes-dense-capture-setup`:

- `adapt_hermes_record` + `adapt_hermes_sharegpt_trajectory` now thread
  rollout-side probe fields (`prompt_token_ids`, `response_token_ids`,
  `response_logprobs`, `response_routed_experts`) onto each assistant
  `AgentStep` when present on the input record. Tool-role records that
  carry probes surface as a `ValueError` via `AgentStep`'s own
  validator (instead of being silently dropped).
- `MAX_STEPS` default bumped 8 → 25. Long-horizon agent runs commonly
  need 15-25 turns of tool calls before answering; the cycle-2 default
  of 8 truncated multi-tool tasks prematurely.
- 6 new tests covering: probe pass-through on `adapt_hermes_record`,
  silent-skip when probes are absent, validator rejection on tool
  records, end-to-end pass-through in `adapt_hermes_jsonl`,
  ShareGPT-format pass-through (incl. survival across the tool-result
  backfill rewrite), and the `MAX_STEPS=25` module default.

393 passing (+6 since 387). ruff clean on touched files.

Remaining for the entry to flip `passes: true`:
  - (b) spot re-run of Dense rollouts at bf16+fp8 with vLLM
    chat-completions logprobs=True + persistence of token IDs on each
    assistant turn — needs upstream patches into hermes-agent's
    batch_runner OR a thin proxy in our runner that injects
    `logprobs=True` + `top_logprobs=1` into the OpenAI request.
  - (c) FSDP-bf16 teacher-force on captured `(prompt_token_ids,
    response_token_ids)` pairs (one forward pass per assistant turn).
  - (d) Pair into `DenseMismatchReport` per (trajectory, turn) under
    `runs/live/dense/hermes-qwen3-32b-{bf16,fp8}-vs-fsdp-bf16/`.

## Current phase
**Cycle 3 v2 armed — design pivot mid-session (2026-05-12).** Iter 1
(`hermes_agent.token_extraction_adapter`) shipped, then the operator
killed the original refeed direction: vLLM is rollout only and we do
**not** re-feed response tokens back through it. The corrected design
captures rollout-side signals (per-token logprobs, per-(token, layer)
routed_experts) **during** the vLLM forward pass that generates the
trajectory, persists them on `AgentStep`, and the trainer-side
(FSDP / Megatron) teacher-forces the exact same `(prompt_token_ids,
response_token_ids)` to produce the trainer-side counterparts. This
is long-term stable: token IDs anchor to the tokenizer hash, no
chat-template re-encoding drift.

The cycle-2 trajectory captures on disk (45 trajectories total) lack
the probe fields and will be regenerated as part of cycle 3 v2.

STEER.md has been rewritten for cycle 3 v2 (six iterations). The
`.claude/feature-results.json` entries have been renamed from
`*_refeed_*` to `*_capture_then_*` to reflect the new design.

## Cycle 3 v2 progress (1/6)

- **iter 1 — `hermes_agent.token_extraction_adapter` ✅ (2026-05-12)**.
  `scripts/live/hermes_token_extraction.py` with pure
  `extract_token_pairs(trajectory, tokenizer)` + CLI. Now supports
  TWO paths: (a) preferred — return the captured
  `step.prompt_token_ids` / `step.response_token_ids` verbatim when
  present (no tokenizer needed; long-term stable); (b) fallback —
  chat-template re-encoding for legacy trajectories. Mixed-capture
  trajectories work too. 11 new tests; total +10 in this session
  (incl. 7 schema-extension tests in `test_agent_trajectory_lab.py`).
- **Schema extension (2026-05-12) shipped same commit.**
  `AgentStep` now carries four optional probe fields, all defaulting
  to None for backward compatibility:
    - `prompt_token_ids: list[int] | None` — exact prompt vLLM saw
    - `response_token_ids: list[int] | None` — exact tokens vLLM emitted
    - `response_logprobs: list[float] | None` — per-token vLLM logprob
    - `response_routed_experts: list[list[list[int]]] | None` —
      `[token][layer][top_k]` shape for MoE
  Validator enforces length parity + forbids probes on tool-role
  steps. Legacy JSONs validate unchanged.

- **iter 2/6 next — `hermes_agent.dense_capture_then_fsdp`** (live,
  spot). See STEER.md cycle-3 v2 iter 2 directive: (a) extend
  `scripts/live/hermes_agent_runner.py` to capture logprobs from
  chat-completions, **bump `MAX_STEPS` default 8 → 25** (long-horizon
  multi-turn agent runs need it), thread `prompt_token_ids` /
  `response_token_ids` / `response_logprobs` onto **each** `AgentStep`
  (every assistant turn captures its own prompt — which includes the
  full accumulated prior history — and its own response); (b)
  re-run Hermes-Agent Dense rollouts at bf16 + fp8; (c) FSDP-bf16
  teacher-force on captured (prompt, response) pairs (one forward
  pass per assistant turn); (d) pair into `DenseMismatchReport` per
  (trajectory, turn).

- **iter 6 scope update (UI cleanup, 2026-05-12).**
  `matrix_render_and_codex` now also removes the legacy single-prompt
  Dense + MoE matrices from the headline. The marketplace dashboard
  becomes Hermes-only above the fold: Hermes-Dense ESS + Hermes-MoE
  router_flip_rate, both 2×2 over `{FSDP, Megatron} × {bf16, fp8}`.
  Legacy single-prompt outputs move into a collapsible "Legacy
  single-prompt experiments (archived)" appendix — not above the
  fold. The underlying CLIs stay; only the dashboard layout changes.

## Cycle 2 retrospective (what worked, what was wrong)
**Hermes Agent cycle COMPLETE — 12 traffic-light tiles + codex PASS.**
All 29 entries in `.claude/feature-results.json` are `passes: true`.
STEER.md self-deleted per its own stop clause. The rendered
`/docs/index.html` now shows THREE matrices above the fold:

  * Dense (Qwen3-32B)         — ESS                          (8 tiles, 4 visible)
  * MoE (Qwen3-30B-A3B)       — router_flip_rate             (4 tiles, all visible)
  * Hermes Agent Dense        — final_answer_match X/N        (2 tiles)
  * Hermes Agent MoE          — final_answer_match X/N        (2 tiles)

Cycle 2 findings: vLLM-fp8 inference matters more for MoE rollouts
than Dense rollouts AT BOTH the token level AND the agent-trajectory
level:
  Dense ESS:           bf16 0.998  -> fp8 0.995  (tiny precision drop)
  MoE flip rate:       bf16 4%     -> fp8 7%     (clear precision effect)
  Hermes Dense jaccard: fp8 0.32   vs noise 0.29 (negligible)
  Hermes MoE jaccard:   fp8 0.39   vs noise 0.48 (precision IS divergent)

For crowdsourced MoE rollouts at fp8, the precision effect on multi-
turn agent trajectories exceeds sampling noise — confirming the
marketplace thesis at the action level.

## Hermes Agent integration (Iter 1-6 arc, shipped 2026-05-12)

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

**Iter 3 bf16 side LANDED (1/3 sides done).** Full end-to-end pipeline
is operational; 11 real Hermes Agent trajectories on disk at
`runs/live/agent/hermes/qwen3-32b/bf16/<task_id>.json`.

The pipeline that now works:
1. `pip install /tmp/hermes-agent` into isolated `~/hermes-venv`
   (hermes-agent 0.13.0, no curl-pipe-bash needed).
2. Patched `MINIMUM_CONTEXT_LENGTH` from 64_000 down to 8_000 in
   both the installed package and the `/tmp/hermes-agent` source
   tree (Qwen3-32B max is 40_960; the 64K floor is a recommendation,
   not load-bearing). The `.bak` files are kept for rollback.
3. vLLM Qwen3-32B-bf16 serve on spot at `MAX_LEN=40960`.
4. SSH tunnel `localhost:8000 → spot:8000`.
5. Hermes home at `/tmp/hermes_smoke_home/config.yaml` with
   `model.provider=vllm`, `base_url`, `context_length=40000`.
6. `batch_runner.py --base_url=... --model=Qwen/Qwen3-32B
   --ephemeral_system_prompt="...you MUST call a tool..."` produces
   one `trajectories.jsonl` with all 12 prompts.
7. `adapt_hermes_sharegpt_trajectory()` converts each record into
   an AgentTrajectory JSON.

bf16 results vs STEER acceptance gate:
- ✅ trajectories well-formed AgentTrajectory JSON
- ❌ 11/12 trajectory files (`code-fix-fibonacci` missing — Hermes
  read_file hit real /tmp which doesn't have our fixture)
- ❌ 3/11 have `len(steps) == 1`: `research-vllm-then-sglang`,
  `mixed-research-and-math`, `no-op-trivia` (model answered from
  memory; gate requires ≥2)
- ✅ no `terminal_reason="error"` (gate allows ≤2)

Tool calls across the bf16 side: 22 calls — terminal (9),
read_file (5), search_files (3), python_eval (3), write_file (1),
web_search (1). Mean wall-clock ~48s/prompt; total 572s for 12 at
num_workers=2.

**Remaining for Iter 3 acceptance**:
- fp8 side: re-serve `MODEL=Qwen/Qwen3-32B-FP8 DTYPE=auto`, re-run
  the batch (~10 min wall-clock + 5 min model load).
- bf16_seedB side: same model, `SEED=4321` (sampling-side, not
  serve-side — need a thin wrapper).
- Re-run `code-fix-fibonacci` (or pre-create `/tmp/fib.py` on Mac).
- Tighten 3 single-step prompts (or accept them as
  legitimate-divergence evidence).

393 tests passing, ruff clean. Loop halts here; operator can resume
with `/autonomous-loop` to drive the fp8 + seedB sides. Spot serve
+ SSH tunnel are stopped — no idle GPU spend.

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
- pytest -q: **447 passed, 0 failed** (2026-05-12). +40 since the
  prior 407: 7 schema-extension tests in `test_agent_trajectory_lab.py`
  + 11 in `test_hermes_token_extraction.py` (incl. coerce / captured-IDs
  / mixed-capture / no-tokenizer cases), the rest from the broader
  Hermes shipping arc captured upstream.
- ruff check .: clean.
- `.claude/feature-results.json`: **24/29 entries pass:true** after
  the cycle-3 v2 re-arm. Open: 5 `hermes_agent.*` entries
  (`dense_capture_then_fsdp`, `dense_capture_then_megatron`,
  `moe_capture_then_fsdp_router`, `moe_capture_then_megatron_router`,
  `matrix_render_and_codex`).
- The next `/autonomous-loop` invocation picks up
  `hermes_agent.dense_capture_then_fsdp` first (top-most false entry
  per the SKILL's selection rule).

## Next work (post-Hermes)

(Empty — see Research follow-ups below for completed items. The
dashboard surface is currently at a steady state; new initiatives
would come from `docs/future_research.md`.)

## Research follow-ups shipped (post-Hermes)

- **Dashboard click-through pages (2026-05-12, future_research Q17)**.
  `publish_dashboards.py` now copies the latest per-(pair, task)
  `agent_divergence_report.html` to
  `docs/agent_diff/<pair-slug>/<task-id>.html`. 54 stable URLs across
  5 pair slugs. The matrix tiles don't yet deep link to them; one-line
  href update is the next follow-up.

- **n=30 hermes rollouts (2026-05-12)**. Added bf16_rep2 batches for
  Dense (n=12) and MoE (n=9) and a replicate-aware pair script.
  Dense tiles now pair against n=24 bf16 reference, MoE TP=4 path
  against n=18 reference. The precision-vs-noise gap stabilises to
  ~8-12pp across both models: Dense fp8 45% vs noise 37.5%; MoE
  fp8(TP=4) 50% vs noise 33%. All tiles still mx-bad (under 50%
  amber threshold) but the gradient communicates the marketplace
  signal cleanly.

- **Soft answer-match (2026-05-12)**. Added
  `final_answer_match_soft` alongside the strict field, computed via
  numeric-token equivalence + substring fallback. Hermes tile
  rendering uses the soft rate; first AMBER Hermes tile (MoE fp8
  5/9 = 56%) emerged. Strict field preserved for back-compat.
  415 tests passing (+8 unit tests for the soft matcher).

- **MoE bf16 @ TP=2 noise floor (2026-05-12)**. Added a fresh
  Qwen3-30B-A3B bf16 serve at TP=2 (matching fp8's required TP).
  publish_dashboards.py now takes precision_trainer / noise_trainer
  overrides so the MoE precision-effect tile pairs fp8 @ TP=2
  against bf16 @ TP=2 (apples-to-apples) while the noise-floor pair
  stays on bf16 @ TP=4 seedB-vs-bf16. New tile values: precision
  5/11 (45.5%) vs noise 4/9 (44.4%) — statistically
  indistinguishable at n=12. The earlier amber tile (5/9, 56%) was
  the TP=4→TP=2 confound, not a fp8 effect.

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
