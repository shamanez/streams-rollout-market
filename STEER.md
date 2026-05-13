# STEER.md — Operator Redirect

Seeded at 2026-05-13T04:24:22Z from Plan Mode (this conversation).

Stop clause: when every entry in `.claude/feature-results.json` has
`passes:true`, delete this STEER.md as the final commit.

---

# Initiative: Remote-vLLM marketplace contract (centralized capture)

## Why

Cycle FP8 proved we can collect ESS + router_flip_rate end-to-end
**when both the vLLM rollout server and the capture proxy run on
the same box** (the spot). The marketplace's actual use case is
different: a third-party operator runs vLLM on their own GPU host,
exposes an OpenAI-compatible HTTP endpoint, and we (the
marketplace coordinator) drive the rollouts and capture the probes
from one central location — without being able to install or run
anything inside the operator's environment beyond the bash script
we hand them at onboarding.

The metric numbers should be identical to cycle FP8's local row
(same model, same precision, same code path) — the goal here is to
**prove the plumbing**: the contract holds over an HTTP boundary
with capture moved client-side.

## Global directives

* MoE only. Qwen3-30B-A3B bf16. No dense.
* Free-tier rule + spot-only rule still apply. The "remote
  operator" for this cycle is *our own spot* with an SSH tunnel —
  enough to prove the contract without involving anyone else.
* New code lives in `scripts/live/`. No new trainer logic.
* The capture proxy (`logprob_capture_proxy.py`) stays in the repo
  as the "co-located" path; client-side capture is an *alternative*
  for the remote case, not a replacement.
* Final dashboard adds a "remote-rollout" row alongside the
  existing local-rollout row so reproducibility is visible.

## Acceptance gates

* `pytest -q && ruff check .` green at the end of every iteration.
* Final dashboard shows the remote-rollout row populated with the
  same (rollout, trainer) flip-rate cells as the local row,
  agreeing to within 1 pp.

## Iterations

### 1. bundle_operator_serve_kit: package the serve script for a remote operator

Bundle exactly what a third-party GPU operator needs to start
serving Qwen3-30B-A3B with the `routed_experts` HTTP shim:
`vllm_serve_moe_dev.sh`, `patch_vllm_routed_experts_http.py`, and
a short `OPERATOR_QUICKSTART.md` that lists the prerequisites
(Python 3.12, CUDA driver, a venv recipe) and the three commands
they actually run (create venv, apply patch, source venv + serve).
Write a `scripts/live/operator_kit/` directory or a `make
operator-kit` target that produces a self-contained tarball at
`/tmp/operator_kit.tgz`. Acceptance: `tar tzf /tmp/operator_kit.tgz`
lists at least the three files; the README explains why each step
is needed.

### 2. client_side_capture_driver: new driver that extracts probes from response JSON

Write `scripts/live/remote_capture_driver.py` — same OpenAI tool-
using loop as `minimal_agent_driver.py` but reads
`choices[0].routed_experts`, `prompt_token_ids`,
`choices[0].token_ids`, and `choices[0].logprobs.token_logprobs`
directly from each chat completion response and emits one line per
assistant turn to a local sidecar JSONL — same shape as the
existing `logprob_capture_proxy.py` writes (so downstream
`merge_probes_into_trajectories.py` consumes it unchanged).
Acceptance: a unit test under `tests/test_remote_capture_driver.py`
mocks a chat completion response and verifies the sidecar JSONL
shape matches `extract_probes`'s output schema field-for-field.

### 3. tunnel_smoke_against_spot: SSH-forward the spot's :8000 and verify

Set up an SSH local port-forward (`ssh -fNL 8001:localhost:8000
my-vllm-spot-instance`) so the laptop's `localhost:8001` reaches
the spot's vLLM. Start `vllm_serve_moe_dev.sh` on the spot, drive
the new client-capture driver against `http://localhost:8001` from
the laptop on one task (`no-op-trivia`), and confirm the resulting
sidecar JSONL is byte-equivalent to what the proxy would have
written (modulo timestamps and request_ids). Acceptance: a single
trajectory captured fully client-side, with `response_routed_experts`
shape `[gen_tokens, 48, 8]` and `response_logprobs` length =
`response_token_ids` length.

### 4. drive_6_task_batch_remote_style: capture the cycle-FP8 task set client-side

Re-run the 6-task batch from cycle FP8 (4 hermes + 2 cwc) through
the SSH tunnel, captured client-side. Write trajectories to
`runs/live/agent/remote/qwen3-30b-a3b/bf16/` so they don't trample
the existing local-rollout dir. Build a corresponding
`/tmp/rollouts_remote_bf16.json`. Acceptance: 6 trajectory JSONs,
each with non-null `response_routed_experts` on every assistant
turn.

### 5. teacher_force_remote_rollouts: FSDP + Megatron over the remote batch

Push `/tmp/rollouts_remote_bf16.json` through both
`run_fsdp_moe_reference.py` and `megatron_moe_reference_launch.sh`
on the spot (the same scripts cycle FP8 used; trainer is always
bf16). Reuse `HF_FLAT_REUSE=~/hf-flat-qwen3-30b-a3b-bf16` so the
Megatron container starts in under a minute. Outputs:
`/tmp/trainers_fsdp_moe_remote_bf16.json` and
`/tmp/megatron_router-remote_bf16.json`. Acceptance: both written,
shape matches the local-rollout outputs of cycle FP8 iters 5+6.

### 6. pair_remote_results: pair the 2 remote × trainer combos

Run `pair_hermes_moe_reports.py` twice — once for FSDP, once for
Megatron — against the remote rollouts and trainer outputs.
Engine label `hermes-qwen3-30b-a3b-bf16-remote` so the dashboard
can tell the rows apart from the existing local-rollout cells. Out
under `runs/live/router/hermes-qwen3-30b-a3b-bf16-remote-vs-...`.
Acceptance: ~12 `router_mismatch_report.json` files written; each
has `router_flip_rate` populated.

### 7. dashboard_add_remote_row: render the remote-rollout row on the matrix

Extend `_render_hermes_router_matrix` (or add a sibling section) to
include a second matrix labelled "Remote-served rollouts (over
HTTP, client-side capture)" with one row per trainer ×
`hermes-qwen3-30b-a3b-bf16-remote` column. Republish. Acceptance:
`docs/index.html` shows BOTH the existing local matrix and a new
remote matrix; the remote-rollout flip rates agree with the local-
rollout ones to within 1 pp on the same trainer (a strong
plumbing-consistency check). Tests + ruff green.
