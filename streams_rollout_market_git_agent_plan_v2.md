# streams-rollout-market: agent-ready git plan v2

## Project decision

This repository is a rollout marketplace / rollout validity layer for agentic RL. It is not a trainer.

The public wedge is the **Rollout Mismatch Observatory** plus the **Off-Policy Budget Controller (OPBC)**. A generic worker queue is not enough. The project is worth building only if it makes remote rollouts trainable: policy-pinned, token-faithful, logprob-complete, mismatch-accounted, and consumable by external trainers.

## Non-goals

- Do not implement PPO, GRPO, RLOO, DPO, FSDP, Megatron, optimizer state, or gradient steps.
- Do not build a payment/settlement layer in v0.
- Do not claim that one correction solves all mismatch sources.
- Do not accept raw text completions as training-ready trajectories.
- Do not allow mixed-policy GRPO groups by default.

## Repository surfaces

```text
src/rollout_market/
  contracts.py              # pydantic data contracts
  opbc.py                   # off-policy budget reports and decisions
  mismatch_metrics.py       # reusable logprob/ESS/materiality metrics
  registry.py               # policy registry reference implementation
  livestore.py              # valid group store
  dispatcher.py             # policy-aware worker/job routing
  verifier.py               # hashes, audits, receipts
  worker.py                 # worker runtime skeleton
experiments/mismatch_observatory/
  README.md                 # demo ladder and reproduction guide
  metrics_contract.json     # JSON output format for benchmark runs
  configs/*.yaml            # endpoint probe, dense lab, MoE lab configs
docs/
  why_rollout_market.md
  mismatch_observatory.md
  opbc_metrics.md
  agent_task_packets.md
  evidence_map.md
.github/ISSUE_TEMPLATE/
  agent_task.md
AGENTS.md
```

## AGENTS.md policy

Create and maintain `AGENTS.md` before assigning Codex or Claude Code tasks. It should be treated as an architecture firewall.

```markdown
# AGENTS.md

This repository implements a decentralized rollout marketplace and rollout validity layer for agentic RL.

It does not implement a trainer.

Do not add PPO, GRPO, RLOO, DPO, FSDP, Megatron, optimizer, gradient-step, or reward-model-training logic except as mocks under tests/ or examples/.

The core product is:
- worker registration and leases
- rollout dispatch
- policy versioning and weight-sync metadata
- LiveStore
- token/logprob contracts
- rollout verification
- off-policyness and training-inference mismatch telemetry
- trainer-facing client APIs
- Mismatch Observatory experiments

All accepted rollouts must preserve:
- token IDs
- sampled-token behavior logprobs
- tokenizer hash
- policy version
- checkpoint digest
- inference engine fingerprint
- precision and quantization class
- tool-token masks
- group membership

GRPO groups must be one-policy-snapshot by default. Mixed-version groups are invalid unless an explicit experimental contract is used.
```

## Task packet template

Every long-running Claude Code or Codex task should be written like this:

```markdown
# Task: <imperative title>

## Context
This repo is a rollout marketplace, not a trainer. Explain the relevant contract and why it matters.

## Goal
Define the specific deliverable.

## Non-goals
List what the agent must not touch.

## Files to read first
- AGENTS.md
- docs/architecture.md
- relevant source files

## Required changes
- bullet list

## Acceptance criteria
- bullet list with observable outputs

## Verification commands
pytest -q
ruff check .
python examples/local_worker_demo.py
```

## Phase 0 - Repo reset

### Task 0.1: Rewrite README around rollout validity
Context: The README must make it impossible to mistake this project for a trainer.
Goal: Rewrite the README to say the repo provides worker/trainer clients, contracts, LiveStore, PolicyRegistry, OPBC, and Mismatch Observatory.
Non-goals: Do not add training code.
Acceptance: A new reader can explain the project in five minutes; no section implies the repo owns PPO/GRPO.
Verification: `python -m pip install -e .[dev] && pytest -q`.

### Task 0.2: Add architecture firewall files
Required changes: Add `AGENTS.md`, `docs/non_goals.md`, `docs/why_rollout_market.md`, `.github/ISSUE_TEMPLATE/agent_task.md`.
Acceptance: Files clearly state no trainer, one-policy-snapshot group invariant, and token/logprob contract.

## Phase 1 - Contract-first core

### Task 1.1: Expand PolicyManifest and WorkerManifest
Goal: Add metadata fields for tokenizer_hash, model_config_hash, checkpoint_digest, precision_class, quantization_class, engine_contract_version, patch lineage, and allowed engines.
Acceptance: Fixtures validate; missing tokenizer_hash and checkpoint_digest fail.

### Task 1.2: Add RolloutJob and RolloutLease
Goal: Jobs should pin policy version and sampling config before any trajectory begins.
Acceptance: Lease has expiry, policy_version, required precision, max_context_tokens, group_size, and idempotency key.

### Task 1.3: Add RolloutGroup rejection reasons
Goal: Replace generic ValueError-style decisions with typed rejection reasons.
Acceptance: Mixed policy version, missing logprobs, missing token IDs, tokenizer mismatch, precision mismatch, and expired lease have distinct reasons.

## Phase 2 - Mismatch Observatory

### Task 2.1: Endpoint identity gap probe
Goal: Implement a CLI that calls OpenAI-compatible endpoints and records token/text/logprob contract coverage.
Non-goals: Do not make policy-gradient claims from uncontrolled endpoints.
Acceptance: Produces `runs/<timestamp>/endpoint_contract_report.json` with provider, model label, token IDs available, sampled logprobs available, top_logprobs available, seed support, and raw response hash.

### Task 2.2: Controlled dense mismatch lab
Goal: Run same checkpoint through reference forward and at least one serving backend; compute delta_t, log_ratio, ESS, E[w^2], clipped_fraction, max_abs_log_ratio, and top_1pct_gradient_mass.
Acceptance: Metrics contract is stable and results can render to HTML/JSON without requiring a trainer.

### Task 2.3: Small MoE router mismatch lab
Goal: Add a contract for router traces and an experiment plan for Qwen1.5-MoE-A2.7B or another small open MoE.
Acceptance: The design records rollout expert ids and compares them with training-side router ids; output includes router_flip_rate and token_expert_disagreement_rate.

## Phase 3 - Local market loop

### Task 3.1: LiveStore lifecycle
Goal: Implement accepted/correctable/quarantine/reject lifecycle with idempotency.
Acceptance: Duplicate group id is rejected; quarantined groups are retained but not served by default.

### Task 3.2: Trainer client API
Goal: Add a trainer-facing consumer API that pulls valid groups by policy_version, max_staleness, precision_class, and replay_tier.
Non-goals: No loss computation.
Acceptance: Example fake trainer pulls groups and prints batch metadata only.

### Task 3.3: Worker SDK
Goal: Add worker registration, heartbeats, policy lease fetch, group submission, and failure reporting.
Acceptance: Three fake workers can run concurrently in a local demo.

## Phase 4 - OPBC v0

### Task 4.1: OPBC decision reasons
Goal: Every group gets `train`, `train_with_correction`, `replay`, `quarantine`, or `reject` plus reasons.
Acceptance: Tests cover stale, missing-logprob, mixed-version, low-ESS, high-clipped-fraction, and clean groups.

### Task 4.2: Trainer feedback ingestion
Goal: External trainers can report ESS, clipped fraction, accepted/rejected, and current-policy logprob statistics after consuming a group.
Acceptance: Feedback updates per-engine/per-worker quality stats without changing trainer logic.

## Phase 5 - Remote/spot worker hardening

### Task 5.1: Heartbeat and lease timeout
Goal: Worker death or spot interruption should not corrupt group state.
Acceptance: Timed-out leases return to the dispatcher or are marked abandoned with audit trail.

### Task 5.2: k-of-n reload quorum design doc
Goal: Define how PolicyRegistry advances a version for a worker pool without global all-or-nothing failure.
Acceptance: Per-group policy pinning remains strict even if pool-level sync is quorum-based.

## Phase 6 - Launch demos

1. Endpoint identity gap dashboard.
2. Controlled dense mismatch dashboard.
3. Local marketplace simulation with toxic workers and OPBC decisions.
4. MoE router mismatch demo.

## The public README sentence

> Cheap rollouts are useless unless they are trainable. streams-rollout-market makes remote rollouts trainable by enforcing policy pins, token/logprob contracts, and an off-policyness budget before they reach any external trainer.
