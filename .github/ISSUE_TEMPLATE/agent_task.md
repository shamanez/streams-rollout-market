---
name: Agent implementation task
about: Task packet for Codex or Claude Code
labels: agent-task
---

# Task: <imperative title>

## Context
This repo is a rollout marketplace and rollout validity layer for agentic RL. It is not a trainer.

## Goal

## Non-goals
- Do not implement PPO, GRPO, RLOO, DPO, FSDP, Megatron, optimizer, or gradient-step logic.

## Files to read first
- AGENTS.md
- README.md
- docs/why_rollout_market.md
- docs/mismatch_observatory.md

## Required changes

## Acceptance criteria

## Verification commands
```bash
pytest -q
ruff check .
python examples/local_worker_demo.py
```
