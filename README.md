# streams-rollout-market

A trainerless rollout marketplace and rollout validity layer for agentic RL.

Cheap rollouts are useless unless they are trainable. This project makes remote rollouts trainable by enforcing policy pins, token/logprob contracts, and an off-policyness budget before an external RL trainer consumes them.

## What this is

`streams-rollout-market` is a reference protocol and implementation for decentralized rollout supply:

- workers register capabilities and run environment + inference loops
- PolicyRegistry publishes versioned policy manifests
- dispatcher leases policy-pinned rollout jobs
- workers submit token IDs, behavior logprobs, rewards, masks, and traces
- OPBC scores mismatch/staleness/quality debt
- LiveStore serves only accepted or explicitly correctable groups
- external trainers pull groups through a client API

## What this is not

This repository is not a trainer. It does not implement PPO, GRPO, RLOO, DPO, FSDP, Megatron, optimizer state, or gradient steps. A tiny fake trainer may exist under tests/examples only to prove the API works.

## Public wedge: Mismatch Observatory

The first thing to show is the Rollout Mismatch Observatory:

1. endpoint identity gap probe for free/cheap OpenAI-compatible endpoints
2. controlled dense mismatch lab with the same checkpoint served under different engines/precision
3. small MoE router mismatch lab
4. market simulation where OPBC accepts, corrects, quarantines, or rejects worker output

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
python examples/local_worker_demo.py
```

## Repository layout

```text
src/rollout_market/
  contracts.py              Pydantic schemas
  opbc.py                   off-policyness budget metrics and decisions
  mismatch_metrics.py       reusable mismatch/materiality metrics
  registry.py               policy manifest reference store
  livestore.py              local rollout group store
  dispatcher.py             assignment and scheduler skeleton
  verifier.py               sample integrity hooks
  worker.py                 worker runtime skeleton
experiments/mismatch_observatory/
  README.md                 demo ladder and reproduction guide
  metrics_contract.json     stable JSON output contract
  configs/                  endpoint/dense/MoE config templates
docs/
  why_rollout_market.md
  mismatch_observatory.md
  opbc_metrics.md
  evidence_map.md
  agent_task_packets.md
```

## One-sentence pitch

A decentralized rollout market where the unit of supply is not a token, but a verified, policy-pinned, mismatch-accounted trajectory group that an external RL trainer can safely use.
