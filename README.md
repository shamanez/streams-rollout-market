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

The Rollout Mismatch Observatory ships as a live, graph-first dashboard:

- **[/docs/index.html](docs/index.html)** — two matrices, eight tiles,
  one question:

  > Does engine + precision + device change inference-time behaviour
  > enough to matter for crowdsourced MoE rollouts?

  *Yes for MoE, no for dense.* At matched precision (bf16/bf16) vLLM-as-
  rollout vs FSDP/Megatron-as-trainer-reference diverges ~4% per
  (token, layer) on router decisions; fp8 inference adds 2-3% on top.
  Dense ESS stays >0.99 across all four cells.

The four labs behind those tiles:

1. dense mismatch lab — same checkpoint, vLLM rollout vs FSDP/Megatron
   teacher-force, ESS / |Δlogp| / clipped-fraction reported per pair
2. MoE router mismatch lab — `output_router_logits=True`, paired with
   vLLM `enable_return_routed_experts=True`, router_flip_rate per layer
3. agent-trajectory lab — first-divergence-step, tool-call Jaccard,
   answer-match across rollout vs trainer-served generations
4. marketplace simulation — OPBC accepts/corrects/quarantines/rejects
   honest/noisy/stale/toxic worker profiles

The endpoint-coverage probe is retired from the public surface (still
shipped as `src/rollout_market/observatory/endpoint_probe.py`).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q                                            # 347 passing
python examples/local_worker_demo.py
```

### View the dashboard locally

```bash
python scripts/live/publish_dashboards.py            # re-renders docs/
python -m http.server -d docs 8000 &
open http://localhost:8000/
```

Live experiments run on the spot instance; see the
**[How to run](CLAUDE.md#how-to-run)** section in `CLAUDE.md` for the
dense + MoE matrix runbook.

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
