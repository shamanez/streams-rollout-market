# scripts/live/ — live-experiment runbook

Cycle 3 v2 — **probe-at-rollout**. The architecture is intentionally
simple: one agent, one rollout engine (vLLM), two precisions per
model, two models, two trainer references. **vLLM is rollout only.**
No refeed.

```
Hermes Agent ── vLLM-hosted Qwen3 (bf16 | fp8)
                     │
                     ├── per-token logprobs                        (Dense)
                     └── per-(token, layer) routed_experts          (MoE)
                                ↓
                  AgentTrajectory JSON on disk
                                ↓
       captured (prompt_token_ids, response_token_ids)
                                ↓
              FSDP teacher-force ─┐
                                  ├─→ DenseMismatchReport / RouterMismatchReport
              Megatron teacher-force ─┘
                                ↓
                       Hermes-only dashboard
```

## Inventory

| File | Role |
|------|------|
| `hermes_agent_runner.py` | Driver. Generates trajectories via the real Hermes Agent CLI against a vLLM OpenAI-compatible endpoint; captures per-token logprobs (and per-(token, layer) routed_experts on MoE) onto each `AgentStep`. |
| `hermes_token_extraction.py` | Pure helper: lifts `(prompt_token_ids, response_token_ids)` pairs from each `AgentStep` (preferred — exact IDs vLLM saw) or falls back to chat-template re-encoding for legacy trajectories. |
| `agent_tasks_hermes.json` | Task fixtures the Hermes Agent CLI consumes (12 multi-turn tool-using tasks). |
| `run_fsdp_reference.py` | Trainer-side teacher-force, Dense (FSDP-bf16). Reads `/tmp/rollouts.json` (one entry per assistant turn), writes per-token `trainer_logprobs`. |
| `run_fsdp_moe_reference.py` | Same, MoE. Emits FSDP-side router decisions. |
| `run_megatron_reference.py` + `megatron_reference_launch.sh` | Megatron-bf16 Dense teacher-force inside `slimerl/slime` docker. |
| `run_megatron_moe_reference.py` + `megatron_moe_reference_launch.sh` | Megatron-MoE-bf16 teacher-force. |
| `publish_dashboards.py` | Re-renders `docs/index.html` from `runs/live/*` reports. |
| `serve_dashboards.py` | One-command local server for the rendered dashboards. |
| `marketplace_real.py` | Marketplace-stack integration runner (validators + OPBC + LiveStore + TrainerClient). Independent of the cycle-3 mismatch flow. |
| `megatron_qwen3_32b_launch.sh` / `megatron_qwen3_32b_runner.sh` | Megatron HF→torch-dist conversion (one-time setup; see `megatron_convert_qwen3_moe.md`). |
| `serve_qwen_for_agent.sh` / `serve_qwen_for_agent_sglang.sh` / `serve_on_spot.sh` | Bring up the vLLM OpenAI endpoint the Hermes Agent calls into. |

## Cells (the marketplace matrix)

```
            │  bf16     │  fp8
────────────┼───────────┼──────────
FSDP        │   tile    │   tile
Megatron    │   tile    │   tile
```

…one such 2×2 per model (Dense Qwen3-32B + MoE Qwen3-30B-A3B). Each
tile is `mean(ESS)` (Dense) or `mean(router_flip_rate)` (MoE)
computed over the agent's response tokens, aggregated across all
assistant turns in the trajectory suite.

## Extending to other models / devices (future)

The design generalises by adding cells: a new (model, precision)
just runs the same `hermes_agent_runner.py` against a different
vLLM-hosted endpoint and the trainer-side teacher-force scripts pick
up the new `runs/live/agent/hermes/<model_slug>/<precision>/`
directory automatically. No core code changes required.

See also:

- `hermes_integration_notes.md` — design rationale for the Hermes
  Agent integration.
- `megatron_convert_qwen3_moe.md` — Megatron checkpoint conversion
  runbook.
- `../../CLAUDE.md` — project-level architecture + firewall rules.
