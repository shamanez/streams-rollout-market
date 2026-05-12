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
| `logprob_capture_proxy.py` | Local OpenAI proxy. Listens on a port, forwards `/v1/chat/completions` to the SSH-tunnelled vLLM upstream, rewrites the request to add `logprobs=True, top_logprobs=1, extra_body.return_tokens_as_token_ids=True, extra_body.return_routed_experts=True`, and appends each turn's probes to a JSONL sidecar keyed by `(prompt_index, turn_idx)` — both auto-derived from the request body so hermes-agent runs against it unchanged. |
| `merge_probes_into_trajectories.py` | Stitcher. Reads the hermes-agent ShareGPT JSONL + the proxy's probe sidecar, joins them by hash of the first user message, and emits one probe-bearing `AgentTrajectory` JSON per task. |
| `fill_prompt_token_ids.py` | Re-derives `prompt_token_ids` via chat-template encoding for assistant steps that captured `response_token_ids` but not the prompt side (the realistic proxy-captured shape — chat-completions doesn't surface prompt IDs). |
| `build_hermes_dense_input.py` | Aggregator. Walks a trajectory dir; flattens probe-bearing turns into `/tmp/rollouts.json` (consumed by `run_fsdp_reference.py` / `run_megatron_reference.py`) + an index file the pair scripts use to join back. |
| `pair_hermes_dense_reports.py` | Joins the index + `/tmp/trainers.json` from the Dense trainer into one `DenseMismatchReport` per `(trajectory, assistant_turn)`. |
| `pair_hermes_moe_reports.py` | MoE counterpart. Joins the index + `/tmp/trainers_moe.json` into one `RouterMismatchReport` per turn. |
| `hermes_token_extraction.py` | Pure helper: lifts `(prompt_token_ids, response_token_ids)` pairs from each `AgentStep` (preferred — exact IDs vLLM saw), with a partial-capture branch for proxy-captured trajectories and a chat-template fallback for legacy data. |
| `agent_tasks_hermes.json` | Task fixtures the Hermes Agent CLI consumes (12 multi-turn tool-using tasks). |
| `run_fsdp_reference.py` | Trainer-side teacher-force, Dense (FSDP-bf16). Reads `/tmp/rollouts.json` (one entry per assistant turn), writes per-token `trainer_logprobs`. |
| `run_fsdp_moe_reference.py` | Same, MoE. Multi-prompt mode (reads `/tmp/rollouts.json` list, writes `/tmp/trainers_moe.json`); single-prompt back-compat preserved. Emits FSDP-side per-(token, layer) router decisions. |
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

## End-to-end Hermes capture pipeline (laptop + spot)

```
# 1. (spot)   bring up vLLM serving the model under test
ssh my-vllm-spot-instance 'bash ~/serve_qwen_for_agent.sh'

# 2. (laptop) tunnel + capture proxy
ssh -L 8000:localhost:8000 my-vllm-spot-instance -N &
python scripts/live/logprob_capture_proxy.py \
    --upstream http://localhost:8000 \
    --port 8001 \
    --sidecar /tmp/hermes_probes.jsonl &

# 3. (laptop) run Hermes-Agent against the proxy
OPENAI_BASE_URL=http://localhost:8001/v1 \
    hermes batch_runner --dataset_file=hermes_tasks.jsonl \
    --batch_size=1 --run_name=qwen3-32b-bf16 --max_iterations=25

# 4. (laptop) stitch trajectories from sharegpt + sidecar
python scripts/live/merge_probes_into_trajectories.py \
    --sharegpt trajectory_outputs/qwen3-32b-bf16/trajectories.jsonl \
    --probes /tmp/hermes_probes.jsonl \
    --tasks scripts/live/agent_tasks_hermes.json \
    --model-id Qwen/Qwen3-32B \
    --engine-label hermes-qwen3-32b-bf16 \
    --engine-fingerprint sha256:hermes-qwen3-32b-bf16-tp4-l40s \
    --out-dir runs/live/agent/hermes/qwen3-32b/bf16/

# 5. (laptop) fill missing prompt_token_ids via chat-template
python scripts/live/fill_prompt_token_ids.py \
    --trajectory-dir runs/live/agent/hermes/qwen3-32b/bf16/

# 6. (laptop) flatten for the trainer
python scripts/live/build_hermes_dense_input.py \
    --trajectory-dir runs/live/agent/hermes/qwen3-32b/bf16/ \
    --precision-class bf16

# 7. (spot)   teacher-force the captured (prompt, response) pairs
scp /tmp/rollouts_hermes_bf16.json my-vllm-spot-instance:/tmp/rollouts.json
ssh my-vllm-spot-instance 'cd ~ && torchrun --nproc-per-node 4 \
    run_fsdp_reference.py'
scp my-vllm-spot-instance:/tmp/trainers.json /tmp/trainers_hermes_bf16.json

# 8. (laptop) pair into DenseMismatchReport per (trajectory, turn)
python scripts/live/pair_hermes_dense_reports.py \
    --index /tmp/hermes_dense_index_bf16.json \
    --trainers /tmp/trainers_hermes_bf16.json \
    --trajectory-dir runs/live/agent/hermes/qwen3-32b/bf16/ \
    --out-root runs/live/dense/hermes-qwen3-32b-bf16-vs-fsdp-bf16/ \
    --trainer-engine-label fsdp-bf16

# 9. (laptop) re-render the dashboard
python scripts/live/publish_dashboards.py
```

The MoE side substitutes `run_fsdp_moe_reference.py` (writes
`/tmp/trainers_moe.json`) + `pair_hermes_moe_reports.py` and lands
reports under `runs/live/router/hermes-qwen3-30b-a3b-{bf16,fp8}-vs-…/`.

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
