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
| `logprob_capture_proxy.py` | OpenAI proxy (runs on the spot alongside vLLM and Hermes). Listens on `localhost:8001`, forwards `/v1/chat/completions` to vLLM on `localhost:8000`, rewrites the request to add `logprobs=True, top_logprobs=1, return_token_ids=True` (and once setup-23 lands, `chat_template_kwargs.enable_thinking=false`), and appends each turn's probes to a JSONL sidecar keyed by `(prompt_index, turn_idx)` — both auto-derived from the request body so hermes-agent runs against it unchanged. |
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

## Topology (the actual working setup)

**Everything except SSH and dashboard rendering runs on the spot.** The
laptop is only an SSH terminal. Hermes-Agent, the logprob capture
proxy, vLLM, the merger, the filler, the builder, the trainer-side
teacher-force, and the pair script — all run on the spot.

```
                ┌──────────────────────────────────────────────────┐
                │ spot (my-vllm-spot-instance, g6e.12xlarge)       │
                │                                                  │
   laptop ─ssh→ │ hermes  → :8001 (proxy)  → :8000 (vLLM)          │
   terminal     │                  │                               │
                │                  ├─→ /tmp/hermes_probes_*.jsonl  │
                │                                                  │
                │ merge → fill → build → torchrun (FSDP/Megatron)  │
                │           │                                      │
                │           └─→ runs/live/agent/hermes/...         │
                │           └─→ /tmp/rollouts_*.json               │
                │           └─→ /tmp/trainers_*.json               │
                │                                                  │
                │ pair → runs/live/{dense,router}/hermes-.../      │
                └──────────────────────────────────────────────────┘
                                       │
                                       │ (rsync runs/ back to laptop
                                       │  ONLY for local dashboard view)
                                       ▼
                              python scripts/live/publish_dashboards.py
                              → docs/index.html
```

The proxy and vLLM both bind `127.0.0.1` only — never expose
externally. SSH port forwarding is **not** part of the data path.

## End-to-end Hermes capture pipeline (single host: the spot)

```
ssh my-vllm-spot-instance

# 1. bring up vLLM at 131K context (per HERMES_INSTALL.md step 3)
tmux new-session -d -s vllm \
  'MODEL=Qwen/Qwen3-32B MAX_LEN=131072 bash ~/serve_for_capture.sh \
     2>&1 | tee /tmp/vllm_serve.log'
# wait for /v1/models to respond (~2 min Dense, ~3 min MoE)

# 2. capture proxy on :8001
cd ~/streams-rollout-market
tmux new-session -d -s proxy \
  'python scripts/live/logprob_capture_proxy.py \
       --upstream http://localhost:8000 \
       --port 8001 \
       --sidecar /tmp/hermes_probes_dense_bf16.jsonl \
       --host 127.0.0.1'

# 3. point Hermes at the proxy
hermes config set model.base_url 'http://localhost:8001/v1'

# 4. run the 12-task suite
cd ~/streams-rollout-market
hermes chat -q "$(jq -r '.[0].task_text' scripts/live/agent_tasks_hermes.json)" \
    --max-turns 25 --yolo --accept-hooks --quiet
# (loop over all 12 tasks; or batch via a wrapper script — to be added)

# 5. stitch trajectories from sharegpt + sidecar (still on spot)
python scripts/live/merge_probes_into_trajectories.py \
    --sharegpt /path/to/hermes_sharegpt.jsonl \
    --probes /tmp/hermes_probes_dense_bf16.jsonl \
    --tasks scripts/live/agent_tasks_hermes.json \
    --model-id Qwen/Qwen3-32B \
    --engine-label hermes-qwen3-32b-bf16 \
    --engine-fingerprint sha256:hermes-qwen3-32b-bf16-tp4-l40s \
    --out-dir runs/live/agent/hermes/qwen3-32b/bf16/

# 6. fill missing prompt_token_ids via chat-template
python scripts/live/fill_prompt_token_ids.py \
    --trajectory-dir runs/live/agent/hermes/qwen3-32b/bf16/

# 7. flatten for the trainer
python scripts/live/build_hermes_dense_input.py \
    --trajectory-dir runs/live/agent/hermes/qwen3-32b/bf16/ \
    --precision-class bf16

# 8. teacher-force the captured (prompt, response) pairs (still on spot)
#    tear down vLLM first to free the GPUs for FSDP:
tmux kill-session -t vllm
tmux kill-session -t proxy
torchrun --nproc-per-node 4 ~/run_fsdp_reference.py

# 9. pair into DenseMismatchReport per (trajectory, turn)
python scripts/live/pair_hermes_dense_reports.py \
    --index /tmp/hermes_dense_index_bf16.json \
    --trainers /tmp/trainers.json \
    --trajectory-dir runs/live/agent/hermes/qwen3-32b/bf16/ \
    --out-root runs/live/dense/hermes-qwen3-32b-bf16-vs-fsdp-bf16/ \
    --trainer-engine-label fsdp-bf16

# 10. (laptop) sync the produced runs/ back and re-render the dashboard
exit
rsync -av my-vllm-spot-instance:~/streams-rollout-market/runs/ runs/
python scripts/live/publish_dashboards.py
python -m http.server -d docs 8000 &
open http://localhost:8000/
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
