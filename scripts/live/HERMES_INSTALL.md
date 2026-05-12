# Installing Hermes-Agent against a self-hosted vLLM Qwen3 endpoint

Verbatim, reproducible install + smoke-test of NousResearch Hermes-Agent
talking to our spot-instance vLLM at 131 K context. Followed verbatim on
2026-05-12 and confirmed working with both `Qwen/Qwen3-32B` (Dense, bf16)
and `Qwen/Qwen3-30B-A3B` (MoE, bf16).

## Topology (the actual working setup)

**Everything runs on the spot host.** The laptop is only an SSH
terminal — no laptop-side install, no SSH port forwarding for the data
path.

```
                    ┌─────────────────────────────────────────────┐
                    │  spot host  my-vllm-spot-instance           │
   laptop  ─ssh─→   │                                             │
   (terminal        │   Hermes-Agent ── http://localhost:8001 ──→ │
   only)            │   (~/.local/bin/hermes)         │           │
                    │                                 ▼           │
                    │                       logprob_capture_proxy │
                    │                       (scripts/live/...py)  │
                    │                                 │           │
                    │                                 ▼           │
                    │                       http://localhost:8000 │
                    │                       vLLM serve            │
                    │                       (Qwen3-32B / 30B-A3B) │
                    └─────────────────────────────────────────────┘
```

Without the proxy (smoke-test path), Hermes points directly at
`localhost:8000`. With the proxy (cycle-3-v2 probe capture path),
Hermes points at `localhost:8001` and the proxy forwards to vLLM on
`:8000`, writing per-turn probes to a sidecar JSONL file on the spot.

## 0. Clean slate (only if a previous install exists)

```bash
ssh my-vllm-spot-instance
rm -rf /tmp/hermes-agent ~/hermes-venv ~/.hermes ~/hermes_smoke_home
rm -f  ~/.local/bin/hermes ~/.local/bin/hermes-agent
```

## 1. Install Hermes-Agent (official one-liner)

Source: <https://hermes-agent.nousresearch.com/docs/getting-started/installation>

Prerequisites on the spot: only `git` and `curl`. The installer brings its
own `uv`, Python 3.11, Node v22, ripgrep, ffmpeg.

```bash
ssh my-vllm-spot-instance
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh \
    -o /tmp/hermes-install.sh
bash /tmp/hermes-install.sh --skip-setup
source ~/.bashrc           # picks up ~/.local/bin on PATH
hermes --help | head -5    # verification
```

What the installer leaves behind (per-user, no sudo):

| Path | Role |
|------|------|
| `~/.hermes/hermes-agent/` | Cloned Hermes-Agent repo + its venv |
| `~/.local/bin/hermes` | Launcher symlink |
| `~/.hermes/config.yaml` | Settings (model, provider, tools, hooks, …) |
| `~/.hermes/.env` | API keys, secrets |
| `~/.hermes/sessions/, cron/, logs/` | Runtime state |

## 2. Point Hermes at the self-hosted vLLM

```bash
ssh my-vllm-spot-instance
hermes config set model.default 'Qwen/Qwen3-32B'
hermes config set model.provider 'custom'
hermes config set model.base_url 'http://localhost:8000/v1'
hermes config set model.context_length 131072
hermes config set model.api_key 'dummy'
hermes config show | grep -A1 Model
```

Notes:

- `provider: custom` is the explicit name. The aliases `vllm`, `ollama`,
  `llamacpp` all map to `custom`, **but** the `vllm` alias did not work
  cleanly in our run — the resolver still asked for `OPENROUTER_API_KEY`.
  Stick with `provider: custom`.
- `api_key: dummy` is required even for keyless vLLM — Hermes' OpenAI
  client builder will refuse to construct without one.
- `context_length: 131072` is critical. Hermes' quickstart says the
  minimum is 64 000 tokens; leaving it unset triggers the auto-detect
  path which can mis-read the served context window.

For Qwen3-30B-A3B (MoE), only `model.default` changes:

```bash
hermes config set model.default 'Qwen/Qwen3-30B-A3B'
```

## 3. Serve Qwen3 at 131 K via `serve_for_capture.sh`

`scripts/live/serve_for_capture.sh` (already on the spot at
`~/serve_for_capture.sh`) handles:

- `--enable-auto-tool-choice --tool-call-parser hermes` (required for
  Hermes-Agent tool calling on the OpenAI shim).
- YaRN rope scaling when `MAX_LEN > 40 960` (Qwen3's native limit):
  injects `--hf-overrides '{"rope_scaling":{"rope_type":"yarn",
  "factor":4.0,"original_max_position_embeddings":32768}}'` and exports
  `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. Required for vLLM ≥ 0.20 — older
  `--rope-scaling` flag was removed.
- `--enable-return-routed-experts` auto-enabled on MoE Qwen3 models
  (`*A3B*`, `*MoE*`); silently OFF for Dense (Dense serves crash on init
  if the flag is set: `Qwen3Config has no attribute num_experts_per_tok`).

Launch (foreground via tmux for clean detach):

```bash
ssh my-vllm-spot-instance
tmux kill-session -t vllm 2>/dev/null || true
tmux new-session -d -s vllm \
  'MODEL=Qwen/Qwen3-32B   MAX_LEN=131072         bash ~/serve_for_capture.sh \
     2>&1 | tee /tmp/vllm_serve.log'
# MoE variant:
# tmux new-session -d -s vllm \
#   'MODEL=Qwen/Qwen3-30B-A3B TP=2 MAX_LEN=131072 bash ~/serve_for_capture.sh \
#      2>&1 | tee /tmp/vllm_serve.log'

# wait for readiness (~2 min for Dense, ~3 min for MoE):
for i in {1..50}; do
  curl -sf -m 3 http://localhost:8000/v1/models > /dev/null && \
    { echo READY; break; } || sleep 12
done
```

## 4. Smoke test — summarise a repo

```bash
ssh my-vllm-spot-instance
rm -rf ~/streams-rollout-market
git clone --depth 1 https://github.com/shamanez/streams-rollout-market.git \
    ~/streams-rollout-market

cd ~/streams-rollout-market
hermes chat \
  -q 'Read the top-level docs (CLAUDE.md, PROGRESS.md, AGENTS.md) of this repo using your file-reading tool, then summarize the repo in 5 bullet points and say what the main entrypoint or driver is.' \
  --max-turns 25 --yolo --accept-hooks --quiet
```

Flags that matter:

| Flag | Reason |
|------|--------|
| `-q` / `--query` | Single-query non-interactive mode (no TUI). |
| `--quiet` (`-Q`) | Suppress banner/spinner; emit only the final response. |
| `--yolo` | Skip dangerous-command approval prompts (required for headless). |
| `--accept-hooks` | Auto-approve any hook prompts from `config.yaml`. |
| `--max-turns 25` | Cap multi-turn tool-call loop (default 60 is fine too). |

Verified runs (2026-05-12):

| Model | TP | Wall-clock | Outcome |
|-------|----|-----------|---------|
| `Qwen/Qwen3-32B` bf16 | 4 | 63 s | Clean 5-bullet summary, identified `/autonomous-loop` as the driver. Multi-turn tool calls (file reads) fired correctly. |
| `Qwen/Qwen3-30B-A3B` bf16 | 2 | 1 658 s (DNF) | Pipeline OK (connected, 3 multi-turn tool calls fired, vLLM generated at 57-84 tok/s, `routed_experts` capture enabled). Run failed at turn 3 with `Context length exceeded: max compression attempts (3) reached` — Qwen3's default reasoning mode produced a `<think>` trace that exceeded even 131 K context. **The install is fine; the model needs `enable_thinking=false`.** |

### Qwen3 reasoning mode caveat (MoE especially)

Qwen3 (both Dense and MoE) ships with reasoning mode **on** by default and
emits `<think>...</think>` blocks before the answer. The Dense run above
returned in 63 s because its trace was modest; the MoE
(Qwen/Qwen3-30B-A3B) at default settings produced enough reasoning
content to fill 131 K context across 3 turns.

To disable reasoning at the request level, the OpenAI body must include::

```json
"chat_template_kwargs": {"enable_thinking": false}
```

Hermes-Agent does not currently thread that field through to vLLM.
Putting `/no_think` in the user message body is **not** equivalent —
Qwen3 reads it as plain user text. Two workable options for our pipeline:

1. **Use the on-spot capture proxy.** `scripts/live/logprob_capture_proxy.py`
   already rewrites every outbound `/v1/chat/completions` body for logprob
   capture; extending it to also set `chat_template_kwargs.enable_thinking=false`
   is one extra `setdefault` and keeps Hermes unchanged. Proxy runs on the
   spot host (`localhost:8001`), forwards to vLLM (`localhost:8000`).
2. **Serve-side override.** Add `--chat-template-content-format
   string` (or a custom chat-template file) on the vLLM `serve` line that
   strips the thinking-mode prefix unconditionally.

The capture proxy (option 1) is the right answer for our use case: the
same proxy is already load-bearing for cycle-3-v2 probe capture, so
threading `enable_thinking=false` through it costs nothing.

## 5. Where the cycle-3-v2 probe capture plugs in

The Hermes-Agent CLI does **not** forward `logprobs=True` or
`chat_template_kwargs` through to the upstream endpoint by default.
To capture per-token logprobs and (on MoE) per-(token, layer)
`routed_experts`, run `scripts/live/logprob_capture_proxy.py` **on the
spot**, between Hermes and vLLM (all three processes on the same host):

```bash
# 1. vLLM already up on localhost:8000 (per step 3)
# 2. start the proxy on localhost:8001, sidecar on /tmp/hermes_probes_*.jsonl:
ssh my-vllm-spot-instance
cd ~/streams-rollout-market
python scripts/live/logprob_capture_proxy.py \
    --upstream http://localhost:8000 \
    --port 8001 \
    --sidecar /tmp/hermes_probes_dense_bf16.jsonl \
    --host 127.0.0.1 &

# 3. point Hermes at the proxy:
hermes config set model.base_url 'http://localhost:8001/v1'
```

The proxy rewrites every outbound `/v1/chat/completions` body to set
`logprobs=True, top_logprobs=1, return_token_ids=True` (and once the
`hermes_agent.proxy_enable_thinking_off` patch lands,
`chat_template_kwargs.enable_thinking=False`). It then writes the
captured response metadata to a JSONL sidecar keyed by
`(prompt_index, turn_idx)` — both auto-derived from the request body
so no Hermes patching is needed.

`routed_experts` capture is server-side only — the
`--enable-return-routed-experts` flag on `vllm serve`
(`serve_for_capture.sh` already sets it for MoE models). The proxy
just lifts the resulting `prompt_routed_experts` /
`choices[i].routed_experts` fields off the response body.

Full downstream pipeline is in `scripts/live/README.md` (proxy →
merger → filler → builder → trainer-side teacher-force → pair). All
steps after the proxy also run on the spot; the laptop is only used
for SSH and (optionally) for re-rendering the dashboard locally.

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `vllm: error: unrecognized arguments: --rope-scaling` | vLLM ≥ 0.20 removed the flag. | Use `--hf-overrides '{"rope_scaling":{"rope_type":"yarn",...}}'` (already in `serve_for_capture.sh`). |
| `User-specified max_model_len (131072) is greater than the derived max_model_len (max_position_embeddings=40960)` | vLLM validates max-len before applying rope override. | Export `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` (already done by `serve_for_capture.sh` when YaRN is on). |
| `Qwen3Config has no attribute num_experts_per_tok` on Dense serve startup | `--enable-return-routed-experts` set on a Dense model. | Don't override `ROUTED_EXPERTS`; the script auto-disables for Dense. |
| `Provider resolver returned an empty API key. Set OPENROUTER_API_KEY` from `hermes chat` | Hermes resolved an upstream that wants a key. | Set `model.provider: custom` (not `vllm`) and `model.api_key: dummy`. |
| Background `nohup` / `setsid` over SSH doesn't survive session close | Inherited TTY signals. | Use `tmux new-session -d -s vllm '<cmd>'` instead (works reliably). |
