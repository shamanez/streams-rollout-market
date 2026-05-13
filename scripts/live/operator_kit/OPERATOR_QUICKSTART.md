# Operator Quickstart — Qwen3-30B-A3B + routed_experts HTTP shim

Hi. You're hosting a vLLM endpoint for the streams-rollout-market
project. The coordinator (us) is going to call your endpoint over
HTTP and read four fields off every chat-completion response:

* `prompt_token_ids` (vLLM emits this by default)
* `choices[0].token_ids` (vLLM emits this by default)
* `choices[0].logprobs.token_logprobs` (vLLM emits this when the
  request has `logprobs=True`)
* **`choices[0].routed_experts`** ← this is the field you need to
  enable. vLLM 0.20.2 has the engine-side data but does not
  serialize it to the OpenAI HTTP response unless you apply the
  surgical patch in `patch_vllm_routed_experts_http.py`.

If you can run vLLM with that patch applied, the coordinator can
do everything else from one central place (no scripts running
inside your environment). The data fields above carry the
marketplace's rollout-integrity probes.

## Prerequisites

* A GPU host (≥ 96 GB VRAM total recommended — 4× L40S or
  equivalent). Single-host TP works fine; we serve TP=4 by default.
* CUDA driver matching vLLM 0.20.2's torch 2.11+cu130 wheel.
* Python 3.12 + `python3.12-venv`.
* `~80 GB free disk` for the model weights (Qwen3-30B-A3B in bf16
  is ~60 GB; the FP8 variant is ~31 GB; you only need one).
* Outbound network for the HuggingFace download.

## The three commands

```bash
# 1. Create a clean venv and install vllm
python3.12 -m venv ~/rmenv-dev
source ~/rmenv-dev/bin/activate
pip install vllm==0.20.2 transformers requests huggingface_hub

# 2. Apply the routed_experts HTTP shim to the installed vllm
python patch_vllm_routed_experts_http.py \
    "$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"

# 3. Source the dev venv and start the server
#    (downloads Qwen3-30B-A3B from HuggingFace on first run, ~60 GB)
bash vllm_serve_moe_dev.sh
```

The server listens on `0.0.0.0:8000`. Make port 8000 reachable to
the coordinator however your hosting setup expects (SSH tunnel,
public IP, internal load balancer — your call). The coordinator
will need a base URL.

## Why each step is needed

| Step | Why |
|------|-----|
| Separate venv `~/rmenv-dev/` | The patch modifies vLLM's source files in `site-packages/vllm/entrypoints/openai/`. Keeping it in a dedicated venv lets you upgrade or roll back without polluting other Python installs on the host. |
| `vllm==0.20.2` pinned | The patch's `_replace_once` line-anchors are hand-tuned for the 0.20.2 wheel. Newer releases need a new patch. |
| Apply the patch | Adds `routed_experts: list[list[list[int]]]` to `CompletionResponseChoice` and `ChatCompletionResponseChoice`, and threads `output.routed_experts.tolist()` into the three response constructors in the wheel's `entrypoints/openai/` shim. Without it, the gen-side router trace stays inside vLLM and never reaches the HTTP response. |
| `vllm_serve_moe_dev.sh` | Sources your venv, sanity-checks the patch was applied, and starts vLLM with the load-bearing flags: `--enable-return-routed-experts` (engine emits expert IDs), `--enable-expert-parallel` (keeps the trace shape valid under TP=4), `--no-async-scheduling` (required when capturing routing), `--enable-auto-tool-choice --tool-call-parser hermes` (so tool-calling tasks work). |

## Verification (run after step 3)

After `/v1/models` returns 200, send one curl:

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3-30B-A3B","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}' \
| python3 -c '
import sys, json
r = json.load(sys.stdin)
choice = r["choices"][0]
print("content:", repr(choice["message"]["content"]))
re = choice.get("routed_experts")
print("routed_experts present:", re is not None)
if re:
    print("shape: [tokens=%d, layers=%d, top_k=%d]" % (
        len(re), len(re[0]), len(re[0][0])))
'
```

You should see `routed_experts present: True` and a shape with
`layers=48, top_k=8` for Qwen3-30B-A3B. If `present` is False, the
patch did not apply cleanly; re-run step 2 and the script should
report the patched files.

## Knobs

The serve script reads these env vars (defaults shown):

| Env | Default | Purpose |
|-----|---------|---------|
| `MODEL` | `Qwen/Qwen3-30B-A3B` | HF model id. Override with `Qwen/Qwen3-30B-A3B-FP8` to serve FP8 instead of bf16 (uses ~half the VRAM for weights). |
| `HOST` | `0.0.0.0` | Listen address. |
| `PORT` | `8000` | Listen port. |
| `TP_SIZE` | `4` | Tensor-parallel rank count. Lower this if you have fewer GPUs (e.g. `TP_SIZE=2` for 2 GPUs). |
| `MAX_LEN` | `40960` | Max context length. |
| `GPU_UTIL` | `0.85` | vLLM's gpu_memory_utilization fraction. |
| `LOG` | `~/vllm-logs/moe_bf16_serve.log` | Server log file (append). |
| `RMENV_DEV` | `$HOME/rmenv-dev` | Path to the patched venv. |
| `HF_HOME` | `$HOME/hf-cache` | HuggingFace cache root. |

## Costs and constraints we'd like you to be aware of

* The patch is read-only at runtime (no live re-patching). If you
  upgrade vLLM, the patch needs to be re-applied for the new
  version — we'll ship an updated kit.
* The serve script does **not** run any other code in your
  environment. It just starts vLLM with specific CLI flags. No
  telemetry, no callbacks.
* Logprobs and routed_experts add a small per-token serialization
  cost on the response. For a 100-token generation it adds maybe
  ~50 KB to the response payload.

If anything in this quickstart breaks, please share the output and
we'll iterate on the kit.
