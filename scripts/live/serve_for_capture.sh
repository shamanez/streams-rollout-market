#!/usr/bin/env bash
# Launch vLLM with routed_experts capture enabled at the engine level.
#
# Cycle 3 v2 (probe-at-rollout): this is the serve script the laptop-side
# logprob_capture_proxy.py talks to over the SSH tunnel. The single load-
# bearing difference vs ~/serve_qwen_for_agent.sh is the
# --enable-return-routed-experts flag, which is the only toggle required
# (per https://docs.vllm.ai/en/latest/training/routed_experts_replay/).
#
# Dense vLLM serves accept the flag and ignore it silently; MoE serves
# emit routed_experts on each response.
#
# Usage on the spot host:
#     MODEL=Qwen/Qwen3-32B       bash ~/serve_for_capture.sh   # Dense bf16
#     MODEL=Qwen/Qwen3-32B-FP8   bash ~/serve_for_capture.sh   # Dense fp8
#     MODEL=Qwen/Qwen3-30B-A3B   TP=2 bash ~/serve_for_capture.sh   # MoE bf16
#     MODEL=Qwen/Qwen3-30B-A3B-FP8 TP=2 bash ~/serve_for_capture.sh   # MoE fp8
#
# Connect from the laptop:
#     ssh -L 8000:localhost:8000 my-vllm-spot-instance -N &
#     python scripts/live/logprob_capture_proxy.py \
#         --upstream http://localhost:8000 --port 8001 \
#         --sidecar /tmp/hermes_probes.jsonl &

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
PORT="${PORT:-8000}"
TOOL_PARSER="${TOOL_PARSER:-hermes}"
DTYPE="${DTYPE:-auto}"
TP="${TP:-4}"
MAX_LEN="${MAX_LEN:-8192}"
MEM_FRAC="${MEM_FRAC:-0.85}"
# Qwen3's native max_position_embeddings is 40960. To go above we
# need YaRN rope scaling. Set ROPE_YARN=1 (default ON when MAX_LEN
# > 40960) to enable it; vLLM applies YaRN extension at serve time.
ROPE_YARN="${ROPE_YARN:-}"
if [[ -z "$ROPE_YARN" ]]; then
    if (( MAX_LEN > 40960 )); then ROPE_YARN=1; else ROPE_YARN=0; fi
fi
# Auto-enable routed-experts capture on MoE Qwen3 models. Pass
# ROUTED_EXPERTS=0 to force OFF, or =1 to force ON. Dense models will
# crash on init with "Qwen3Config has no attribute num_experts_per_tok"
# if the flag is enabled, so we keep this strictly MoE-only.
if [[ -z "${ROUTED_EXPERTS:-}" ]]; then
    case "$MODEL" in
        *A3B*|*MoE*|*moe*) ROUTED_EXPERTS=1 ;;
        *)                 ROUTED_EXPERTS=0 ;;
    esac
fi

source ~/rmenv/bin/activate
export HF_HOME=/home/ubuntu/hf-cache

ROUTED_FLAG=""
if [[ "$ROUTED_EXPERTS" == "1" ]]; then
    ROUTED_FLAG="--enable-return-routed-experts"
fi
ROPE_FLAG=""
if [[ "$ROPE_YARN" == "1" ]]; then
    # vLLM 0.20+ removed --rope-scaling; rope settings go via --hf-overrides.
    ROPE_FLAG=(--hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}')
    # vLLM validates max_model_len against the model's native max_position_embeddings
    # BEFORE applying the HF override. The env var lets the YaRN-extended limit through.
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
fi
echo "[serve-capture] model=$MODEL parser=$TOOL_PARSER tp=$TP dtype=$DTYPE port=$PORT routed_experts=$ROUTED_EXPERTS rope_yarn=$ROPE_YARN max_len=$MAX_LEN allow_long=${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-0}"
exec vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$MEM_FRAC" \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_PARSER" \
    $ROUTED_FLAG \
    "${ROPE_FLAG[@]}" \
    --host 0.0.0.0 \
    --port "$PORT"
