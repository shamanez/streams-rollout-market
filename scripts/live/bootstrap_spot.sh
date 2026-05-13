#!/usr/bin/env bash
# Bootstrap a fresh spot instance for the streams-rollout-market
# pipeline. Idempotent: safe to re-run.
#
# Run THIS script on the spot host (not from the laptop). Typical
# laptop invocation:
#
#   rsync -av --delete-excluded \
#       --exclude='__pycache__' --exclude='*.pyc' \
#       scripts/live/ my-vllm-spot-instance:~/streams-rollout-market/scripts/live/
#   ssh my-vllm-spot-instance 'bash ~/streams-rollout-market/scripts/live/bootstrap_spot.sh'
#
# What it does:
#   1. Make sure the directory layout exists:
#        $CKPT_ROOT/{qwen3-30b-a3b,qwen3-32b}/{hf,megatron}/
#        $CKPT_ROOT/logs/
#   2. If the HF cache has the model snapshots, point the hf/ symlinks
#      at the canonical snapshot dirs. If not, optionally fetch them.
#   3. Verify the vLLM venv ($VENV) exists, or print the install command.
#   4. Verify docker + slimerl/slime image, or print pull command.
#   5. Report what's ready and what's still needed.
#
# It does NOT push the 117GB Megatron distcp checkpoints — those live
# on the laptop ($CKPT_BACKUP_LAPTOP) and are restored separately via:
#
#   bash scripts/live/restore_megatron_checkpoints.sh
#
# Env vars (with defaults):
#   CKPT_ROOT          $HOME/checkpoint
#   HF_HOME            $HOME/hf-cache
#   VENV               $HOME/rmenv
#   FETCH_HF           0  (set to 1 to auto-download HF models if missing)

set -euo pipefail

CKPT_ROOT="${CKPT_ROOT:-$HOME/checkpoint}"
HF_HOME="${HF_HOME:-$HOME/hf-cache}"
VENV="${VENV:-$HOME/rmenv}"
FETCH_HF="${FETCH_HF:-0}"

DENSE_MODEL="Qwen/Qwen3-32B"
MOE_MODEL="Qwen/Qwen3-30B-A3B"
MOE_FP8_MODEL="Qwen/Qwen3-30B-A3B-FP8"
VENV_DEV="${VENV_DEV:-$HOME/rmenv-dev}"     # patched HTTP-shim venv (cycle 3)
HF_FLAT_PINNED="${HF_FLAT_PINNED:-$HOME/hf-flat-qwen3-30b-a3b-bf16}"  # cycle FP8

log() { printf '\033[1;34m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[bootstrap]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[bootstrap]\033[0m %s\n' "$*" >&2; exit 2; }

# 1. Directory layout ----------------------------------------------------------
log "Materializing $CKPT_ROOT/{qwen3-30b-a3b,qwen3-32b}/{hf,megatron} ..."
mkdir -p "$CKPT_ROOT"/qwen3-30b-a3b/megatron
mkdir -p "$CKPT_ROOT"/qwen3-30b-a3b-fp8
mkdir -p "$CKPT_ROOT"/qwen3-32b/megatron
mkdir -p "$CKPT_ROOT"/logs

# 2. HF cache --------------------------------------------------------------------
log "Checking HF cache at $HF_HOME ..."
ensure_hf_snapshot() {
  local repo="$1"        # e.g. Qwen/Qwen3-32B
  local link_target="$2" # e.g. $CKPT_ROOT/qwen3-32b/hf
  local hub_repo
  hub_repo="$HF_HOME/hub/models--${repo/\//--}"
  local snap
  if [[ -d "$hub_repo/snapshots" ]]; then
    snap=$(ls -dt "$hub_repo/snapshots"/*/ 2>/dev/null | head -1)
    snap="${snap%/}"
  else
    snap=""
  fi

  if [[ -z "$snap" ]]; then
    if [[ "$FETCH_HF" = "1" ]]; then
      log "  $repo not cached. Fetching ($FETCH_HF=1) ..."
      if [[ ! -d "$VENV" ]]; then
        fail "  cannot fetch — venv $VENV missing. See section 3 below."
      fi
      # shellcheck disable=SC1091
      source "$VENV/bin/activate"
      HF_HOME="$HF_HOME" hf download "$repo"
      snap=$(ls -dt "$hub_repo/snapshots"/*/ 2>/dev/null | head -1)
      snap="${snap%/}"
    else
      warn "  $repo not cached at $hub_repo/snapshots/. To fetch:"
      warn "    FETCH_HF=1 bash $0    # or"
      warn "    source $VENV/bin/activate && HF_HOME=$HF_HOME hf download $repo"
      return 1
    fi
  fi
  log "  $repo snapshot: $snap"
  ln -sfn "$snap" "$link_target"
  log "  symlink: $link_target -> $snap"
  return 0
}

ensure_hf_snapshot "$DENSE_MODEL"   "$CKPT_ROOT/qwen3-32b/hf"        || true
ensure_hf_snapshot "$MOE_MODEL"     "$CKPT_ROOT/qwen3-30b-a3b/hf"    || true
ensure_hf_snapshot "$MOE_FP8_MODEL" "$CKPT_ROOT/qwen3-30b-a3b-fp8/hf" || true

# 3. vLLM venvs ----------------------------------------------------------------
# Two venvs on the spot:
#   $VENV      — production vllm (Megatron MoE conversion + FSDP teacher-force
#                can use either; cycle-FP8 uses $VENV_DEV everywhere)
#   $VENV_DEV  — vllm + the routed_experts HTTP-shim patch (cycle 3+ rollouts)
log "Checking vLLM venv at $VENV ..."
if [[ ! -d "$VENV" ]]; then
  warn "  $VENV missing. Install with:"
  warn "    python3.12 -m venv $VENV && source $VENV/bin/activate && \\"
  warn "      pip install vllm==0.20.2 transformers requests"
fi

log "Checking patched dev venv at $VENV_DEV ..."
if [[ ! -d "$VENV_DEV" ]]; then
  warn "  $VENV_DEV missing — cycle FP8 / router_replay needs this. Build with:"
  warn "    python3.12 -m venv $VENV_DEV && source $VENV_DEV/bin/activate && \\"
  warn "      pip install vllm==0.20.2 transformers requests huggingface_hub && \\"
  warn "      python ~/streams-rollout-market/scripts/live/patch_vllm_routed_experts_http.py \\"
  warn "        \"\$(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')\""
elif source "$VENV_DEV/bin/activate" 2>/dev/null && python -c "
from vllm.entrypoints.openai.completion.protocol import CompletionResponseChoice
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponseChoice
import sys
missing = [c.__name__ for c in (CompletionResponseChoice, ChatCompletionResponseChoice)
           if 'routed_experts' not in c.model_fields]
sys.exit(0 if not missing else 1)
" 2>/dev/null; then
  log "  $VENV_DEV looks patched (routed_experts present on both ResponseChoice classes)"
else
  warn "  $VENV_DEV exists but routed_experts shim missing — apply patch:"
  warn "    source $VENV_DEV/bin/activate && python \\"
  warn "      ~/streams-rollout-market/scripts/live/patch_vllm_routed_experts_http.py \\"
  warn "      \"\$(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')\""
fi

# 3b. Pinned dereferenced HF dir (cycle FP8) ----------------------------------
# The Megatron MoE container can't follow the HF snapshot's relative symlinks
# across a bind-mount. The launcher dereferences with `cp -rL` into a temp dir
# on every run unless HF_FLAT_REUSE points at a pinned copy. Pre-build one so
# back-to-back launches stay fast.
log "Checking pinned dereffed HF dir at $HF_FLAT_PINNED ..."
if [[ ! -f "$HF_FLAT_PINNED/config.json" ]]; then
  bf16_snap="$(readlink -f "$CKPT_ROOT/qwen3-30b-a3b/hf" 2>/dev/null || true)"
  if [[ -f "$bf16_snap/config.json" ]]; then
    log "  populating $HF_FLAT_PINNED from $bf16_snap (one-shot cp -rL, ~60 s)"
    mkdir -p "$HF_FLAT_PINNED"
    cp -rL "$bf16_snap"/. "$HF_FLAT_PINNED"/
    log "  done. $HF_FLAT_PINNED is ready for HF_FLAT_REUSE=$HF_FLAT_PINNED on Megatron MoE launches."
  else
    warn "  bf16 HF snapshot not found — populate after \`ensure_hf_snapshot\` succeeds."
  fi
else
  log "  $HF_FLAT_PINNED already populated."
fi

# 4. Docker + slimerl/slime image ----------------------------------------------
log "Checking docker + slimerl/slime:latest ..."
if ! command -v docker >/dev/null 2>&1; then
  warn "  docker not on PATH."
elif ! docker image inspect slimerl/slime:latest >/dev/null 2>&1; then
  warn "  slimerl/slime:latest not pulled. Pull with:"
  warn "    docker pull slimerl/slime:latest"
fi

# 5. Hermes Agent CLI -----------------------------------------------------------
log "Checking Hermes-Agent CLI at ~/.local/bin/hermes ..."
if [[ ! -x "$HOME/.local/bin/hermes" ]]; then
  warn "  hermes CLI missing. See scripts/live/HERMES_INSTALL.md"
fi

# 6. Megatron distcp checkpoints -----------------------------------------------
log "Checking Megatron distcp checkpoints ..."
for sub in qwen3-32b qwen3-30b-a3b; do
  rel="$CKPT_ROOT/$sub/megatron/release"
  if [[ -d "$rel" ]] && [[ -n "$(ls -A "$rel" 2>/dev/null)" ]]; then
    log "  $sub/megatron/release: $(du -sh "$rel" | cut -f1)"
  else
    warn "  $sub/megatron is empty. Either:"
    warn "    a) re-run conversion: bash scripts/live/megatron_${sub//-/_}_launch.sh"
    warn "    b) restore from laptop: bash scripts/live/restore_megatron_checkpoints.sh"
  fi
done

# 7. Final report --------------------------------------------------------------
log "Layout:"
ls -la "$CKPT_ROOT"/qwen3-30b-a3b/ "$CKPT_ROOT"/qwen3-32b/ 2>/dev/null

log "Done. Pipeline-ready when:"
log "  - HF symlinks resolve for bf16, fp8, dense (above)"
log "  - Both vLLM venvs exist: $VENV (prod) and $VENV_DEV (HTTP shim)"
log "  - $HF_FLAT_PINNED exists (Megatron MoE launcher's HF_FLAT_REUSE target)"
log "  - docker + slimerl/slime:latest pulled"
log "  - Megatron distcp release/ dirs are populated"
