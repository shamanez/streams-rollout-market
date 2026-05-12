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

log() { printf '\033[1;34m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[bootstrap]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[bootstrap]\033[0m %s\n' "$*" >&2; exit 2; }

# 1. Directory layout ----------------------------------------------------------
log "Materializing $CKPT_ROOT/{qwen3-30b-a3b,qwen3-32b}/{hf,megatron} ..."
mkdir -p "$CKPT_ROOT"/qwen3-30b-a3b/megatron
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

ensure_hf_snapshot "$DENSE_MODEL" "$CKPT_ROOT/qwen3-32b/hf"     || true
ensure_hf_snapshot "$MOE_MODEL"   "$CKPT_ROOT/qwen3-30b-a3b/hf" || true

# 3. vLLM venv -----------------------------------------------------------------
log "Checking vLLM venv at $VENV ..."
if [[ ! -d "$VENV" ]]; then
  warn "  venv missing. Install with:"
  warn "    python3.12 -m venv $VENV && source $VENV/bin/activate && \\"
  warn "      pip install vllm==0.20.2 transformers requests"
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
log "  - HF symlinks resolve (above)"
log "  - vLLM venv exists ($VENV)"
log "  - docker + slimerl/slime:latest pulled"
log "  - Megatron distcp release/ dirs are populated"
