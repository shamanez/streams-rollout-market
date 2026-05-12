#!/usr/bin/env bash
# Push Megatron torch-dist checkpoints from the laptop backup back to a
# fresh spot host. Pairs with bootstrap_spot.sh — after the spot is
# bootstrapped, this populates $CKPT_ROOT/<model>/megatron/release/.
#
# Run on the LAPTOP:
#   bash scripts/live/restore_megatron_checkpoints.sh
#
# Env (defaults):
#   SPOT                my-vllm-spot-instance
#   CKPT_BACKUP_LAPTOP  $HOME/Documents/checkpoints-from-spot
#   CKPT_ROOT_SPOT      \$HOME/checkpoint    (literal — expanded on spot)
#
# Source layout on the laptop (each dir contains a `release/` with the
# distcp shards):
#   $CKPT_BACKUP_LAPTOP/qwen3-30b-a3b-megatron/
#   $CKPT_BACKUP_LAPTOP/qwen3-32b-megatron/

set -euo pipefail

SPOT="${SPOT:-my-vllm-spot-instance}"
CKPT_BACKUP_LAPTOP="${CKPT_BACKUP_LAPTOP:-$HOME/Documents/checkpoints-from-spot}"

log() { printf '\033[1;34m[restore]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[restore]\033[0m %s\n' "$*" >&2; exit 2; }

if [[ ! -d "$CKPT_BACKUP_LAPTOP" ]]; then
  fail "Laptop backup $CKPT_BACKUP_LAPTOP not found. Did you rsync from the prior spot?"
fi

push_one() {
  local src="$1"   # laptop path
  local dst="$2"   # spot path (literal, $HOME expanded on spot)
  if [[ ! -d "$src" ]]; then
    log "skip: $src missing"
    return
  fi
  local size
  size=$(du -sh "$src" | cut -f1)
  log "rsync $src ($size) -> $SPOT:$dst"
  ssh "$SPOT" "mkdir -p $dst"
  rsync -aP --partial "$src/" "$SPOT:$dst/"
}

push_one "$CKPT_BACKUP_LAPTOP/qwen3-30b-a3b-megatron" '$HOME/checkpoint/qwen3-30b-a3b/megatron'
push_one "$CKPT_BACKUP_LAPTOP/qwen3-32b-megatron"     '$HOME/checkpoint/qwen3-32b/megatron'

log "Verifying on spot:"
ssh "$SPOT" 'for s in qwen3-30b-a3b qwen3-32b; do
  rel="$HOME/checkpoint/$s/megatron/release"
  if [[ -d "$rel" ]] && [[ -n "$(ls -A "$rel" 2>/dev/null)" ]]; then
    printf "  %-16s  %s\n" "$s" "$(du -sh "$rel" | cut -f1)"
  else
    printf "  %-16s  EMPTY\n" "$s"
  fi
done'

log "Done."
