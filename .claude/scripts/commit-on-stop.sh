#!/usr/bin/env bash
# Stop hook: auto-commit tracked changes so nothing is lost between sessions.
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

# Only commit if there are uncommitted changes to tracked files
if ! git diff --quiet HEAD 2>/dev/null || ! git diff --cached --quiet HEAD 2>/dev/null; then
  git add -u
  git commit -m "Auto-commit: session end checkpoint

See PROGRESS.md for current state." 2>/dev/null || true
  echo "Auto-committed session work."
else
  echo "No uncommitted changes to commit."
fi
