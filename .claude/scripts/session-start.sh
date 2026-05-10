#!/usr/bin/env bash
# SessionStart hook: inject context into a fresh Claude Code session.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

echo "=== SESSION START ==="
echo ""

echo "--- Git status ---"
git -C "$ROOT" status --short 2>/dev/null || echo "Not a git repo"
echo ""

echo "--- Recent commits ---"
git -C "$ROOT" log --oneline -5 2>/dev/null || echo "No commits"
echo ""

echo "--- Test status ---"
cd "$ROOT" && python -m pytest -q --tb=no 2>&1 | tail -3 || echo "Tests not runnable"
echo ""

if [ -f "$ROOT/PROGRESS.md" ]; then
  echo "--- PROGRESS.md ---"
  cat "$ROOT/PROGRESS.md"
  echo ""
fi

if [ -f "$ROOT/STEER.md" ]; then
  echo "--- STEER.md (operator redirect active) ---"
  cat "$ROOT/STEER.md"
  echo ""
fi

if [ -f "$ROOT/AGENT_STOP" ]; then
  echo "!!! AGENT_STOP is active. Do not proceed. Update PROGRESS.md, commit, and stop. !!!"
fi

echo "=== END SESSION START ==="
