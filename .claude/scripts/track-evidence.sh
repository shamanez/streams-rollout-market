#!/usr/bin/env bash
# PostToolUse hook for Bash: record test/lint execution evidence.
# This is the default-FAIL contract — features aren't complete until evidence exists.
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
EVIDENCE_DIR="$ROOT/.claude/evidence"
mkdir -p "$EVIDENCE_DIR"

# Read tool input from stdin (hook receives JSON)
INPUT=$(cat)

# Extract the command from the hook input
COMMAND=$(echo "$INPUT" | grep -oE '"command"\s*:\s*"[^"]*"' | head -1 | sed 's/.*"command"\s*:\s*"//;s/"$//' 2>/dev/null || echo "")

# Only track pytest and ruff runs
if echo "$COMMAND" "$INPUT" | grep -qE "(pytest|ruff)"; then
  TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
  echo "{\"timestamp\":\"$TIMESTAMP\",\"command\":\"$COMMAND\"}" \
    >> "$EVIDENCE_DIR/evidence_log.jsonl"
fi

exit 0
