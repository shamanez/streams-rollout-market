#!/usr/bin/env bash
# Verify that test evidence exists before allowing a task to be marked complete.
# Usage: bash .claude/scripts/verify-gate.sh
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
EVIDENCE_LOG="$ROOT/.claude/evidence/evidence_log.jsonl"

if [ ! -f "$EVIDENCE_LOG" ]; then
  echo "FAIL: No evidence log found. Run 'pytest -q' before marking any task complete."
  exit 1
fi

# Check for recent pytest evidence (last 10 entries)
RECENT=$(tail -10 "$EVIDENCE_LOG" | grep -c "pytest" || true)
if [ "$RECENT" -eq 0 ]; then
  echo "FAIL: No recent pytest evidence. Run 'pytest -q' before marking complete."
  exit 1
fi

echo "PASS: Test evidence found ($RECENT recent pytest entries)."
exit 0
