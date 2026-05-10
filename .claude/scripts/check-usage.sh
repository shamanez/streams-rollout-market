#!/usr/bin/env bash
# Usage awareness script.
# Claude Code doesn't expose a usage API directly, but we can track
# our own session metrics to estimate token consumption.
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
EVIDENCE_DIR="$ROOT/.claude/evidence"
mkdir -p "$EVIDENCE_DIR"
USAGE_LOG="$EVIDENCE_DIR/usage_log.jsonl"

# Count tool calls this session (proxy for token usage)
TOOL_CALLS=0
if [ -f "$USAGE_LOG" ]; then
  TOOL_CALLS=$(wc -l < "$USAGE_LOG" | tr -d ' ')
fi

# Count evidence entries (tests run)
EVIDENCE_ENTRIES=0
if [ -f "$EVIDENCE_DIR/evidence_log.jsonl" ]; then
  EVIDENCE_ENTRIES=$(wc -l < "$EVIDENCE_DIR/evidence_log.jsonl" | tr -d ' ')
fi

# Count commits this session
COMMITS_TODAY=$(git log --since="midnight" --oneline 2>/dev/null | wc -l | tr -d ' ')

echo "=== Session Usage Estimate ==="
echo "Tool calls logged: $TOOL_CALLS"
echo "Test evidence entries: $EVIDENCE_ENTRIES"
echo "Commits today: $COMMITS_TODAY"
echo ""
echo "--- Token Conservation Tips ---"
echo "- Sonnet sub-agents: use sparingly (may be at limit)"
echo "- Prefer /codex-review over Claude evaluator for reviews"
echo "- Use haiku test-runner for quick test checks"
echo "- Batch related changes into single commits"

# Log this check
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
echo "{\"timestamp\":\"$TIMESTAMP\",\"tool_calls\":$TOOL_CALLS,\"evidence\":$EVIDENCE_ENTRIES,\"commits\":$COMMITS_TODAY}" >> "$USAGE_LOG"
