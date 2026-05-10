#!/usr/bin/env bash
# Operator tool: write STEER.md to redirect the agent mid-session.
# Usage: bash .claude/scripts/steer.sh "Switch to Phase 2.1 next"
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

if [ $# -eq 0 ]; then
  echo "Usage: bash .claude/scripts/steer.sh \"<instruction>\""
  exit 1
fi

cat > "$ROOT/STEER.md" <<EOF
# STEER.md -- Operator Redirect

## Instruction
$1

Updated at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "STEER.md written. Agent will read on next session start or cycle."
echo "To clear: rm $ROOT/STEER.md"
