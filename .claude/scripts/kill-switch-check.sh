#!/usr/bin/env bash
# PreToolUse hook: block all tool calls if AGENT_STOP file exists.
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

if [ -f "$ROOT/AGENT_STOP" ]; then
  cat <<'STOP_MSG' >&2
AGENT_STOP file detected. You MUST stop immediately.
1. Update PROGRESS.md with current state
2. Commit all work (git add -u && git commit)
3. Stop the session
STOP_MSG
  exit 2
fi

exit 0
