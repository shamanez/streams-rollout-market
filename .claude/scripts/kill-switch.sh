#!/usr/bin/env bash
# Operator tool: create AGENT_STOP file to halt the agent.
# Usage: bash .claude/scripts/kill-switch.sh
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
touch "$ROOT/AGENT_STOP"
echo "AGENT_STOP created at $ROOT/AGENT_STOP"
echo "Agent will halt on next tool call."
echo "To resume: rm $ROOT/AGENT_STOP"
